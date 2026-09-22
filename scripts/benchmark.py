"""Resumable fixed-test CPU / GPU / official Jev benchmark. Never persist credentials."""
import argparse
import datetime as dt
import getpass
import importlib.metadata
import json
import os
import platform
import sys
import time
from pathlib import Path

from openjev.benchmark import METRIC_DEFINITIONS, METRIC_VERSION, prediction_vectors, read_jsonl, sha256, summarize
from openjev.protocol import parse_request


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def environment():
    result = {"python": platform.python_version(), "platform": platform.platform(),
              "processor": platform.processor(),
              "packages": {name: importlib.metadata.version(name) for name in ["torch", "transformers", "httpx"]}}
    if os.name == "nt":
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            result["processor"] = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    return result


def run(args):
    rows = read_jsonl(args.data)
    if args.limit:
        rows = rows[:args.limit]
    if not rows or len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("Empty data or duplicate sample IDs")
    args.output.mkdir(parents=True, exist_ok=True)
    signature = {"data_sha256": sha256(args.data), "requests": len(rows),
                 "sample_ids": [r["sample_id"] for r in rows], "backend": args.backend,
                 "metric_version": METRIC_VERSION}
    if args.backend in {"cpu", "gpu"}:
        checkpoint = Path(args.checkpoint)
        if not (checkpoint / "COMPLETE").is_file():
            raise ValueError("Incomplete checkpoint")
        files = sorted(p for p in checkpoint.rglob("*") if p.is_file() and p.name != "training_state.pt")
        signature.update(checkpoint=checkpoint.name, device="cpu", dtype="float32", threads=args.threads,
                         micro_batch_size=1, weights_and_config_sha256={str(p.relative_to(checkpoint)): sha256(p) for p in files})
        if args.backend == "gpu":
            import torch
            if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
                raise RuntimeError("GPU evaluation requires CUDA with BF16 support; CPU fallback is disabled")
            signature.update(device="cuda:0", dtype="bfloat16", head_dtype="float32",
                             autocast_dtype="bfloat16", gpu_name=torch.cuda.get_device_name(0),
                             compute_capability=list(torch.cuda.get_device_capability(0)),
                             torch_version=torch.__version__, cuda_version=torch.version.cuda,
                             float32_matmul_precision=torch.get_float32_matmul_precision(),
                             allow_tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                             allow_tf32_cudnn=torch.backends.cudnn.allow_tf32)
    else:
        signature.update(endpoint="https://api.typesafe.ai/v1/systemone", requested_model="jev-latest")
    meta_path = args.output / "run.json"
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if metadata["signature"] != signature:
            raise ValueError("Existing cache belongs to a different dataset/model/configuration")
        metadata["metrics"] = METRIC_DEFINITIONS
        dump(meta_path, metadata)
    else:
        dump(meta_path, {"signature": signature, "started_at": now(), "platform": platform.platform(),
                         "processor": platform.processor(), "environment": environment(), "metrics": METRIC_DEFINITIONS})
    cache = args.output / "predictions.jsonl"
    done = read_jsonl(cache) if cache.exists() else []
    ids = {p["sample_id"] for p in done}
    if len(ids) != len(done) or not ids <= set(signature["sample_ids"]):
        raise ValueError("Invalid or duplicate cached sample IDs")
    pending = [r for r in rows if r["sample_id"] not in ids]
    if args.backend in {"cpu", "gpu"} and pending:
        import torch
        from openjev.inference import Engine
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        engine = Engine(args.checkpoint, device=signature["device"], micro_batch_size=1)
        if args.backend == "cpu":
            if any(p.device.type != "cpu" or p.dtype != torch.float32 for p in engine.model.parameters()):
                raise RuntimeError("CPU FP32 evaluation requires all parameters on CPU in FP32")
        else:
            if any(p.device != torch.device("cuda:0") for p in engine.model.parameters()):
                raise RuntimeError("GPU evaluation requires all parameters on CUDA:0")
            if (any(p.dtype != torch.bfloat16 for p in engine.model.backbone.parameters()) or
                    any(p.dtype != torch.float32 for p in engine.model.head.parameters())):
                raise RuntimeError("GPU evaluation requires BF16 backbone and FP32 decision head")
            torch.cuda.synchronize(0)
        predict = engine.predict
        print(f"Loaded {checkpoint.name} on {signature['device']} {signature['dtype']}, "
              f"threads={args.threads}; pending={len(pending)}", flush=True)
    elif args.backend == "official" and pending:
        import httpx
        key = (sys.stdin.readline().strip() if args.api_key_stdin else
               os.environ.get("TYPESAFE_API_KEY") or getpass.getpass("TypeSafe API key: "))
        if not key:
            raise ValueError("API key is required")
        client = httpx.Client(headers={"Authorization": "Bearer " + key}, timeout=120, follow_redirects=False)
        del key

        def predict(request):
            payload = dict(request, model="jev-latest")
            for attempt in range(7):
                try:
                    result = client.post(signature["endpoint"], json=payload)
                except httpx.TransportError:
                    if attempt == 6:
                        raise RuntimeError("Official API transport failed after seven attempts") from None
                    time.sleep(min(2 ** attempt, 30))
                    continue
                if result.status_code == 200:
                    return result.json()
                if result.status_code in {429, 500, 502, 503, 504, 529} and attempt < 6:
                    time.sleep(min(2 ** (attempt + 1), 60))
                    continue
                # Do not log request headers, credentials, or uncontrolled exception bodies.
                raise RuntimeError(f"Official API returned HTTP {result.status_code}")
        print(f"Official Jev evaluation; pending={len(pending)}", flush=True)
    try:
        with cache.open("a", encoding="utf-8", buffering=1) as stream:
            for row in pending:
                if args.backend == "gpu":
                    torch.cuda.synchronize(0)
                started = time.perf_counter()
                response = predict(row["request"])
                if args.backend == "gpu":
                    torch.cuda.synchronize(0)
                elapsed = time.perf_counter() - started
                try:
                    prediction_vectors(parse_request(row["request"]), response)
                except (ValueError, KeyError, TypeError):
                    dump(args.output / "invalid-response.json", {"sample_id": row["sample_id"], "response": response})
                    raise
                record = {"sample_id": row["sample_id"], "response": response, "seconds": elapsed, "completed_at": now()}
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                done.append(record)
                print(f"{args.backend} {len(done)}/{len(rows)} {row['sample_id']} {elapsed:.2f}s", flush=True)
    finally:
        if args.backend == "official" and pending:
            client.close()
    result = {"completed_at": now(), "signature": signature, "result": summarize(rows, done)}
    dump(args.output / "summary.json", result)
    print(json.dumps(result["result"]["metrics"], ensure_ascii=True), flush=True)


def report(args):
    rows = read_jsonl(args.data)
    runs = []
    for directory in args.runs:
        path = Path(directory)
        meta = json.loads((path / "run.json").read_text(encoding="utf-8"))
        signature = meta["signature"]
        if signature["metric_version"] != METRIC_VERSION:
            raise ValueError("Cached benchmark uses a different metric version")
        if signature["data_sha256"] != sha256(args.data) or signature["sample_ids"] != [r["sample_id"] for r in rows]:
            raise ValueError("Only full runs on the identical test set may be compared")
        runs.append({"signature": signature, "started_at": meta["started_at"],
                     "environment": meta.get("environment", {"platform": meta["platform"], "processor": meta["processor"]}),
                     "finished_at": max(p["completed_at"] for p in read_jsonl(path / "predictions.jsonl")),
                     "result": summarize(rows, read_jsonl(path / "predictions.jsonl"))})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dump(args.output, {"created_at": now(), "report_environment": environment(), "test_data_sha256": sha256(args.data),
                       "reference": "agent_synthetic", "metric_definitions": METRIC_DEFINITIONS, "runs": runs})
    print(str(args.output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--backend", choices=["cpu", "gpu", "official"], required=True)
    run_parser.add_argument("--checkpoint")
    run_parser.add_argument("--threads", type=int, default=8)
    run_parser.add_argument("--data", type=Path, default=Path("data/prepared/test_requests.jsonl"))
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--limit", type=int, default=0, help="For smoke runs only; cannot be included in a full report")
    run_parser.add_argument("--api-key-stdin", action="store_true")
    run_parser.set_defaults(func=run)
    report_parser = sub.add_parser("report")
    report_parser.add_argument("--data", type=Path, default=Path("data/prepared/test_requests.jsonl"))
    report_parser.add_argument("--runs", nargs="+", required=True)
    report_parser.add_argument("--output", type=Path, required=True)
    report_parser.set_defaults(func=report)
    args = parser.parse_args()
    if args.command == "run" and (args.threads < 1 or (args.backend in {"cpu", "gpu"} and not args.checkpoint)):
        parser.error("Local run needs --checkpoint and a positive thread count")
    args.func(args)


if __name__ == "__main__":
    main()
