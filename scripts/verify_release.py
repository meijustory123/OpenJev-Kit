"""Extract a release elsewhere and test its isolated runtime, CPU and launcher."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def request(url, token=None, body=None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["X-OpenJev-Token"] = token
    payload = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    with OPENER.open(urllib.request.Request(url, data=payload, headers=headers), timeout=90) as response:
        return json.load(response)


def environment(device):
    env = os.environ.copy()
    env.update(PATH=str(Path(env["SYSTEMROOT"]) / "System32"),
               PYTHONHOME="Z:/no-system-python", PYTHONPATH="Z:/no-system-packages",
               HTTP_PROXY="http://127.0.0.1:9", HTTPS_PROXY="http://127.0.0.1:9",
               HF_HUB_OFFLINE="1", CUDA_VISIBLE_DEVICES="-1" if device == "cpu" else "0")
    return env


def check_model(package, device, use_cmd):
    env = environment(device)
    env["HF_HOME"] = str(package / "outputs/hf-cache")
    cmd = str(Path(env["SYSTEMROOT"]) / "System32/cmd.exe")
    launch = ([cmd, "/d", "/c", "启动决策模型.cmd"] if use_cmd else
              [str(package / "runtime/pythonw.exe"), "-I", "-X", "utf8",
               str(package / "scripts/portable_entry.py"), "--no-browser"])
    info = None
    log_path = package.parent / f"verification-{device}.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(launch, cwd=package, env=env, stdout=log, stderr=log,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                record = package / "outputs/webapp/server.json"
                if record.exists():
                    try:
                        candidate = json.loads(record.read_text(encoding="utf-8"))
                        url = f"http://127.0.0.1:{int(candidate['port'])}"
                        health = request(url + "/health")
                        if health.get("workspace") != str(package):
                            raise RuntimeError("Server belongs to another workspace")
                        info = dict(candidate, url=url)
                        state = request(url + "/api/status")
                        if state["status"] == "error":
                            raise RuntimeError(state["message"])
                        if state["status"] == "ready":
                            break
                    except (OSError, ValueError):
                        pass
                if process.poll() not in {None, 0}:
                    raise RuntimeError(f"Launcher failed, see {log_path}")
                time.sleep(1)
            else:
                raise RuntimeError(f"Model did not become ready, see {log_path}")
            assert state["device"] == device, state
            assert state["checkpoint"] == "checkpoint-001800", state
            assert state["max_length"] == 8192
            with OPENER.open(info["url"] + "/", timeout=10) as page:
                assert page.status == 200 and b"/assets/app.js" in page.read()
            raw = json.loads((package / "examples/request.json").read_text(encoding="utf-8"))
            response = request(info["url"] + "/v1/systemone", token=info["token"], body=raw)
            from openjev.benchmark import prediction_vectors
            from openjev.protocol import parse_request
            prediction_vectors(parse_request(raw), response)
            assert {a["type"] for a in response["answers"].values()} == {"choice", "score", "noul"}
            assert all("confidence" not in a for a in response["answers"].values())
            assert response["usage"]["input_tokens"] > 0
            result = {"device": device, "checkpoint": state["checkpoint"], "max_length": state["max_length"],
                      "webpage_http_status": 200, "real_prediction": response, "cmd_launcher": use_cmd}
            print(f"{device}: checkpoint 1800 loaded; Choice, Score and Noul predictions passed", flush=True)
            stopped = subprocess.run([cmd, "/d", "/c", "关闭决策模型.cmd"], cwd=package, env=env,
                                     creationflags=subprocess.CREATE_NO_WINDOW, timeout=30)
            assert stopped.returncode == 0
            for _ in range(30):
                if not (package / "outputs/webapp/server.json").exists():
                    break
                time.sleep(1)
            else:
                raise RuntimeError("Stop launcher did not shut down the packaged server")
            result["stop_launcher"] = True
            return result
        finally:
            if info and (package / "outputs/webapp/server.json").exists():
                request(info["url"] + "/api/shutdown", token=info["token"], body={})
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.terminate()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    destination = ROOT / "outputs/package-verification" / uuid.uuid4().hex[:8] / "解压 验证"
    destination.mkdir(parents=True)
    print(f"Extracting and checking all files in {destination}", flush=True)
    verified = 0
    last_progress = time.monotonic()
    with zipfile.ZipFile(args.archive) as archive:
        names = archive.namelist()
        assert len(names) == len(set(names))
        manifest_name = next(n for n in names if n.endswith("/package-manifest.json"))
        manifest = json.loads(archive.read(manifest_name))
        package = destination / manifest["name"]
        assert len(names) == len(manifest["files"]) + 1
        benchmark = json.loads((ROOT / "reports/2026-09-22-checkpoints-vs-jev.json").read_text(encoding="utf-8"))
        reference = next(r["signature"]["weights_and_config_sha256"] for r in benchmark["runs"]
                         if r["signature"].get("checkpoint") == "checkpoint-001800")
        for relative, fingerprint in reference.items():
            entry = "outputs/decision-full/checkpoint-001800/" + relative.replace("\\", "/")
            assert manifest["files"][entry]["sha256"] == fingerprint, entry
        for name in names:
            target = (destination / name).resolve()
            assert target.is_relative_to(destination.resolve()), name
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            with archive.open(name) as source, target.open("xb") as output:
                for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    digest.update(block)
                    output.write(block)
            if name != manifest_name:
                entry = manifest["files"][target.relative_to(package).as_posix()]
                assert digest.hexdigest() == entry["sha256"] and target.stat().st_size == entry["bytes"], name
                verified += 1
            if time.monotonic() - last_progress >= 15:
                print(f"Verified {verified}/{len(manifest['files'])} files", flush=True)
                last_progress = time.monotonic()
    assert not list(package.rglob("training_state.pt"))
    assert not (package / ".venv").exists()
    info = json.loads(subprocess.check_output([
        str(package / "runtime/python.exe"), "-I", "-c",
        "import sys,json; print(json.dumps({'executable':sys.executable,'paths':sys.path,'version':sys.version.split()[0]}))"
    ], env=environment("cpu"), text=True))
    assert all(Path(p).resolve().is_relative_to(package) for p in info["paths"])
    results = [check_model(package, "cpu", use_cmd=False)]
    if args.gpu:
        results.append(check_model(package, "cuda", use_cmd=True))
    archive_hash = hashlib.sha256()
    with args.archive.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            archive_hash.update(block)
    assert args.archive.with_suffix(".zip.sha256").read_text().split()[0] == archive_hash.hexdigest()
    report = {"verified_at": dt.datetime.now(dt.timezone.utc).isoformat(), "archive": args.archive.name,
              "bytes": args.archive.stat().st_size, "sha256": archive_hash.hexdigest(),
              "all_file_hashes_passed": verified, "python": info["version"],
              "isolated_bundled_runtime": True, "relocated_path_with_unicode_and_spaces": True,
              "system_python_not_used": True, "tests": results}
    report["checkpoint_matches_evaluated_1800"] = True
    target = ROOT / "reports/portable-release-check.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Verified release: {target}", flush=True)


if __name__ == "__main__":
    main()
