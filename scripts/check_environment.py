import importlib.metadata
import json
import platform
import sys
from pathlib import Path


def main():
    import torch
    import psutil
    from openjev.model import model_files
    report = {"python": sys.version, "platform": platform.platform(),
              "ram_gb": round(psutil.virtual_memory().total / 2**30, 2),
              "packages": {}, "cuda_available": torch.cuda.is_available()}
    for name in ["torch", "transformers", "accelerate", "safetensors", "fastapi"]:
        report["packages"][name] = importlib.metadata.version(name)
    if torch.cuda.is_available():
        device = torch.cuda.get_device_properties(0)
        report.update(gpu=device.name, vram_gb=round(device.total_memory / 2**30, 2),
                      cuda=torch.version.cuda, capability=list(torch.cuda.get_device_capability()),
                      bf16_supported=torch.cuda.is_bf16_supported())
        tensor = torch.randn(64, 64, device="cuda", requires_grad=True)
        (tensor @ tensor.T).square().mean().backward()
        torch.cuda.synchronize()
        report["cuda_forward_backward"] = bool(torch.isfinite(tensor.grad).all())
    try:
        report["model_files"] = [str(p) for p in model_files("Qwen3.5-0.8B")]
        report["model_ready"] = True
    except Exception as exc:
        report["model_ready"] = False
        report["model_status"] = str(exc)
    Path("reports").mkdir(exist_ok=True)
    Path("reports/environment.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report.get("cuda_forward_backward"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
