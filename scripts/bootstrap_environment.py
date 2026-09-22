"""Inspect and repair project dependencies after PowerShell bootstraps Python."""
import argparse
import importlib
import importlib.metadata as metadata
import json
import os
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMPORTS = {
    "torch": "torch", "transformers": "transformers", "accelerate": "accelerate",
    "safetensors": "safetensors", "fastapi": "fastapi", "uvicorn": "uvicorn",
    "httpx": "httpx", "psutil": "psutil", "pytest": "pytest", "filelock": "filelock",
}


def requirements(path=ROOT / "requirements.txt"):
    result = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([\w.-]+)==([^\s#]+)", line)
        if not match:
            raise ValueError(f"Expected a pinned package requirement: {line}")
        result[match[1]] = match[2]
    return result


def torch_compatible(version, variant="auto"):
    versions = {"cpu": "2.10.0+cpu", "cuda": "2.10.0+cu128"}
    return version in (versions.values() if variant == "auto" else [versions[variant]])


def inspect_environment(variant="auto"):
    issues, versions, import_errors = {}, {}, []
    for name, expected in {**requirements(), "torch": None}.items():
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
        if (not torch_compatible(versions[name], variant) if name == "torch" else versions[name] != expected):
            issues[name] = {"installed": versions[name], "expected": expected or f"PyTorch 2.10.0 ({variant})"}
    if not issues:
        for module, distribution in IMPORTS.items():
            try:
                imported = importlib.import_module(module)
                if module == "torch":
                    # CPU-only check, including when CUDA is busy training elsewhere.
                    assert imported.isfinite(imported.ones(2, device="cpu").sum()).item()
                elif module == "transformers":
                    from transformers import Qwen3_5TextConfig, Qwen3_5TextModel  # noqa: F401
            except Exception as exc:
                import_errors.append({"distribution": distribution, "missing_module": getattr(exc, "name", None),
                                      "error": f"{type(exc).__name__}: {exc}"})
        if not import_errors:
            try:
                importlib.import_module("openjev.webapp")
            except Exception as exc:
                import_errors.append({"distribution": None, "missing_module": getattr(exc, "name", None),
                                      "error": f"{type(exc).__name__}: {exc}"})
    return {"ready": not issues and not import_errors, "python": sys.version.split()[0],
            "versions": versions, "issues": issues, "import_errors": import_errors}


def fresh_check(variant):
    process = subprocess.run([sys.executable, "-m", "scripts.bootstrap_environment", "--check", "--torch", variant],
                             cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    for line in reversed(process.stdout.splitlines()):
        if line.startswith('{"ready":'):
            return json.loads(line)
    raise RuntimeError("环境检查进程未能运行：" + (process.stderr or process.stdout)[-2000:])


def select_torch(variant):
    if variant != "auto":
        return variant
    executable = shutil.which("nvidia-smi")
    if executable:
        try:
            result = subprocess.run([executable, "--query-gpu=name", "--format=csv,noheader"],
                                    capture_output=True, timeout=15)
            if result.returncode == 0 and result.stdout.strip():
                return "cuda"
        except (OSError, subprocess.TimeoutExpired):
            pass
    return "cpu"


def run_module(module, *arguments):
    subprocess.run([sys.executable, "-m", module, *map(str, arguments)], cwd=ROOT, check=True)


def repair_specs(state):
    """Reinstall only failing distributions, retaining their installed versions."""
    distributions = metadata.packages_distributions()
    names = set()
    for error in state["import_errors"]:
        if error["distribution"]:
            names.add(error["distribution"])
        if error["missing_module"]:
            names.update(distributions.get(error["missing_module"].split(".")[0], []))
    specs = []
    for name in sorted(names - {"torch"}):
        try:
            specs.append(f"{name}=={metadata.version(name)}")
        except metadata.PackageNotFoundError:
            if name in requirements():
                specs.append(f"{name}=={requirements()[name]}")
    return specs


def ensure_environment(variant, index_url):
    state = fresh_check(variant)
    if state["ready"]:
        print("依赖已齐全，跳过下载和安装。", flush=True)
        return state
    print("检测到缺失或不可用的依赖，开始准备……", flush=True)
    print(json.dumps({"issues": state["issues"], "import_errors": state["import_errors"]}, ensure_ascii=False), flush=True)
    run_module("ensurepip", "--upgrade")
    runtime = ROOT / ".runtime"
    runtime.mkdir(exist_ok=True)
    constraint = runtime / "torch-constraint.txt"
    constraint.write_text("torch==2.10.0\n", encoding="utf-8")

    def pip(*arguments):
        run_module("scripts.install_dependencies", *arguments, "--index-url", index_url, "--constraint", constraint)

    torch_failed = "torch" in state["issues"] or any(e["distribution"] == "torch" for e in state["import_errors"])
    if torch_failed:
        selected = select_torch(variant)
        print(f"正在安装 PyTorch 2.10.0（{'CUDA 12.8' if selected == 'cuda' else 'CPU'}）……", flush=True)
        if selected == "cuda":
            try:
                run_module("scripts.download_torch")
            except subprocess.CalledProcessError:
                print("镜像下载失败，尝试官方源……", flush=True)
                run_module("scripts.download_torch", "--source", "official")
            source = str(ROOT / ".uv-cache/downloads/torch-2.10.0+cu128-cp311-cp311-win_amd64.whl")
        else:
            config = json.loads((ROOT / "configs/bootstrap.json").read_text(encoding="utf-8"))["torch_cpu"]
            source = config["url"] + "#sha256=" + config["sha256"]
        pip(source, "--force-reinstall", "--no-deps")
    pip("-r", ROOT / "requirements.txt")
    result = fresh_check(variant)
    if not result["ready"]:
        specs = repair_specs(result)
        if specs:
            print("正在修复存在版本信息但无法导入的依赖……", flush=True)
            pip("--force-reinstall", "--no-deps", *specs)
            pip("-r", ROOT / "requirements.txt")
            result = fresh_check(variant)
    if not result["ready"]:
        raise RuntimeError("依赖仍未通过检查：" + json.dumps(result, ensure_ascii=False))
    run_module("pip", "check")
    output = ROOT / "outputs/setup"
    output.mkdir(parents=True, exist_ok=True)
    snapshot = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], cwd=ROOT)
    (output / "requirements-installed.txt").write_bytes(snapshot)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--torch", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--index-url", default="https://mirrors.aliyun.com/pypi/simple/")
    args = parser.parse_args()
    os.chdir(ROOT)
    if sys.version_info[:2] != (3, 11) or struct.calcsize("P") != 8:
        raise RuntimeError("需要 Python 3.11 x64")
    if args.check:
        result = inspect_environment(args.torch)
        print(json.dumps(result, ensure_ascii=False))
        return 0 if result["ready"] else 1
    try:
        result = ensure_environment(args.torch, args.index_url)
        (ROOT / ".runtime").mkdir(exist_ok=True)
        (ROOT / ".runtime/environment.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Python 与依赖检查通过，可以启动网页。", flush=True)
        return 0
    except (RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        print(f"环境准备失败：{exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
