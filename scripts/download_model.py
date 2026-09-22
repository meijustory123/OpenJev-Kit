"""Fetch and verify the published checkpoint before launching the local webpage."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import uuid

from filelock import FileLock

ROOT = Path(__file__).resolve().parents[1]


def project_path(path):
    path = Path(path).resolve()
    if path == ROOT.resolve() or not path.is_relative_to(ROOT.resolve()):
        raise ValueError("模型下载和安装目录必须位于当前项目内")
    return path


def load_manifest(path):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if not re.fullmatch(r"checkpoint-\d{6}", manifest["checkpoint_name"]):
        raise ValueError("无效的检查点名称")
    remote, local = set(), set()
    for entry in manifest["files"]:
        for key, seen in [("remote_path", remote), ("local_path", local)]:
            value = entry[key]
            parts = PurePosixPath(value)
            if (not value or value != parts.as_posix() or value == "." or "\\" in value or ":" in value or parts.is_absolute()
                    or ".." in parts.parts or value in seen):
                raise ValueError("模型文件清单含不安全或重复路径")
            seen.add(value)
        if type(entry["size"]) is not int or entry["size"] < 1 or not re.fullmatch(r"[a-f0-9]{64}", entry["sha256"]):
            raise ValueError("无效的模型文件大小或校验值")
    required = {"COMPLETE", "decision_config.json", "decision_head.safetensors", "backbone/config.json",
                "backbone/model.safetensors", "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"}
    if not required <= local:
        raise ValueError("模型文件清单不完整")
    return manifest


def verified(path, entry):
    if not path.is_file() or path.stat().st_size != entry["size"]:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest() == entry["sha256"]


def fetch_snapshot(manifest, cache):
    # The inference process stays offline; only this download process goes online.
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"
    # Use the SDK's resumable HTTP download on Windows without a native Xet worker.
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    from huggingface_hub import constants, snapshot_download
    from tqdm.auto import tqdm
    constants.HF_HUB_OFFLINE = False
    constants.HF_HUB_DISABLE_XET = True
    constants.HF_HUB_DISABLE_PROGRESS_BARS = False

    class Progress(tqdm):
        def __init__(self, *args, **kwargs):
            kwargs.pop("name", None)
            kwargs.update(disable=False, ascii=True, dynamic_ncols=True, mininterval=0.5)
            super().__init__(*args, **kwargs)

    return snapshot_download(
        repo_id=manifest["repo_id"], revision=manifest["revision"],
        endpoint="https://huggingface.co", local_dir=cache,
        allow_patterns=[entry["remote_path"] for entry in manifest["files"]],
        token=False, max_workers=2, tqdm_class=Progress,
    )


def ensure_model(output=None, manifest_path=None):
    from openjev.webapp import available_checkpoints
    output = project_path(output or ROOT / "outputs/decision-full")
    runtime = project_path(ROOT / ".runtime")
    runtime.mkdir(exist_ok=True)
    with FileLock(runtime / "model-download.lock"):
        checkpoints = available_checkpoints(output)
        if checkpoints:
            print(f"已找到本地模型 {checkpoints[-1].name}，跳过下载。", flush=True)
            return checkpoints[-1]
        manifest = load_manifest(manifest_path or ROOT / "configs/model-download.json")
        name = manifest["checkpoint_name"]
        cache = project_path(ROOT / ".hf-cache" / ("download-" + name))
        cache.mkdir(parents=True, exist_ok=True)
        total = sum(entry["size"] for entry in manifest["files"])
        print(f"本地没有完整模型，正在从 Hugging Face 下载第 1800 步模型（约 {total / 1024**3:.2f} GiB）。", flush=True)
        print(f"来源：https://huggingface.co/{manifest['repo_id']}\n下载中断后再次启动可继续，无需登录。", flush=True)
        fetch_snapshot(manifest, cache)
        for index, entry in enumerate(manifest["files"], 1):
            source = project_path(cache / entry["remote_path"])
            print(f"校验 [{index}/{len(manifest['files'])}] {entry['remote_path']}", flush=True)
            if not verified(source, entry):
                # Preserve the bad file for diagnosis and let the next attempt re-fetch it.
                if source.is_file():
                    source.rename(project_path(source.with_name(source.name + ".bad-" + uuid.uuid4().hex)))
                raise ValueError(f"{entry['remote_path']} 校验失败，未安装模型。请再次启动以重新下载该文件。")
        output.mkdir(parents=True, exist_ok=True)
        staging = project_path(output / ("." + name + ".install-" + uuid.uuid4().hex))
        staging.mkdir()
        # Write COMPLETE last, and publish the fully verified directory in one rename.
        for entry in sorted(manifest["files"], key=lambda entry: entry["local_path"] == "COMPLETE"):
            source = project_path(cache / entry["remote_path"])
            destination = project_path(staging / entry["local_path"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            if not verified(destination, entry):
                raise ValueError("模型安装校验失败，未启用该模型")
        target = project_path(output / name)
        if target.exists():
            backup = project_path(output / (name + ".backup-" + uuid.uuid4().hex))
            project_path(target).rename(backup)
            print(f"原有不完整目录已保留为 {backup.name}。", flush=True)
        project_path(staging).rename(project_path(target))
        print("模型下载、校验和安装完成。", flush=True)
        return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Project-local checkpoint directory")
    args = parser.parse_args()
    try:
        ensure_model(args.output)
    except Exception as exc:
        message = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
        print(f"模型准备失败：{message}\n请检查网络和磁盘空间后重新双击启动器；已完成的下载会复用。", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
