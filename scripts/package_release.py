"""Build a relocatable Windows ZIP with checkpoint 1800 and an offline runtime."""
import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sysconfig
import time
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[1]
NAME = "OpenJev-1800-Windows-x64"
SOURCE_FILES = [
    "openjev/__init__.py", "openjev/protocol.py", "openjev/model.py",
    "openjev/inference.py", "openjev/webapp.py", "scripts/__init__.py",
    "scripts/launch_web.py", "scripts/portable_entry.py",
    "scripts/bootstrap_environment.py", "requirements.txt",
    "web/index.html", "web/style.css", "web/app.js", "examples/request.json",
    "schemas/request.schema.json", "schemas/response.schema.json",
]

START = '''@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
if not exist "runtime\\python.exe" goto missing
"%~dp0runtime\\python.exe" -I -X utf8 "%~dp0scripts\\portable_entry.py" --check
if errorlevel 1 goto failed
start "" "%~dp0runtime\\pythonw.exe" -I -X utf8 "%~dp0scripts\\portable_entry.py"
exit /b 0
:missing
echo 未找到运行环境。请先完整解压压缩包，再运行本文件。
:failed
echo 启动失败，请查看上方信息或 outputs\\webapp\\server.log。
pause
exit /b 1
'''
STOP = '''@echo off
setlocal
cd /d "%~dp0"
if exist "runtime\\pythonw.exe" start "" "%~dp0runtime\\pythonw.exe" -I -X utf8 "%~dp0scripts\\portable_entry.py" --stop
exit /b 0
'''
GUIDE = '''OpenJev 第 1800 步模型 · Windows 免安装版

1. 将压缩包完整解压到一个可写的文件夹，不要在压缩软件里直接运行。
2. 双击“启动决策模型.cmd”。网页会自动打开，等待显示“模型已就绪”。
3. 填写内容和问题，或选用网页里的示例，点击“开始判断”。
4. 使用结束后，点击网页“退出工具”或双击“关闭决策模型.cmd”。

已包含第 1800 步（第 3 轮）微调模型、Python 3.11 和全部程序依赖。
无需预先安装 Python，无需联网下载模型或依赖，不需要 API 密钥。
支持 Windows 10/11 x64；请为解压后的文件和运行日志预留约 9 GB 磁盘空间。
自动选择可用的 NVIDIA 显卡；没有兼容显卡时使用 CPU。显卡推理需要本机已有兼容驱动。
CPU 推理可用，速度取决于电脑配置；建议至少 16 GB 内存。
最长输入为 8192 token，超长输入请在网页中缩短。

请保留整个文件夹的结构。关闭工具后可整体移动，不依赖原来的解压位置。
重复双击会打开已有网页；只关闭浏览器不会结束后台模型进程。
启动检查会关闭当前窗口的快速编辑模式，避免鼠标误选文字导致暂停。
旧版若卡住且窗口标题显示“选择”，按 Esc 或回车即可恢复。
启动日志：outputs\\webapp\\server.log
模型位置：outputs\\decision-full\\checkpoint-001800

本包用于推理，不包含训练数据、优化器状态和其他训练检查点。
文件校验与版本信息见 package-manifest.json；第三方说明见 THIRD_PARTY_NOTICES.txt。
'''
NOTICES = '''OpenJev includes a full fine-tuning of the text backbone of Qwen/Qwen3.5-0.8B.
The checkpoint was modified by full fine-tuning; it is not the original Qwen checkpoint.
Qwen model license: licenses/Qwen-LICENSE.txt (Apache-2.0).
Model source: https://huggingface.co/Qwen/Qwen3.5-0.8B

CPython comes from Astral python-build-standalone (3.11.16, build 20260901).
Python license: runtime/LICENSE.txt.
https://github.com/astral-sh/python-build-standalone/releases/tag/20260901

Python package license files, including PyTorch and its bundled CUDA libraries,
are retained in runtime/Lib/site-packages, in each package and its .dist-info folder.
Their original license terms apply to the respective components.

Release Microsoft C++ runtime DLLs are included from the installed Visual Studio
x64 CRT redistributable directory for application-local deployment.
https://learn.microsoft.com/en-us/cpp/windows/determining-which-dlls-to-redistribute
https://visualstudio.microsoft.com/license-terms/
'''


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.replace("\r\n", "\n").replace("\n", "\r\n").encode("utf-8"))


def ignored(directory, names):
    return [n for n in names if n in {"__pycache__", ".pytest_cache", "_virtualenv.pth", "_virtualenv.py", "direct_url.json"}
            or n.endswith((".pyc", ".pyo"))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-root", type=Path, required=True, help="Standalone CPython 3.11 x64 directory")
    parser.add_argument("--crt-root", type=Path, required=True, help="Visual Studio release x64 CRT redist directory")
    parser.add_argument("--site-packages", type=Path, default=Path(sysconfig.get_paths()["purelib"]))
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "outputs/decision-full/checkpoint-001800")
    parser.add_argument("--output", type=Path, default=ROOT / "dist" / (NAME + ".zip"))
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT) or output.exists() or output.with_suffix(".zip.partial").exists():
        raise ValueError("Use a new output archive within the project directory")
    for path in [args.python_root / "python.exe", args.python_root / "pythonw.exe",
                 args.crt_root / "msvcp140.dll", args.crt_root / "vcruntime140.dll",
                 args.checkpoint / "COMPLETE", args.site_packages / "torch"]:
        if not path.exists():
            raise FileNotFoundError(path)
    if args.checkpoint.name != "checkpoint-001800":
        raise ValueError("This release is specifically for checkpoint 1800")
    python_info = json.loads(subprocess.check_output([
        str((args.python_root / "python.exe").resolve()), "-I", "-c",
        "import json,platform,struct; print(json.dumps([platform.python_version(),struct.calcsize('P')]))"
    ], text=True))
    if not python_info[0].startswith("3.11.") or python_info[1] != 8:
        raise ValueError("The bundled runtime must be CPython 3.11 x64")
    stage = ROOT / "outputs/package-build" / uuid.uuid4().hex / NAME
    stage.mkdir(parents=True)
    print(f"Staging release: {stage}", flush=True)
    for name in SOURCE_FILES:
        target = stage / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    print("Copying standalone Python and installed dependencies...", flush=True)
    shutil.copytree(args.python_root, stage / "runtime", ignore=ignored)
    shutil.copytree(args.site_packages, stage / "runtime/Lib/site-packages", dirs_exist_ok=True, ignore=ignored)
    for library in args.crt_root.glob("*.dll"):
        shutil.copy2(library, stage / "runtime" / library.name)
    print("Copying checkpoint 1800 (inference files only)...", flush=True)
    checkpoint_destination = stage / "outputs/decision-full/checkpoint-001800"
    shutil.copytree(args.checkpoint, checkpoint_destination,
                    ignore=lambda directory, names: [n for n in names if n == "training_state.pt" or n == "__pycache__"])
    write_text(stage / "启动决策模型.cmd", START)
    write_text(stage / "关闭决策模型.cmd", STOP)
    write_text(stage / "使用说明.txt", GUIDE)
    write_text(stage / "THIRD_PARTY_NOTICES.txt", NOTICES)
    (stage / "licenses").mkdir()
    shutil.copy2(ROOT / "Qwen3.5-0.8B/LICENSE", stage / "licenses/Qwen-LICENSE.txt")
    files = sorted(p for p in stage.rglob("*") if p.is_file())
    total = sum(p.stat().st_size for p in files)
    metadata = {"name": NAME, "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "checkpoint": "checkpoint-001800", "training_epoch": 3, "offline": True,
                "python": python_info[0], "platform": "Windows x64",
                "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions(path=[str(args.site_packages)])},
                "uncompressed_bytes": total, "files": {}}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".zip.partial")
    last_progress = time.monotonic()
    processed = 0
    print(f"Compressing {len(files)} files, {total / 1024**3:.2f} GiB (ZIP64)...", flush=True)
    with zipfile.ZipFile(temporary, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
        for path in files:
            relative = path.relative_to(stage).as_posix()
            digest = hashlib.sha256()
            info = zipfile.ZipInfo.from_file(path, arcname=f"{NAME}/{relative}")
            info.compress_type = zipfile.ZIP_DEFLATED
            info._compresslevel = 1
            with path.open("rb") as source, archive.open(info, "w", force_zip64=True) as destination:
                for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    digest.update(block)
                    destination.write(block)
                    processed += len(block)
                    if time.monotonic() - last_progress >= 15:
                        print(f"Compressed {processed / total:.0%}: {relative}", flush=True)
                        last_progress = time.monotonic()
            metadata["files"][relative] = {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}
        manifest = json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
        archive.writestr(f"{NAME}/package-manifest.json", manifest)
    output_tmp = temporary.resolve()
    if not output_tmp.is_relative_to(ROOT) or not output.resolve().is_relative_to(ROOT):
        raise ValueError("Output paths must remain inside the project")
    output_tmp.rename(output)
    write_text(stage / "package-manifest.json", manifest)
    digest = hashlib.sha256()
    with output.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    output.with_suffix(".zip.sha256").write_text(f"{digest.hexdigest()}  {output.name}\n", encoding="ascii")
    print(json.dumps({"archive": str(output), "bytes": output.stat().st_size, "sha256": digest.hexdigest(),
                      "stage": str(stage), "files": len(files) + 1}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
