"""Run the distribution using only its bundled Python and dependencies."""
import json
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def disable_console_quick_edit():
    """Prevent accidental mouse selection from blocking this console's output."""
    if os.name != "nt":
        return False
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetStdHandle.argtypes = [wintypes.DWORD]
    kernel32.GetStdHandle.restype = wintypes.HANDLE
    kernel32.GetConsoleMode.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel32.GetConsoleMode.restype = wintypes.BOOL
    kernel32.SetConsoleMode.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.SetConsoleMode.restype = wintypes.BOOL

    handle = kernel32.GetStdHandle(-10)  # STD_INPUT_HANDLE
    mode = wintypes.DWORD()
    if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
        # pythonw, redirected input and headless checks may have no console.
        return False
    # EXTENDED_FLAGS is required when changing QUICK_EDIT_MODE. Keep Ctrl+C,
    # line input and all other existing flags; do not change registry defaults.
    return bool(kernel32.SetConsoleMode(handle, (mode.value | 0x0080) & ~0x0040))


def main():
    if "--check" in sys.argv:
        disable_console_quick_edit()
        print("正在检查运行环境，首次启动请稍候……", flush=True)
        from scripts.bootstrap_environment import inspect_environment
        state = inspect_environment()
        if not state["ready"]:
            print(json.dumps(state, ensure_ascii=False, indent=2))
            raise SystemExit(1)
        from openjev.webapp import available_checkpoints
        checkpoints = available_checkpoints(ROOT / "outputs/decision-full")
        if not checkpoints or checkpoints[-1].name != "checkpoint-001800":
            raise RuntimeError("缺少完整的第 1800 步模型，请完整解压压缩包。")
        print("运行环境和第 1800 步模型检查通过，正在打开网页。")
        return
    runpy.run_module("scripts.launch_web", run_name="__main__")


if __name__ == "__main__":
    main()
