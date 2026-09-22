"""Portable startup regression checks, including a real hidden Windows console."""
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import portable_entry as portable


@pytest.mark.parametrize("ready,checkpoint,error", [
    (True, "checkpoint-001800", None),
    (False, "checkpoint-001800", SystemExit),
    (True, "checkpoint-001200", RuntimeError),
    (True, None, RuntimeError),
])
def test_check_protects_console_before_progress_and_dependency_imports(
        monkeypatch, capsys, ready, checkpoint, error):
    calls = []
    monkeypatch.setattr(sys, "argv", ["portable_entry.py", "--check"])
    monkeypatch.setattr(portable, "disable_console_quick_edit", lambda: calls.append("console"))

    def inspect():
        assert calls == ["console"]
        assert "正在检查运行环境" in capsys.readouterr().out
        return {"ready": ready}

    monkeypatch.setitem(sys.modules, "scripts.bootstrap_environment", SimpleNamespace(inspect_environment=inspect))
    monkeypatch.setitem(sys.modules, "openjev.webapp", SimpleNamespace(
        available_checkpoints=lambda _: [Path(checkpoint)] if checkpoint else []))
    if error:
        with pytest.raises(error) as exc:
            portable.main()
        if error is SystemExit:
            assert exc.value.code == 1
        assert "检查通过" not in capsys.readouterr().out
    else:
        portable.main()
        assert "检查通过" in capsys.readouterr().out


@pytest.mark.parametrize("argument", [None, "--stop", "--no-browser"])
def test_background_launch_preserves_arguments_without_console_setup(monkeypatch, argument):
    args = ["portable_entry.py"] + ([argument] if argument else [])
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(portable, "disable_console_quick_edit", lambda: pytest.fail("background console changed"))
    calls = []
    monkeypatch.setattr(portable.runpy, "run_module", lambda *a, **kw: calls.append((a, kw, list(sys.argv))))
    portable.main()
    assert calls == [(("scripts.launch_web",), {"run_name": "__main__"}, args)]


@pytest.mark.skipif(os.name != "nt", reason="Windows console API")
def test_quick_edit_disabled_on_real_console_with_other_flags_preserved():
    code = r'''
import ctypes
from ctypes import wintypes
import msvcrt
import os
from scripts.portable_entry import disable_console_quick_edit

kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
kernel32.SetStdHandle.argtypes = [wintypes.DWORD, wintypes.HANDLE]
kernel32.SetStdHandle.restype = wintypes.BOOL
kernel32.GetConsoleMode.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
kernel32.GetConsoleMode.restype = wintypes.BOOL
kernel32.SetConsoleMode.argtypes = [wintypes.HANDLE, wintypes.DWORD]
kernel32.SetConsoleMode.restype = wintypes.BOOL
# Captured stdout can cause inherited stdin to be a pipe; explicitly attach the
# input of this child process's own console. Never change the caller's console.
fd = os.open("CONIN$", os.O_RDWR)
handle = msvcrt.get_osfhandle(fd)
assert kernel32.SetStdHandle(-10, handle)
mode = wintypes.DWORD()
assert kernel32.GetConsoleMode(handle, ctypes.byref(mode))
assert kernel32.SetConsoleMode(handle, mode.value | 0x0040 | 0x0080)
assert kernel32.GetConsoleMode(handle, ctypes.byref(mode))
before = mode.value
assert before & 0x0040
assert disable_console_quick_edit()
assert kernel32.GetConsoleMode(handle, ctypes.byref(mode))
assert mode.value == before & ~0x0040, (before, mode.value)
assert disable_console_quick_edit()  # Repeated checks are harmless.
assert kernel32.GetConsoleMode(handle, ctypes.byref(mode))
assert mode.value == before & ~0x0040
assert kernel32.SetStdHandle(-10, None)
assert not disable_console_quick_edit()
assert kernel32.SetStdHandle(-10, ctypes.c_void_p(-1))
assert not disable_console_quick_edit()
os.close(fd)
'''
    startup = subprocess.STARTUPINFO()
    startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = subprocess.SW_HIDE
    result = subprocess.run([sys.executable, "-c", code], cwd=portable.ROOT,
                            capture_output=True, timeout=30, startupinfo=startup,
                            creationflags=subprocess.CREATE_NEW_CONSOLE)
    assert result.returncode == 0, result.stderr.decode(errors="replace")


@pytest.mark.skipif(os.name != "nt", reason="Windows console API")
def test_headless_redirected_input_does_not_break_startup():
    result = subprocess.run([sys.executable, "-c",
                             "from scripts.portable_entry import disable_console_quick_edit; "
                             "assert not disable_console_quick_edit()"],
                            cwd=portable.ROOT, input=b"", capture_output=True, timeout=30,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    assert result.returncode == 0, result.stderr.decode(errors="replace")
