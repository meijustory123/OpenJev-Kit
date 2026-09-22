"""Bootstrap tests exercise reuse, dependency repair, and Windows path guards."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import bootstrap_environment as bootstrap


def test_ready_environment_never_downloads_or_installs(monkeypatch):
    state = {"ready": True}
    monkeypatch.setattr(bootstrap, "fresh_check", lambda _: state)
    def forbidden(*args, **kwargs):
        pytest.fail("A ready environment must not run an installer")
    monkeypatch.setattr(bootstrap, "run_module", forbidden)
    monkeypatch.setattr(bootstrap, "select_torch", forbidden)
    assert bootstrap.ensure_environment("auto", "https://example.invalid") is state


def test_missing_dependency_does_not_replace_working_torch(monkeypatch, tmp_path):
    monkeypatch.setattr(bootstrap, "ROOT", tmp_path)
    before = {"ready": False, "issues": {"httpx": {}}, "import_errors": []}
    after = {"ready": True}
    checks = iter([before, after])
    monkeypatch.setattr(bootstrap, "fresh_check", lambda _: next(checks))
    calls = []
    monkeypatch.setattr(bootstrap, "run_module", lambda *args: calls.append(args))
    monkeypatch.setattr(bootstrap.subprocess, "check_output", lambda *args, **kwargs: b"httpx==0.28.1\n")
    assert bootstrap.ensure_environment("auto", "https://example.invalid") is after
    assert any(c[0] == "scripts.install_dependencies" and "-r" in c for c in calls)
    assert not any(c[0] == "scripts.download_torch" or "--force-reinstall" in c for c in calls)


def test_torch_variants_and_device_detection(monkeypatch):
    assert bootstrap.torch_compatible("2.10.0+cpu")
    assert bootstrap.torch_compatible("2.10.0+cu128")
    assert not bootstrap.torch_compatible("2.10.0+cpu", "cuda")
    assert not bootstrap.torch_compatible("2.11.0+cu128")
    monkeypatch.setattr(bootstrap.shutil, "which", lambda _: None)
    assert bootstrap.select_torch("auto") == "cpu"
    assert bootstrap.select_torch("cuda") == "cuda"
    monkeypatch.setattr(bootstrap.shutil, "which", lambda _: "nvidia-smi.exe")
    monkeypatch.setattr(bootstrap.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, b"GPU\n"))
    assert bootstrap.select_torch("auto") == "cuda"
    monkeypatch.setattr(bootstrap.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 1, b""))
    assert bootstrap.select_torch("auto") == "cpu"


def test_failed_install_is_not_reported_ready(monkeypatch, tmp_path):
    monkeypatch.setattr(bootstrap, "ROOT", tmp_path)
    monkeypatch.setattr(bootstrap, "fresh_check", lambda _: {"ready": False, "issues": {"httpx": {}}, "import_errors": []})
    def failed(module, *args):
        if module == "scripts.install_dependencies":
            raise subprocess.CalledProcessError(1, "pip")
    monkeypatch.setattr(bootstrap, "run_module", failed)
    with pytest.raises(subprocess.CalledProcessError):
        bootstrap.ensure_environment("auto", "https://example.invalid")
    assert not (tmp_path / ".runtime/environment.json").exists()


def test_repair_includes_missing_transitive_distribution(monkeypatch):
    monkeypatch.setattr(bootstrap.metadata, "packages_distributions", lambda: {"starlette": ["starlette"]})
    monkeypatch.setattr(bootstrap.metadata, "version", lambda name: {"fastapi": "0.128.0", "starlette": "0.50.0"}[name])
    state = {"import_errors": [{"distribution": "fastapi", "missing_module": "starlette.middleware"}]}
    assert bootstrap.repair_specs(state) == ["fastapi==0.128.0", "starlette==0.50.0"]


def quote_ps(value):
    return "'" + str(value).replace("'", "''") + "'"


def powershell(code):
    script = Path(__file__).resolve().parents[1] / "scripts/bootstrap.ps1"
    return subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                           "-Command", f". {quote_ps(script)}; {code}"], capture_output=True, timeout=30)


@pytest.mark.skipif(os.name != "nt", reason="Windows bootstrap")
def test_powershell_refuses_paths_outside_project_and_preserves_backups(tmp_path):
    original = tmp_path / "broken env"
    original.mkdir()
    (original / "keep.txt").write_text("keep")
    result = powershell(
        f"$ProjectRoot={quote_ps(tmp_path)}; "
        f"try {{ Assert-ProjectPath {quote_ps(tmp_path.parent)}; exit 9 }} catch {{}}; "
        f"Backup-ProjectPath {quote_ps(original)}")
    assert result.returncode == 0, result.stderr
    backups = list(tmp_path.glob("broken env.backup-*"))
    assert len(backups) == 1 and (backups[0] / "keep.txt").read_text() == "keep"
    assert not original.exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows bootstrap")
def test_python_checksum_failure_prevents_extraction(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/bootstrap.json").write_text(json.dumps({"python": {
        "version": "3.11.16", "build": "test", "url": "https://example.invalid/python", "size": 3, "sha256": "0" * 64,
    }}))
    result = powershell(
        f"$ProjectRoot={quote_ps(tmp_path)}; "
        "function Invoke-WebRequest { param($Uri,$OutFile,$TimeoutSec,[switch]$UseBasicParsing); "
        "[IO.File]::WriteAllBytes($OutFile,[byte[]](1,2,3)) }; "
        "try { Install-ProjectPython; exit 9 } catch { if ($_.Exception.Message -notlike '*校验失败*') { throw }; exit 0 }")
    assert result.returncode == 0, result.stderr
    assert not list((tmp_path / ".runtime").glob("python-stage-*"))


@pytest.mark.skipif(os.name != "nt", reason="Windows bootstrap")
def test_python_probe_handles_unicode_and_spaces(tmp_path):
    location = tmp_path / "中文 environment"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(location)], check=True)
    executable = location / "Scripts/python.exe"
    result = powershell(
        f"$found=Find-CompatiblePython {quote_ps(executable)}; "
        f"if ($found -ne {quote_ps(executable)}) {{ throw 'Interpreter path did not round-trip' }}")
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(os.name != "nt", reason="Windows bootstrap")
def test_unsupported_platform_reports_reason_before_opening_log():
    script = Path(__file__).resolve().parents[1] / "scripts/bootstrap.ps1"
    env = dict(os.environ, PROCESSOR_ARCHITECTURE="ARM64", PROCESSOR_ARCHITEW6432="ARM64")
    result = subprocess.run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script), "-NoLaunch"],
                            env=env, capture_output=True, timeout=30)
    assert result.returncode == 1
    assert b"Windows 10/11 x64" in result.stdout
