"""Download/install boundaries: no partial model is published or existing data lost."""
import hashlib
import json

import pytest

from openjev.webapp import available_checkpoints
from scripts import download_model as download
from tests.test_webapp import checkpoint


@pytest.fixture
def release(tmp_path, monkeypatch):
    monkeypatch.setattr(download, "ROOT", tmp_path)
    names = ["COMPLETE", "decision_config.json", "decision_head.safetensors", "backbone/config.json",
             "backbone/model.safetensors", "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json",
             "tokenizer/chat_template.jinja"]
    payloads = {name.split("/")[-1]: (name + " test payload").encode() for name in names}
    manifest = {"repo_id": "test/release", "revision": "a" * 40, "checkpoint_name": "checkpoint-001800",
                "files": [{"remote_path": name.split("/")[-1], "local_path": name,
                           "size": len(payloads[name.split("/")[-1]]),
                           "sha256": hashlib.sha256(payloads[name.split("/")[-1]]).hexdigest()} for name in names]}
    path = tmp_path / "configs/model-download.json"
    path.parent.mkdir()
    path.write_text(json.dumps(manifest), encoding="utf-8")

    def fetch(_manifest, cache):
        for name, data in payloads.items():
            (cache / name).write_bytes(data)

    monkeypatch.setattr(download, "fetch_snapshot", fetch)
    return tmp_path, manifest, payloads, fetch


def test_existing_complete_checkpoint_skips_network(release, monkeypatch):
    root, *_ = release
    old = checkpoint(root / "outputs/decision-full", 600)
    newest = checkpoint(root / "outputs/decision-full", 1200)
    checkpoint(root / "outputs/decision-full", 1800, complete=False)
    monkeypatch.setattr(download, "fetch_snapshot", lambda *_: pytest.fail("must not contact HF"))
    assert download.ensure_model() == newest
    assert old.exists()


def test_verified_install_maps_files_and_reuses_local_model(release, monkeypatch):
    root, manifest, payloads, _ = release
    result = download.ensure_model()
    assert available_checkpoints(result.parent) == [result]
    for entry in manifest["files"]:
        assert (result / entry["local_path"]).read_bytes() == payloads[entry["remote_path"]]
    monkeypatch.setattr(download, "fetch_snapshot", lambda *_: pytest.fail("no repeated download"))
    assert download.ensure_model() == result


@pytest.mark.parametrize("failure", ["corrupt", "missing", "interrupted"])
def test_failed_download_is_never_loadable_and_retry_succeeds(release, monkeypatch, failure):
    root, manifest, payloads, fetch = release
    cache_file = root / ".hf-cache/download-checkpoint-001800/model.safetensors"

    def broken(manifest, cache):
        fetch(manifest, cache)
        if failure == "corrupt":
            cache_file.write_bytes(b"x" * len(payloads["model.safetensors"]))
        else:
            cache_file.unlink()
        if failure == "interrupted":
            (cache / "model.safetensors.incomplete").write_bytes(b"partial")
            raise ConnectionError("test network interruption")

    monkeypatch.setattr(download, "fetch_snapshot", broken)
    with pytest.raises((ValueError, ConnectionError)):
        download.ensure_model()
    assert available_checkpoints(root / "outputs/decision-full") == []
    assert not (root / "outputs/decision-full/checkpoint-001800/COMPLETE").exists()
    assert (cache_file.parent / "tokenizer.json").read_bytes() == payloads["tokenizer.json"]
    if failure == "corrupt":
        assert list(cache_file.parent.glob("model.safetensors.bad-*"))
    monkeypatch.setattr(download, "fetch_snapshot", fetch)
    assert download.ensure_model().name == manifest["checkpoint_name"]


def test_incomplete_checkpoint_is_backed_up_and_install_is_atomic(release, monkeypatch):
    root, _, _, _ = release
    output = root / "outputs/decision-full"
    previous = checkpoint(output, 1800, complete=False)
    (previous / "my-notes.txt").write_text("preserve me")
    copied = []
    original_copy = download.shutil.copy2

    def observe_copy(source, destination):
        assert available_checkpoints(output) == []
        copied.append(destination.name)
        return original_copy(source, destination)

    monkeypatch.setattr(download.shutil, "copy2", observe_copy)
    download.ensure_model()
    assert copied[-1] == "COMPLETE"
    backups = list(output.glob("checkpoint-001800.backup-*"))
    assert len(backups) == 1
    assert (backups[0] / "my-notes.txt").read_text() == "preserve me"
    assert len(available_checkpoints(output)) == 1


@pytest.mark.parametrize("unsafe", ["../escape", "/absolute", "C:/escape", "a\\b", ".", "a//b"])
def test_unsafe_manifest_paths_are_rejected(release, unsafe):
    root, manifest, *_ = release
    manifest["files"][0]["local_path"] = unsafe
    path = root / "configs/model-download.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        download.ensure_model()


def test_install_outside_project_is_rejected(release):
    root, *_ = release
    with pytest.raises(ValueError):
        download.ensure_model(root.parent / "outside-model-download")


def test_missing_tokenizer_does_not_count_as_complete(release):
    root, *_ = release
    old = checkpoint(root / "outputs/decision-full", 1800)
    (old / "tokenizer/tokenizer.json").unlink()
    assert available_checkpoints(old.parent) == []
    assert (download.ensure_model() / "tokenizer/tokenizer.json").is_file()


def test_public_download_ignores_login_and_enables_progress(tmp_path, monkeypatch):
    import huggingface_hub
    from huggingface_hub import constants
    root = tmp_path
    manifest = {"repo_id": "test/release", "revision": "a" * 40, "files": [{"remote_path": "model.safetensors"}]}
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    monkeypatch.setenv("HF_TOKEN", "not-a-real-token")
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    monkeypatch.setattr(constants, "HF_HUB_DISABLE_PROGRESS_BARS", True)
    calls = []

    def public_snapshot(**kwargs):
        calls.append(kwargs)
        assert constants.HF_HUB_OFFLINE is False
        with kwargs["tqdm_class"](total=1) as progress:
            assert not progress.disable
            progress.update()
        return str(root / "cache")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", public_snapshot)
    assert download.fetch_snapshot(manifest, root / "cache") == str(root / "cache")
    assert calls[0]["token"] is False
    assert calls[0]["endpoint"] == "https://huggingface.co"
    assert calls[0]["revision"] == manifest["revision"]
