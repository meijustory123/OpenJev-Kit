"""Web lifecycle tests use explicit test engines, never fake production outputs."""
import threading
import time

import pytest
from fastapi.testclient import TestClient

from openjev.protocol import build_response
from openjev.webapp import ModelManager, available_checkpoints, create_web_app
from tests.test_protocol import example


def checkpoint(root, step, complete=True):
    path = root / f"checkpoint-{step:06d}"
    for name in ["decision_config.json", "decision_head.safetensors", "backbone/config.json",
                 "backbone/model.safetensors", "tokenizer/tokenizer_config.json", "tokenizer/tokenizer.json"]:
        file = path / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("{}")
    if complete:
        (path / "COMPLETE").write_text("ok")
    return path


def wait_ready(manager, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if manager.snapshot()["status"] in {"ready", "error"}:
            return manager.snapshot()
        time.sleep(.01)
    raise AssertionError("test loader did not finish")


class TestEngine:
    __test__ = False
    max_length = 8192

    def __init__(self, path, device):
        self.path = path
        self.device = device

    def predict(self, raw):
        _, labels = example()
        return build_response(raw, labels, input_tokens=123)


def test_complete_checkpoint_selection_and_auto_loading(tmp_path):
    first = checkpoint(tmp_path, 100)
    checkpoint(tmp_path, 200, complete=False)
    assert available_checkpoints(tmp_path) == [first]
    manager = ModelManager(tmp_path, TestEngine, lambda _: "cpu")
    manager.start()
    assert wait_ready(manager)["checkpoint"] == first.name
    latest = checkpoint(tmp_path, 200)
    assert manager.snapshot()["latest_checkpoint"] == latest.name
    assert manager.snapshot()["checkpoint"] == first.name
    manager.request_load("auto")
    assert wait_ready(manager)["checkpoint"] == latest.name
    manager.stop()


def test_web_api_waiting_ready_json_validation_and_shutdown(tmp_path):
    manager = ModelManager(tmp_path, TestEngine, lambda _: "cpu")
    stopped = threading.Event()
    app = create_web_app(manager, "test-token", stopped.set)
    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        assert client.get("/assets/app.js").status_code == 200
        assert client.get("/api/status").json()["status"] == "waiting"
        assert client.get("/api/session").json() == {"token": "test-token"}
        raw, _ = example()
        assert client.post("/v1/systemone", json=raw).status_code == 403
        headers = {"X-OpenJev-Token": "test-token"}
        assert client.post("/v1/systemone", json=raw, headers=headers).status_code == 503
        assert client.post("/api/reload", json={"device": "auto"}, headers=headers).status_code == 503
        checkpoint(tmp_path, 100)
        assert client.post("/api/reload", json={"device": "auto"}, headers=headers).status_code == 200
        assert wait_ready(manager)["status"] == "ready"
        response = client.post("/v1/systemone", json=raw, headers=headers)
        assert response.status_code == 200
        assert set(response.json()) == {"model", "answers", "usage"}
        assert "confidence" not in response.text
        assert client.post("/v1/systemone", content='{"model":"a","model":"b"}', headers=headers).status_code == 422
        assert client.post("/v1/systemone", json={}, headers=headers).status_code == 422
        assert client.post("/api/reload", json=[], headers=headers).status_code == 422
        assert client.post("/api/reload", json={"device": "invalid"}, headers=headers).status_code == 422
        assert client.post("/api/shutdown", json={}, headers=headers).status_code == 200
        assert stopped.is_set()
        assert client.get("/", headers={"Host": "malicious.example"}).status_code == 400


def test_load_failure_is_visible_and_retry_works(tmp_path):
    checkpoint(tmp_path, 100)
    def fail(*args, **kwargs):
        raise RuntimeError("test load failure")
    manager = ModelManager(tmp_path, fail, lambda _: "cpu")
    manager.request_load()
    assert "test load failure" in wait_ready(manager)["message"]
    manager.engine_factory = TestEngine
    manager.request_load()
    assert wait_ready(manager)["status"] == "ready"
    manager.stop()


def test_busy_prediction_is_rejected(tmp_path):
    from openjev.webapp import ModelUnavailable
    checkpoint(tmp_path, 100)
    manager = ModelManager(tmp_path, TestEngine, lambda _: "cpu")
    manager.request_load()
    wait_ready(manager)
    raw, _ = example()
    manager._operation.acquire()
    try:
        with pytest.raises(ModelUnavailable):
            manager.predict(raw)
    finally:
        manager._operation.release()
    manager.stop()


def test_waiting_server_automatically_loads_first_saved_checkpoint(tmp_path):
    manager = ModelManager(tmp_path, TestEngine, lambda _: "cpu")
    manager.start()
    assert manager.snapshot()["status"] == "waiting"
    checkpoint(tmp_path, 100, complete=False)
    assert manager.snapshot()["status"] == "waiting"
    checkpoint(tmp_path, 100, complete=True)
    assert wait_ready(manager, timeout=8)["checkpoint"] == "checkpoint-000100"
    manager.stop()


def test_auto_device_uses_cpu_when_gpu_lacks_bf16(tmp_path, monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    manager = ModelManager(tmp_path)
    assert manager._choose_device("auto") == "cpu"
    with pytest.raises(ValueError, match="BF16"):
        manager._choose_device("cuda")
