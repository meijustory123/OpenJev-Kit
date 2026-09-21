"""Opt-in CUDA test for the actual training CLI, resume and inference service."""
import copy
import json
import os
import subprocess
import sys
from pathlib import Path
import pytest


@pytest.mark.skipif(os.environ.get("OPENJEV_TEST_GPU") != "1", reason="set OPENJEV_TEST_GPU=1 for CUDA pipeline test")
def test_full_training_resume_and_inference(tmp_path):
    import torch
    from safetensors import safe_open
    from transformers import Qwen3_5TextConfig, Qwen3_5TextModel
    from openjev.model import get_tokenizer
    from openjev.data import catalog, question_records, write_jsonl
    from openjev.inference import Engine
    from openjev.protocol import parse_request, labels_from_response
    from tests.test_data import fixture_rows

    assert torch.cuda.is_available()
    cfg = json.loads(Path("Qwen3.5-0.8B/config.json").read_text(encoding="utf-8"))["text_config"]
    cfg.update(hidden_size=64, intermediate_size=128, num_hidden_layers=4,
               num_attention_heads=4, num_key_value_heads=2, head_dim=16,
               linear_key_head_dim=16, linear_value_head_dim=16,
               linear_num_key_heads=4, linear_num_value_heads=4,
               layer_types=["linear_attention"] * 3 + ["full_attention"],
               rope_parameters={"rope_type": "default", "rope_theta": 10000000,
                                "partial_rotary_factor": .5, "mrope_section": [1, 1, 2]})
    torch.manual_seed(11)
    backbone = Qwen3_5TextModel(Qwen3_5TextConfig(**cfg))
    base = tmp_path / "base"
    backbone.save_pretrained(base)
    get_tokenizer("Qwen3.5-0.8B").save_pretrained(base)
    del backbone
    topics = catalog()
    train_id = next(k for k, r in topics.items() if r["split"] == "train")
    valid_id = next(k for k, r in topics.items() if r["split"] == "validation")
    write_jsonl(tmp_path / "train.jsonl", question_records(fixture_rows(train_id))[:2])
    write_jsonl(tmp_path / "valid.jsonl", question_records(fixture_rows(valid_id))[:1])
    config = json.loads(Path("configs/train.json").read_text(encoding="utf-8"))
    config.update(base_model=str(base), train_file=str(tmp_path / "train.jsonl"),
                  validation_file=str(tmp_path / "valid.jsonl"), output_dir=str(tmp_path / "continuous"),
                  epochs=1, gradient_accumulation_steps=1, save_every_steps=1)
    config_file = tmp_path / "train-config.json"
    config_file.write_text(json.dumps(config), encoding="utf-8")
    command = [sys.executable, "-m", "scripts.train", "--config", str(config_file)]
    env = {**os.environ, "PYTHONUTF8": "1", "TOKENIZERS_PARALLELISM": "false"}
    result = subprocess.run(command, capture_output=True, env=env, timeout=180)
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
    first = tmp_path / "continuous/checkpoint-000001"
    final = tmp_path / "continuous/checkpoint-000002"
    assert (final / "COMPLETE").exists()
    config["output_dir"] = str(tmp_path / "resumed")
    config_file.write_text(json.dumps(config), encoding="utf-8")
    result = subprocess.run([*command, "--resume", str(first)], capture_output=True, env=env, timeout=180)
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
    resumed = tmp_path / "resumed/checkpoint-000002"
    maximum_difference = 0.0
    for relative in ["backbone/model.safetensors", "decision_head.safetensors"]:
        with safe_open(str(final / relative), framework="pt") as a, safe_open(str(resumed / relative), framework="pt") as b:
            assert set(a.keys()) == set(b.keys())
            for key in a.keys():
                left, right = a.get_tensor(key), b.get_tensor(key)
                maximum_difference = max(maximum_difference, (left-right).abs().max().item())
                assert torch.allclose(left, right, atol=1e-6, rtol=1e-5), key
    engine = Engine(final)
    assert engine.max_length == config["max_length"] == 8192
    raw = fixture_rows(train_id)[0]["request"]
    alone = engine.predict(raw)
    mixed = copy.deepcopy(raw)
    mixed["questions"]["额外判断"] = {"type": "noul", "instructions": "是否提供了状态描述？"}
    combined = engine.predict(mixed)
    assert alone["answers"]["choice"] == combined["answers"]["choice"]
    renamed = copy.deepcopy(raw)
    renamed["questions"] = {"改名": renamed["questions"]["choice"]}
    assert alone["answers"]["choice"] == engine.predict(renamed)["answers"]["改名"]
    reverse = copy.deepcopy(raw)
    reverse["questions"]["choice"]["criteria"] = dict(reversed(list(reverse["questions"]["choice"]["criteria"].items())))
    reversed_answer = engine.predict(reverse)["answers"]["choice"]
    assert alone["answers"]["choice"]["probabilities"] == reversed_answer["probabilities"]
    labels_from_response(parse_request(mixed), combined)
    from fastapi.testclient import TestClient
    from scripts.serve import create_app
    client = TestClient(create_app(engine))
    response = client.post("/v1/systemone", json=mixed)
    assert response.status_code == 200
    assert "confidence" not in response.text
    Path("reports/training-pipeline.json").write_text(json.dumps({
        "kind": "tiny_random_architecture", "train_cli": True, "resume_cli": True,
        "continuous_vs_resumed_max_difference": maximum_difference,
        "question_isolation": True, "question_id_invariance": True,
        "choice_order_invariance": True, "http_inference": True,
        "checkpoint_inference_max_length": engine.max_length,
    }, indent=2), encoding="utf-8")
