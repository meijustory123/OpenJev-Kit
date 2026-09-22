"""Ensure CPU evaluation does not round FP32 checkpoint weights through BF16."""
import json

import torch
from safetensors.torch import save_file
from transformers import Qwen3_5TextConfig, Qwen3_5TextModel

from openjev.inference import Engine
from openjev.model import DecisionModel, PROMPT_VERSION


def test_cpu_engine_preserves_checkpoint_fp32_values(tmp_path, monkeypatch):
    config = Qwen3_5TextConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        layer_types=["full_attention"],
        rope_parameters={"rope_type": "default", "rope_theta": 10000000,
                         "partial_rotary_factor": 0.5, "mrope_section": [1, 1, 2]},
    )
    model = DecisionModel(Qwen3_5TextModel(config))
    with torch.no_grad():
        next(model.backbone.parameters()).fill_(0.12345678)
    expected = next(model.backbone.parameters()).detach().clone()
    assert not torch.equal(expected, expected.bfloat16().float())
    model.backbone.save_pretrained(tmp_path / "backbone", safe_serialization=True)
    save_file(model.head.state_dict(), str(tmp_path / "decision_head.safetensors"))
    (tmp_path / "decision_config.json").write_text(json.dumps({"prompt_version": PROMPT_VERSION,
                                                             "training": {"max_length": 8192}}))
    (tmp_path / "COMPLETE").write_text("ok")
    monkeypatch.setattr("openjev.inference.get_tokenizer", lambda _: object())
    engine = Engine(tmp_path, device="cpu")
    assert torch.equal(next(engine.model.backbone.parameters()), expected)
    assert all(p.dtype == torch.float32 and p.device.type == "cpu" and not p.requires_grad
               for p in engine.model.parameters())
