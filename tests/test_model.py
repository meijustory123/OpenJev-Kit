"""Input-boundary and full-gradient checks for candidate replay."""
import copy
import os

import pytest
import torch
from transformers import Qwen3_5TextConfig, Qwen3_5TextModel

from openjev.model import DecisionModel, encode_candidates, get_tokenizer, score_encoded, backward_soft_labels
from openjev.protocol import parse_request


def test_real_tokenizer_accepts_8192_and_rejects_8193_without_truncation():
    tokenizer = get_tokenizer("Qwen3.5-0.8B")
    question = parse_request({"model": "jev-latest", "state": "",
                              "questions": {"q": {"type": "noul", "instructions": "是否表达满意？"}}}).questions["q"]
    encoded, _ = encode_candidates(tokenizer, "1" * 8192, question, 10000)
    overhead = max(map(len, encoded["input_ids"])) - 8192
    state = "1" * (8192 - overhead)
    encoded, _ = encode_candidates(tokenizer, state, question, 8192)
    assert max(map(len, encoded["input_ids"])) == 8192
    with pytest.raises(ValueError, match="8193"):
        encode_candidates(tokenizer, state + "1", question, 8192)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    os.environ.get("OPENJEV_TEST_GPU") != "1", reason="set OPENJEV_TEST_GPU=1 for BF16 check"))])
def test_candidate_replay_preserves_full_parameter_gradients_and_rng(device):
    torch.manual_seed(71)
    cfg = Qwen3_5TextConfig(
        vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        linear_key_head_dim=16, linear_value_head_dim=16,
        linear_num_key_heads=2, linear_num_value_heads=2,
        layer_types=["linear_attention", "full_attention"], attention_dropout=0.1,
        rope_parameters={"rope_type": "default", "rope_theta": 10000000,
                         "partial_rotary_factor": 0.5, "mrope_section": [1, 1, 2]},
    )
    cfg._attn_implementation = "sdpa"
    cfg.use_cache = False
    ordinary = DecisionModel(Qwen3_5TextModel(cfg)).to(device).train()
    checkpointed = copy.deepcopy(ordinary).train()
    checkpointed.backbone.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    tokenizer = get_tokenizer("Qwen3.5-0.8B")
    encoded = {"input_ids": [[12, 13, 14, 15], [12, 13, 21], [12, 22, 23, 24, 25]],
               "attention_mask": [[1] * 4, [1] * 3, [1] * 5]}
    target = torch.tensor([0.7, 0.2, 0.1], device=device)
    torch.manual_seed(79)
    logits = score_encoded(ordinary, tokenizer, encoded, device, amp=device == "cuda")
    expected_loss = -(target * logits.log_softmax(0)).sum()
    (expected_loss / 3).backward()
    expected_rng = torch.get_rng_state()
    expected_cuda_rng = torch.cuda.get_rng_state() if device == "cuda" else None
    torch.manual_seed(79)
    actual_loss = backward_soft_labels(checkpointed, tokenizer, encoded, target, device, scale=1/3, amp=device == "cuda")
    torch.testing.assert_close(actual_loss, expected_loss.detach())
    assert torch.equal(torch.get_rng_state(), expected_rng)
    if device == "cuda":
        assert torch.equal(torch.cuda.get_rng_state(), expected_cuda_rng)
    for (name, expected), (_, actual) in zip(ordinary.named_parameters(), checkpointed.named_parameters()):
        assert expected.requires_grad and actual.requires_grad, name
        assert expected.grad is not None and actual.grad is not None, name
        torch.testing.assert_close(actual.grad, expected.grad, atol=1e-6, rtol=1e-4, msg=name)
