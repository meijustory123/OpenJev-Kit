"""GPU architecture smoke test, using random tiny weights unless --real-model."""
import argparse
import json
import tempfile
import time
from pathlib import Path

import torch
from transformers import Qwen3_5TextConfig, Qwen3_5TextModel, Adafactor
from openjev.model import (DecisionModel, get_tokenizer, build_model, encode_candidates,
                           score_encoded, save_checkpoint, backward_soft_labels)
from openjev.protocol import parse_request, compact


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--real-model", action="store_true")
    config = json.loads(Path("configs/train.json").read_text(encoding="utf-8"))
    p.add_argument("--stress", action="store_true", help="验证 6 个候选、接近输入长度上限的训练峰值")
    p.add_argument("--max-length", type=int, default=config["max_length"])
    args = p.parse_args()
    torch.manual_seed(7)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用")
    tokenizer = get_tokenizer("Qwen3.5-0.8B")
    if args.real_model:
        model = build_model("Qwen3.5-0.8B", training=True)
    else:
        base = json.loads(Path("Qwen3.5-0.8B/config.json").read_text(encoding="utf-8"))["text_config"]
        base.update(hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                    num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                    linear_key_head_dim=16, linear_value_head_dim=16,
                    linear_num_key_heads=4, linear_num_value_heads=4,
                    layer_types=["linear_attention"] * 3 + ["full_attention"],
                    rope_parameters={"rope_type": "default", "rope_theta": 10000000,
                                     "partial_rotary_factor": 0.5, "mrope_section": [1, 1, 2]})
        config = Qwen3_5TextConfig(**base)
        config._attn_implementation = "sdpa"
        config.use_cache = False
        model = DecisionModel(Qwen3_5TextModel(config))
    model = model.float().cuda()
    model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    request = parse_request({"model": "jev-latest", "state": "客人明确说要换一双鞋，不要退款。",
                             "questions": {"q": {"type": "choice", "instructions": "客人要哪种处理？",
                                                   "criteria": {"换货": "更换商品", "退款": "退回款项"}}}})
    question = request.questions["q"]
    if args.stress:
        question.criteria.update({"维修": "修理商品", "物流": "查询运输", "发票": "开具发票", "其他": "其他诉求"})
        original = request.state
        # Binary search reaches the configured limit without a fixed repetition cap.
        low, high = 0, args.max_length
        while low < high:
            repeat = (low + high + 1) // 2
            state = "背景记录：其他服务运行正常。" * repeat + original
            try:
                encode_candidates(tokenizer, state, question, args.max_length)
                low = repeat
            except ValueError:
                high = repeat - 1
        request.state = "背景记录：其他服务运行正常。" * low + original
        # Digits consume individual tokens with this tokenizer; fill the small remainder.
        while True:
            try:
                encode_candidates(tokenizer, request.state + "1", question, args.max_length)
                request.state += "1"
            except ValueError:
                break
    encoded, tokens = encode_candidates(tokenizer, request.state, question, args.max_length)
    if args.stress:
        assert max(map(len, encoded["input_ids"])) == args.max_length
    total = sum(p.numel() for p in model.parameters())
    assert all(p.requires_grad for p in model.parameters())
    optimizer = Adafactor(model.parameters(), lr=1e-4, scale_parameter=False,
                          relative_step=False, warmup_init=False)
    tracked = next(p for n, p in model.backbone.named_parameters() if "layers.0" in n and p.ndim == 2)
    before = tracked.detach().clone()
    losses = []
    started = time.monotonic()
    model.train()
    print(compact({"stage": "training", "max_candidate_length": max(map(len, encoded["input_ids"])),
                   "candidate_count": len(question.criteria)}), flush=True)
    for step in range(2):
        target = torch.tensor([.95] + [.05 / (len(question.criteria) - 1)] * (len(question.criteria) - 1), device="cuda")
        loss = backward_soft_labels(model, tokenizer, encoded, target, "cuda")
        assert torch.isfinite(loss)
        assert tracked.grad is not None and bool(torch.isfinite(tracked.grad).all())
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(loss.item())
        del loss
        print(compact({"stage": "step_complete", "step": step + 1, "loss": losses[-1],
                       "elapsed_seconds": round(time.monotonic() - started, 1)}), flush=True)
    changed = bool(torch.any(tracked.detach() != before))
    assert changed, "底层骨干参数没有更新"
    del before
    model.eval()
    with torch.no_grad():
        expected = score_encoded(model, tokenizer, encoded, "cuda").cpu()
    # Save/reload checks key mapping, rotary buffers, head and tokenizer artifacts.
    with tempfile.TemporaryDirectory(prefix="openjev-smoke-") as temporary:
        assert Path(temporary).resolve().parent == Path(tempfile.gettempdir()).resolve()
        checkpoint = Path(temporary) / "checkpoint"
        save_checkpoint(checkpoint, model, tokenizer, {"max_length": args.max_length})
        del optimizer, tracked, model
        torch.cuda.empty_cache()
        loaded = build_model(None, checkpoint, training=True).cuda().eval()
        with torch.no_grad():
            actual = score_encoded(loaded, tokenizer, encoded, "cuda").cpu()
        assert torch.allclose(expected, actual, atol=1e-5), (expected, actual)
    report = {"kind": "real_base_model" if args.real_model else "tiny_random_architecture",
              "parameters": total, "all_parameters_trainable": True,
              "backbone_parameter_changed": changed, "losses": losses,
              "checkpoint_reload_matches": True, "candidate_tokens": tokens,
              "candidate_count": len(question.criteria),
              "configured_max_length": args.max_length,
              "max_candidate_length": max(len(ids) for ids in encoded["input_ids"]),
              "peak_cuda_allocated_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3),
              "seconds": round(time.monotonic()-started, 2)}
    Path("reports").mkdir(exist_ok=True)
    filename = ("smoke-real" if args.real_model else "smoke-tiny") + (f"-stress-{args.max_length}" if args.stress else "") + ".json"
    Path("reports", filename).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(compact(report))


if __name__ == "__main__":
    main()
