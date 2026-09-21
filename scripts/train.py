import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch
from transformers import Adafactor

from openjev.data import read_jsonl
from openjev.model import (build_model, get_tokenizer, encode_candidates,
                           score_encoded, save_checkpoint, backward_soft_labels)
from openjev.protocol import parse_request, compact


def question(record):
    return parse_request({"model": "local", "state": record["state"],
                          "questions": {"q": record["question"]}}).questions["q"]


def preload(path, tokenizer, max_length):
    records = read_jsonl(path)
    if not records:
        raise ValueError(f"训练或验证集为空: {path}")
    for record in records:
        record["encoded"], _ = encode_candidates(tokenizer, record["state"], question(record), max_length)
        target = record["target"]
        if len(target) != len(record["encoded"]["input_ids"]) or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1
            for v in target
        ) or abs(sum(target) - 1) > 1e-6:
            raise ValueError(f"无效概率标签: {record['sample_id']}")
    return records


def loss_for(model, tokenizer, record, config):
    logits = score_encoded(model, tokenizer, record["encoded"], config["device"],
                           config["candidate_micro_batch_size"])
    target = torch.tensor(record["target"], device=logits.device, dtype=torch.float32)
    return -(target * logits.log_softmax(dim=0)).sum()


@torch.no_grad()
def validate(model, tokenizer, records, config):
    model.eval()
    total = sum(loss_for(model, tokenizer, r, config).item() for r in records)
    model.train()
    return total / len(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/train.json"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.max_steps is not None:
        config["max_steps"] = args.max_steps
    if config["optimizer"] != "adafactor":
        raise ValueError("当前全量方案支持 Adafactor")
    if config["parameter_dtype"] != "float32" or config["autocast_dtype"] != "bfloat16":
        raise ValueError("训练参数必须为 float32，前向混合精度为 bfloat16")
    if any(config[key] < 1 for key in ["epochs", "gradient_accumulation_steps", "candidate_micro_batch_size", "max_length", "save_every_steps"]):
        raise ValueError("轮次与批次设置必须为正整数")
    if config["max_steps"] is not None and config["max_steps"] < 1:
        raise ValueError("max_steps 必须为正整数")
    if config["device"].startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，请运行环境检查")
    if config["device"].startswith("cuda") and not torch.cuda.is_bf16_supported():
        raise RuntimeError("当前 GPU 不支持 BF16")
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    tokenizer = get_tokenizer(args.resume / "tokenizer" if args.resume else config["base_model"])
    train = preload(config["train_file"], tokenizer, config["max_length"])
    valid = preload(config["validation_file"], tokenizer, config["max_length"])
    # Topic-level separation is mandatory even for manually supplied prepared data.
    if {r["topic_id"] for r in train} & {r["topic_id"] for r in valid}:
        raise ValueError("训练与验证集命题重叠")
    fingerprint = {name: hashlib.sha256(Path(config[name]).read_bytes()).hexdigest()
                   for name in ["train_file", "validation_file"]}
    if args.resume and not (args.resume / "COMPLETE").exists():
        raise ValueError("断点未完整写入，不能恢复")
    model = build_model(config["base_model"], args.resume, training=True).to(config["device"])
    if config["gradient_checkpointing"]:
        model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    total_parameters = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if total_parameters != trainable:
        raise RuntimeError("全量微调不允许冻结任何文本骨干或决策头参数")
    optimizer = Adafactor([
        {"params": model.backbone.parameters(), "lr": config["learning_rate"]},
        {"params": model.head.parameters(), "lr": config["head_learning_rate"]},
    ], scale_parameter=False, relative_step=False, warmup_init=False,
       weight_decay=config["weight_decay"])
    accum = config["gradient_accumulation_steps"]
    steps_per_epoch = math.ceil(len(train) / accum)
    planned = config["epochs"] * steps_per_epoch
    maximum = min(planned, config["max_steps"] or planned)
    start_epoch, offset, step = 0, 0, 0
    if args.resume:
        state = torch.load(args.resume / "training_state.pt", map_location="cpu", weights_only=True)
        if state["fingerprint"] != fingerprint:
            raise ValueError("数据文件已变化，不能恢复旧训练游标")
        old_config = json.loads((args.resume / "decision_config.json").read_text(encoding="utf-8"))["training"]
        for key in ["seed", "optimizer", "gradient_accumulation_steps", "learning_rate", "head_learning_rate", "max_length"]:
            if config[key] != old_config[key]:
                raise ValueError(f"恢复训练时不能改变 {key}")
        optimizer.load_state_dict(state["optimizer"])
        # Adafactor's factorized state must live beside each parameter.
        for parameter, values in optimizer.state.items():
            for key, value in values.items():
                if isinstance(value, torch.Tensor):
                    values[key] = value.to(parameter.device)
        start_epoch, offset, step = state["epoch"], state["offset"], state["step"]
        torch.set_rng_state(state["torch_rng"])
        if torch.cuda.is_available() and state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        if step >= maximum:
            raise ValueError("断点已达到计划步数；增加 epochs 或 max_steps 后再恢复")
    out = Path(config["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    if not args.resume and any(out.glob("checkpoint-*")):
        raise FileExistsError("输出目录已有断点，请使用 --resume 或新目录")
    print(compact({"parameters": total_parameters, "trainable": trainable,
                   "training_questions": len(train), "planned_steps": maximum}), flush=True)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()
    last_checkpoint = None

    def save(epoch, next_offset):
        nonlocal last_checkpoint
        destination = out / f"checkpoint-{step:06d}"
        if destination == last_checkpoint:
            return
        save_checkpoint(destination, model, tokenizer, config, {
            "optimizer": optimizer.state_dict(), "epoch": epoch, "offset": next_offset,
            "step": step, "fingerprint": fingerprint, "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        })
        last_checkpoint = destination
        (out / "latest.txt").write_text(str(destination.resolve()), encoding="utf-8")

    for epoch in range(start_epoch, config["epochs"]):
        indices = list(range(len(train)))
        random.Random(config["seed"] + epoch).shuffle(indices)
        start = offset if epoch == start_epoch else 0
        for begin in range(start, len(indices), accum):
            batch_indices = indices[begin:begin + accum]
            losses = []
            for index in batch_indices:
                record = train[index]
                loss = backward_soft_labels(model, tokenizer, record["encoded"], record["target"],
                                            config["device"], config["candidate_micro_batch_size"],
                                            scale=1 / len(batch_indices))
                losses.append(loss.item())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            # Fixed absolute LR plus short warmup; no implicit Adafactor schedule.
            warmup = min(1.0, (step + 1) / max(1, math.ceil(planned * 0.03)))
            for group, lr in zip(optimizer.param_groups, [config["learning_rate"], config["head_learning_rate"]]):
                group["lr"] = lr * warmup
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            log = {"step": step, "epoch": epoch + 1, "loss": sum(losses) / len(losses),
                   "elapsed_seconds": round(time.monotonic() - started, 1)}
            print(compact(log), flush=True)
            with (out / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(compact(log) + "\n")
            next_offset = begin + len(batch_indices)
            if step % config["save_every_steps"] == 0 or step >= maximum:
                save(epoch, next_offset)
            if step >= maximum:
                metric = {"step": step, "validation_soft_cross_entropy": validate(model, tokenizer, valid, config)}
                print(compact(metric), flush=True)
                with (out / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(compact(metric) + "\n")
                print(f"训练停止于完整优化步，断点: {last_checkpoint}", flush=True)
                return
        metric = {"epoch": epoch + 1, "validation_soft_cross_entropy": validate(model, tokenizer, valid, config)}
        print(compact(metric), flush=True)
        with (out / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(compact(metric) + "\n")
        save(epoch + 1, 0)


if __name__ == "__main__":
    main()
