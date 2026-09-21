import argparse
import json
from pathlib import Path
from openjev.data import catalog, read_jsonl, validate_shards, question_records
from openjev.model import get_tokenizer, encode_candidates
from openjev.protocol import parse_request, compact
from openjev.generation_lengths import summarize_lengths


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--group", type=int, choices=range(1, 21))
    p.add_argument("--prepared", action="store_true")
    p.add_argument("--topics", nargs="+", help="仅预检指定命题，例如 T0061 T0062")
    p.add_argument("--enforce-mix", action="store_true",
                   help="要求所选每题恰好9条<=512 token、1条513—1024 token；旧数据默认只报告")
    config = json.loads(Path("configs/train.json").read_text(encoding="utf-8"))
    p.add_argument("--max-length", type=int, default=config["max_length"])
    p.add_argument("--base-model", default="Qwen3.5-0.8B")
    args = p.parse_args()
    tokenizer = get_tokenizer(args.base_model)
    if args.prepared:
        records = []
        for split in ["train", "validation", "test"]:
            records.extend(read_jsonl(Path("data/prepared") / f"{split}.jsonl"))
    else:
        topics = catalog()
        paths = sorted(Path("data/shards").glob("T*.jsonl"))
        if args.group:
            paths = [p for p in paths if topics.get(p.stem, {}).get("group") == args.group]
        records = question_records(validate_shards(paths))
    if args.topics:
        selected = set(args.topics)
        available = {r["topic_id"] for r in records}
        if selected - available:
            raise ValueError(f"所选命题不存在：{sorted(selected - available)}")
        records = [r for r in records if r["topic_id"] in selected]
    lengths = []
    request_lengths = {}
    for record in records:
        question = parse_request({"model": "local", "state": record["state"],
                                  "questions": {"q": record["question"]}}).questions["q"]
        try:
            encoded, _ = encode_candidates(tokenizer, record["state"], question, args.max_length)
        except ValueError as exc:
            raise ValueError(f"{record['sample_id']}/{record['question_id']}: {exc}") from exc
        lengths.extend(len(ids) for ids in encoded["input_ids"])
        request_lengths[record["sample_id"]] = max(
            request_lengths.get(record["sample_id"], 0),
            max(map(len, encoded["input_ids"])))
    if not lengths:
        raise ValueError("没有可检查的数据")
    lengths.sort()
    coverage = summarize_lengths(request_lengths)
    print(compact({"questions": len(records), "candidate_inputs": len(lengths),
                   "configured_max_length": args.max_length,
                   "max_tokens": max(lengths), "p95_tokens": lengths[int((len(lengths)-1)*0.95)],
                   "request_length_coverage": coverage}))
    if args.enforce_mix and coverage["nonconforming_topics"]:
        raise ValueError(f"未满足每题9短1中：{coverage['nonconforming_topics']}")


if __name__ == "__main__":
    main()
