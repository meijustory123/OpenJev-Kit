import argparse
import json
from pathlib import Path
from openjev.data import catalog, read_jsonl, validate_shards, question_records
from openjev.model import get_tokenizer, encode_candidates
from openjev.protocol import parse_request, compact


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--group", type=int, choices=range(1, 21))
    p.add_argument("--prepared", action="store_true")
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
    print(compact({"questions": len(records), "candidate_inputs": len(lengths),
                   "configured_max_length": args.max_length,
                   "max_tokens": max(lengths), "p95_tokens": lengths[int((len(lengths)-1)*0.95)],
                   "request_length_coverage": {
                       "short_up_to_1024": sum(n <= 1024 for n in request_lengths.values()),
                       "medium_1025_to_4096": sum(1024 < n <= 4096 for n in request_lengths.values()),
                       "long_4097_to_8192": sum(4096 < n <= 8192 for n in request_lengths.values()),
                       "over_8192": sum(n > 8192 for n in request_lengths.values())}}))


if __name__ == "__main__":
    main()
