"""Measure the 9-short / 1-medium length policy on drafts before publishing.

Works on draft files (with ``labels``) and on published shards (with ``response``).
Reports the longest complete candidate input of every request, so an author can
adjust the medium sample of a topic and re-run until the mix is exactly 9 + 1.
"""
import argparse
import json
from pathlib import Path

from openjev.data import read_jsonl
from openjev.generation_lengths import summarize_lengths, length_band, SHORT_MAX, MEDIUM_MAX
from openjev.model import get_tokenizer, encode_candidates
from openjev.protocol import build_response, compact, parse_request


def request_token_lengths(tokenizer, row, max_length):
    request = row["request"]
    labels = row.get("labels")
    if labels is None:
        from openjev.data import labels_from_response
        labels = labels_from_response(parse_request(request), row["response"])
    parsed = parse_request(request)
    lengths = {}
    for key, question in parsed.questions.items():
        encoded, _ = encode_candidates(tokenizer, parsed.state, question, max_length)
        lengths[key] = max(len(ids) for ids in encoded["input_ids"])
    return lengths


def load_rows(paths):
    rows = []
    for path in paths:
        rows.extend(read_jsonl(path))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--base-model", default="Qwen3.5-0.8B")
    parser.add_argument("--enforce-mix", action="store_true",
                        help="每题必须恰好9条短、1条中；否则退出码为1")
    parser.add_argument("--json", type=Path, help="把完整测量结果写入该文件")
    args = parser.parse_args()

    rows = load_rows(args.paths)
    if not rows:
        raise ValueError("没有可测量数据")
    tokenizer = get_tokenizer(args.base_model)
    per_sample = {}
    detail = []
    for row in rows:
        lengths = request_token_lengths(tokenizer, row, args.max_length)
        longest = max(lengths.values())
        per_sample[row["sample_id"]] = longest
        detail.append({"sample_id": row["sample_id"], "slot": row["slot"],
                       "tokens": longest, "band": length_band(longest),
                       "per_question": lengths})
    coverage = summarize_lengths(per_sample)
    for entry in detail:
        marker = "OK " if entry["band"] == "short" else "MID" if entry["band"] == "medium" else "OVER"
        print(compact({"sample": entry["sample_id"], "slot": entry["slot"],
                       "tokens": entry["tokens"], "band": marker}))
    summary = {"samples": len(detail), "max_tokens": max(per_sample.values()),
               "requests": coverage["requests"], "nonconforming_topics": coverage["nonconforming_topics"],
               "topics": coverage["topics"]}
    print(compact(summary))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"summary": summary, "samples": detail},
                                        ensure_ascii=False, indent=2), encoding="utf-8")
    if args.enforce_mix and coverage["nonconforming_topics"]:
        raise SystemExit(f"未满足每题9短1中：{coverage['nonconforming_topics']}")


if __name__ == "__main__":
    main()
