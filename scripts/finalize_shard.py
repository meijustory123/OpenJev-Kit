"""Convert agent-authored labels into a complete, checked response envelope."""
import argparse
from pathlib import Path
from openjev.data import read_jsonl, write_jsonl, validate_shards
from openjev.protocol import build_response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("draft", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/shards"))
    args = parser.parse_args()
    rows = read_jsonl(args.draft)
    if not rows:
        raise ValueError("草稿为空")
    topic = rows[0]["topic_id"]
    destination = args.output_dir / f"{topic}.jsonl"
    if destination.exists():
        raise FileExistsError(f"不覆盖已有分片，请先核对并另存原文件: {destination}")
    completed = []
    for row in rows:
        if "response" in row or "labels" not in row:
            raise ValueError("草稿应包含 labels，不应包含 response")
        row = dict(row)
        labels = row.pop("labels")
        row["response"] = build_response(row["request"], labels)
        completed.append(row)
    # Validate the entire shard in a private staging directory before publishing.
    staging = args.output_dir / "_staging" / f"{topic}.jsonl"
    write_jsonl(staging, completed)
    validate_shards([staging])
    staging.replace(destination)
    print(f"已生成并校验 {destination}，共 {len(completed)} 条")


if __name__ == "__main__":
    main()
