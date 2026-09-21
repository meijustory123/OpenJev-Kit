"""Flag topics whose samples reuse an identical state apart from digits/whitespace.

The generation plan requires the ten samples of a topic to change substantive
facts, wording, task boundary or noise -- not merely a number inside one fixed
template. This mirrors the audit heuristic so a fix can be verified on its own.
"""
import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

from openjev.data import read_jsonl


def text_values(value):
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return "\n".join(text_values(v) for v in value.values())
    if isinstance(value, list):
        return "\n".join(text_values(v) for v in value)
    return str(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("topics", nargs="*", help="命题 ID，例如 T0182 T0190；留空则检查全部分片")
    parser.add_argument("--shards", type=Path, default=Path("data/shards"))
    args = parser.parse_args()

    if args.topics:
        paths = [args.shards / f"{topic}.jsonl" for topic in args.topics]
    else:
        paths = sorted(args.shards.glob("T*.jsonl"))
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise SystemExit(f"分片不存在：{missing}")

    collisions = defaultdict(list)
    for path in paths:
        groups = defaultdict(list)
        for row in read_jsonl(path):
            normalized = re.sub(r"\s+", "", re.sub(r"\d+", "#", text_values(row["request"]["state"])))
            groups[hashlib.sha256(normalized.encode()).hexdigest()].append(row["sample_id"])
        for samples in groups.values():
            if len(samples) > 1:
                collisions[path.stem].extend(sorted(samples))
    if not collisions:
        print(json.dumps({"checked_shards": len(paths), "collisions": {}}, ensure_ascii=False))
        return
    print(json.dumps({"checked_shards": len(paths),
                      "collisions": {topic: sorted(set(samples)) for topic, samples in sorted(collisions.items())}},
                     ensure_ascii=False))
    raise SystemExit(f"{len(collisions)} 个命题仍有仅数字不同的 state：{sorted(collisions)}")


if __name__ == "__main__":
    main()
