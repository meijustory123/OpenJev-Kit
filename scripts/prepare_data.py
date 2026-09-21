import argparse
from collections import Counter
from pathlib import Path
from openjev.data import (catalog, validate_shards, question_records, write_jsonl)
from openjev.protocol import compact


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shards", type=Path, default=Path("data/shards"))
    p.add_argument("--output", type=Path, default=Path("data/prepared"))
    p.add_argument("--allow-partial", action="store_true",
                   help="仅用于试跑；正式训练必须具备全部 500 题")
    args = p.parse_args()
    topics = catalog()
    rows = validate_shards(sorted(args.shards.glob("T*.jsonl")), require_all=not args.allow_partial)
    manifest = {"partial": args.allow_partial, "samples": len(rows), "splits": {}}
    for split in ["train", "validation", "test"]:
        selected = [r for r in rows if topics[r["topic_id"]]["split"] == split]
        records = question_records(selected)
        write_jsonl(args.output / f"{split}.jsonl", records)
        write_jsonl(args.output / f"{split}_requests.jsonl", selected)
        manifest["splits"][split] = {"requests": len(selected), "questions": len(records),
                                     "types": dict(Counter(r['question']['type'] for r in records))}
    (args.output / "manifest.json").write_text(compact(manifest), encoding="utf-8")
    print(compact(manifest))


if __name__ == "__main__":
    main()
