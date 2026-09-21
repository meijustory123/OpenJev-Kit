import argparse
from collections import Counter
from pathlib import Path
from openjev.data import catalog, validate_shards
from openjev.protocol import compact


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shards", type=Path, default=Path("data/shards"))
    p.add_argument("--group", type=int, choices=range(1, 21))
    p.add_argument("--require-all", action="store_true")
    args = p.parse_args()
    topics = catalog()
    paths = sorted(args.shards.glob("T*.jsonl"))
    if args.group:
        expected = {t for t, r in topics.items() if r["group"] == args.group}
        paths = [path for path in paths if path.stem in expected]
        missing = expected - {path.stem for path in paths}
        if missing:
            raise ValueError(f"组 {args.group} 缺少 {sorted(missing)}")
    rows = validate_shards(paths, require_all=args.require_all)
    print(compact({"shards": len(paths), "samples": len(rows),
                   "questions": sum(len(r['request']['questions']) for r in rows),
                   "splits": dict(Counter(topics[r['topic_id']]['split'] for r in rows))}))


if __name__ == "__main__":
    main()
