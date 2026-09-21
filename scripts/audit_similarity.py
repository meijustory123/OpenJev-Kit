"""Find similar synthetic states for review; never rewrite labels automatically."""
import argparse
import json
import re
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

from openjev.data import catalog, read_jsonl
from scripts.audit_generation import text_values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=.82)
    parser.add_argument("--output", type=Path, default=Path("reports/generation-similarity.json"))
    args = parser.parse_args()
    topics = catalog()
    samples, shingles = [], []
    buckets = defaultdict(list)
    rng = np.random.default_rng(20260921)
    multipliers = rng.integers(1, 2**32, size=32, dtype=np.uint64) | np.uint64(1)
    increments = rng.integers(0, 2**32, size=32, dtype=np.uint64)
    modulus = np.uint64(4294967311)
    for path in sorted(Path("data/shards").glob("T*.jsonl")):
        for row in read_jsonl(path):
            text = re.sub(r"\d+", "数", text_values(row["request"]["state"]))
            text = re.sub(r"[^\w\u4e00-\u9fff]", "", text).lower()
            grams = {text[i:i+4] for i in range(max(0, len(text)-3))}
            if not grams:
                continue
            index = len(samples)
            samples.append(row["sample_id"])
            shingles.append(grams)
            hashes = np.array([zlib.crc32(g.encode()) for g in grams], dtype=np.uint64)
            signature = ((hashes[:, None] * multipliers + increments) % modulus).min(axis=0)
            for band in range(8):
                buckets[(band, tuple(map(int, signature[4*band:4*band+4])))].append(index)
    candidates = set()
    for members in buckets.values():
        for offset, left in enumerate(members):
            for right in members[offset+1:]:
                candidates.add((left, right))
    matches = []
    for left, right in candidates:
        a, b = shingles[left], shingles[right]
        intersection = len(a & b)
        similarity = intersection / (len(a) + len(b) - intersection)
        if similarity >= args.threshold:
            ids = [samples[left], samples[right]]
            splits = [topics[s.split("-")[0]]["split"] for s in ids]
            matches.append({"samples": ids, "jaccard_4gram": round(similarity, 4),
                            "splits": splits, "cross_split": splits[0] != splits[1]})
    matches.sort(key=lambda x: (not x["cross_split"], -x["jaccard_4gram"]))
    report = {"samples": len(samples), "candidate_pairs": len(candidates),
              "threshold": args.threshold, "matches": matches,
              "scope": "Approximate MinHash candidate retrieval and exact character-4gram Jaccard on normalized state text. Not exhaustive semantic duplicate detection; all matches require review."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"samples":len(samples),"review_pairs":len(matches),
                      "cross_split_pairs":sum(x["cross_split"] for x in matches)}))


if __name__ == "__main__":
    main()
