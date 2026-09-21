"""Build the fixed 500-topic catalog and deterministic topic-level splits."""
import hashlib
from collections import Counter
from pathlib import Path
from openjev.protocol import compact

ROOT = Path(__file__).resolve().parents[1]


def main():
    rows = []
    domain = None
    for line in (ROOT / "data/topics_source.txt").read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            domain = line[2:]
        elif line.strip():
            fields = line.split("｜")
            assert len(fields) in (2, 3), line
            title, brief = fields[:2]
            task_family = fields[2] if len(fields) == 3 else "综合判断"
            assert task_family in {"综合判断", "情感分类", "意图识别"}, line
            i = len(rows) + 1
            rows.append({"topic_id": f"T{i:04d}", "domain": domain, "title": title,
                         "brief": brief, "task_family": task_family, "group": (i - 1) // 25 + 1,
                         "expected_samples": 10})
    assert len(rows) == 500, f"Expected 500, got {len(rows)}"
    assert len({row['title'] for row in rows}) == 500
    counts = Counter(row["domain"] for row in rows)
    assert len(counts) == 50 and set(counts.values()) == {10}, counts
    families = Counter(row["task_family"] for row in rows)
    assert families == {"综合判断": 450, "情感分类": 25, "意图识别": 25}, families
    for domain in counts:
        domain_rows = [r for r in rows if r["domain"] == domain]
        ordered = sorted(domain_rows, key=lambda r: hashlib.sha256(
            ("20260921:" + r["topic_id"]).encode()).hexdigest())
        for i, row in enumerate(ordered):
            row["split"] = "train" if i < 8 else "validation" if i == 8 else "test"
    (ROOT / "data/topics.jsonl").write_text(
        "".join(compact(row) + "\n" for row in rows), encoding="utf-8")
    md = ["# 500 个中文决策训练命题", "", "50 个领域 × 每领域 10 题；每题产出 10 条请求，共 5,000 条。",
          "", "专项命题：情感分类 25 题、意图识别 25 题，分布于 25 个领域；其余 450 题为综合判断。专项类型不是接口题型，仍使用 Choice / Score / Noul。",
          "", "分组：每组 25 题，共 20 组。数据集按命题划分，避免同题泄漏。", ""]
    for domain in counts:
        md.extend([f"## {domain}", "", "| 编号 | 命题 | 专项类型 | 简介 | 组 | 集合 |", "|---|---|---|---|---|---|"])
        for row in rows:
            if row["domain"] == domain:
                split = {"train": "训练", "validation": "验证", "test": "测试"}[row["split"]]
                md.append(f"| {row['topic_id']} | {row['title']} | {row['task_family']} | {row['brief']} | {row['group']} | {split} |")
        md.append("")
    (ROOT / "docs/2、500个中文命题.md").write_text("\n".join(md), encoding="utf-8")
    print(compact({"topics": len(rows), "domains": len(counts),
                   "task_families": dict(families), "splits": dict(Counter(row['split'] for row in rows))}))


if __name__ == "__main__":
    main()
