import hashlib
import json
from collections import Counter
from pathlib import Path

from openjev.protocol import (compact, parse_request, labels_from_response,
                             candidates, label_vector, strict_loads)

ROOT = Path(__file__).resolve().parents[1]
SLOTS = {
    1: ["choice"], 2: ["choice"], 3: ["choice"], 4: ["choice"],
    5: ["score"], 6: ["score"], 7: ["score"],
    8: ["noul"], 9: ["noul"], 10: ["choice", "score", "noul"],
}
CASE_TYPES = {1: "明确匹配", 2: "易混类别", 3: "信息不足", 4: "否定或干扰",
              5: "低等级", 6: "中间或边界等级", 7: "高等级",
              8: "明确肯定", 9: "明确否定", 10: "混合独立问题"}


def read_jsonl(path):
    rows = []
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"{path}:{i}: 不允许空行")
        try:
            rows.append(strict_loads(line))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{path}:{i}: {exc}") from exc
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(compact(row) + "\n" for row in rows), encoding="utf-8")
    tmp.replace(path)


def catalog():
    return {r["topic_id"]: r for r in read_jsonl(ROOT / "data/topics.jsonl")}


def validate_row(row, topics):
    required = {"sample_id", "topic_id", "slot", "request", "response", "review"}
    if set(row) != required:
        raise ValueError(f"样本顶层必须是 {sorted(required)}")
    topic_id, slot = row["topic_id"], row["slot"]
    if topic_id not in topics or type(slot) is not int or slot not in SLOTS:
        raise ValueError("命题编号或槽位无效")
    if row["sample_id"] != f"{topic_id}-{slot:02d}":
        raise ValueError("sample_id 与命题/槽位不一致")
    request = parse_request(row["request"])
    types = sorted(q.type for q in request.questions.values())
    if types != sorted(SLOTS[slot]):
        raise ValueError(f"槽位 {slot} 应为 {SLOTS[slot]}，实际 {types}")
    labels = labels_from_response(request, row["response"])
    review = row["review"]
    if set(review) != {"case_type", "evidence", "label_source"}:
        raise ValueError("review 字段须为 case_type、evidence、label_source")
    if review["case_type"] != CASE_TYPES[slot]:
        raise ValueError("case_type 与槽位不一致")
    if review["label_source"] != "agent_synthetic":
        raise ValueError("本批标签来源必须标记 agent_synthetic")
    if set(review["evidence"]) != set(request.questions):
        raise ValueError("每个问题均需独立 evidence")
    if any(not isinstance(e, str) or len(e.strip()) < 8 for e in review["evidence"].values()):
        raise ValueError("evidence 必须为至少 8 字的简短依据")
    return request, labels


def validate_shards(paths, require_all=False):
    topics = catalog()
    rows, seen_ids, seen_states, seen_requests = [], set(), {}, {}
    for path in paths:
        shard = read_jsonl(path)
        topic_id = Path(path).stem
        if topic_id not in topics:
            raise ValueError(f"未知分片文件名: {path}")
        if len(shard) != 10 or {r.get("slot") for r in shard} != set(range(1, 11)):
            raise ValueError(f"{path}: 必须恰好 10 条且槽位为 1—10")
        for row in shard:
            if row.get("topic_id") != topic_id:
                raise ValueError(f"{path}: 样本写入了其他命题分片")
            validate_row(row, topics)
            if row["sample_id"] in seen_ids:
                raise ValueError(f"重复样本 ID: {row['sample_id']}")
            seen_ids.add(row["sample_id"])
            # Metadata and question IDs cannot hide exact duplicated requests.
            request = row["request"]
            normalized = {"state": request["state"],
                          "questions": sorted([compact(q) for q in request["questions"].values()])}
            digest = hashlib.sha256(json.dumps(normalized, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
            if digest in seen_requests:
                raise ValueError(f"重复请求: {row['sample_id']} 与 {seen_requests[digest]}")
            seen_requests[digest] = row["sample_id"]
            state_hash = hashlib.sha256(json.dumps(request["state"], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
            split = topics[topic_id]["split"]
            if state_hash in seen_states and seen_states[state_hash] != split:
                raise ValueError(f"相同 state 跨集合泄漏: {row['sample_id']}")
            seen_states[state_hash] = split
            rows.append(row)
    if require_all and {r["topic_id"] for r in rows} != set(topics):
        missing = sorted(set(topics) - {r["topic_id"] for r in rows})
        raise ValueError(f"500 个命题尚未齐全，缺少 {len(missing)} 题: {missing[:10]}")
    if not rows:
        raise ValueError("没有样本；不会生成空训练文件")
    return rows


def question_records(rows):
    result = []
    for row in rows:
        request = parse_request(row["request"])
        labels = labels_from_response(request, row["response"])
        for key, question in request.questions.items():
            result.append({"sample_id": row["sample_id"], "topic_id": row["topic_id"],
                           "question_id": key, "state": request.state,
                           "question": question.model_dump(exclude_none=True),
                           "target": label_vector(question, labels[key])})
    return result
