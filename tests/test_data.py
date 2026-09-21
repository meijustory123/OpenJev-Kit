import copy
from pathlib import Path
import pytest

from openjev.data import (CASE_TYPES, SLOTS, catalog, validate_shards,
                         question_records, write_jsonl, read_jsonl)
from openjev.protocol import build_response


def fixture_rows(topic_id):
    """Structural fixtures only; never part of the real training collection."""
    rows = []
    for slot in range(1, 11):
        questions, labels = {}, {}
        for kind in SLOTS[slot]:
            questions[kind] = {"type": kind, "instructions": f"测试字段 {kind} 的格式"}
            if kind == "choice":
                questions[kind]["criteria"] = {"甲": "测试类别甲", "乙": "测试类别乙"}
                labels[kind] = {"probabilities": {"甲": .7, "乙": .3}}
            elif kind == "score":
                questions[kind]["criteria"] = ["测试低等级", "测试高等级"]
                labels[kind] = {"probabilities": {"0": .25, "1": .75}}
            else:
                labels[kind] = {"noul": .8}
        request = {"model": "jev-latest", "state": f"程序测试状态 {topic_id} 的槽位 {slot}，不得用于训练", "questions": questions}
        rows.append({"sample_id": f"{topic_id}-{slot:02d}", "topic_id": topic_id, "slot": slot,
                     "request": request, "response": build_response(request, labels),
                     "review": {"case_type": CASE_TYPES[slot], "label_source": "agent_synthetic",
                                "evidence": {k: "这是程序结构测试的说明，不是业务标注。" for k in questions}}})
    return rows


def test_complete_shard_unfolds_twelve_questions_without_metadata(tmp_path):
    path = tmp_path / "T0001.jsonl"
    write_jsonl(path, fixture_rows("T0001"))
    records = question_records(validate_shards([path]))
    assert len(records) == 12
    assert all("review" not in record and "response" not in record for record in records)
    assert sum(r["question"]["type"] == "choice" for r in records) == 5


def test_incomplete_shards_and_full_collection_requirement(tmp_path):
    path = tmp_path / "T0001.jsonl"
    rows = fixture_rows("T0001")
    write_jsonl(path, rows[:9])
    with pytest.raises(ValueError, match="10 条"):
        validate_shards([path])
    write_jsonl(path, rows)
    with pytest.raises(ValueError, match="尚未齐全"):
        validate_shards([path], require_all=True)


def test_semantic_response_mismatch_rejected(tmp_path):
    rows = fixture_rows("T0001")
    rows[4]["response"]["answers"]["score"]["score"] = .2
    path = tmp_path / "T0001.jsonl"
    write_jsonl(path, rows)
    with pytest.raises(ValueError, match="数值"):
        validate_shards([path])


def test_cross_split_state_leak_is_rejected(tmp_path):
    topics = catalog()
    train_id = next(k for k, r in topics.items() if r["split"] == "train")
    valid_id = next(k for k, r in topics.items() if r["split"] == "validation")
    a, b = fixture_rows(train_id), fixture_rows(valid_id)
    b[0]["request"]["state"] = a[0]["request"]["state"]
    b[0]["request"]["questions"]["choice"]["instructions"] += "变体"
    paths = [tmp_path / f"{train_id}.jsonl", tmp_path / f"{valid_id}.jsonl"]
    write_jsonl(paths[0], a)
    write_jsonl(paths[1], b)
    with pytest.raises(ValueError, match="跨集合泄漏"):
        validate_shards(paths)


def test_finalize_command_publishes_complete_shard_and_refuses_overwrite(tmp_path):
    import subprocess
    import sys
    from openjev.protocol import parse_request, labels_from_response
    rows = fixture_rows("T0001")
    for row in rows:
        row["labels"] = labels_from_response(parse_request(row["request"]), row.pop("response"))
    draft = tmp_path / "draft.jsonl"
    output = tmp_path / "shards"
    write_jsonl(draft, rows)
    command = [sys.executable, "-m", "scripts.finalize_shard", str(draft), "--output-dir", str(output)]
    first = subprocess.run(command, capture_output=True)
    assert first.returncode == 0, first.stderr.decode("utf-8", errors="replace")
    published = output / "T0001.jsonl"
    assert len(validate_shards([published])) == 10
    original = published.read_bytes()
    second = subprocess.run(command, capture_output=True)
    assert second.returncode != 0
    assert published.read_bytes() == original
