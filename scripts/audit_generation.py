"""Audit generated training shards without altering agent-authored labels."""
import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from openjev.data import catalog, read_jsonl, validate_shards
from openjev.protocol import build_response, compact, parse_request


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
    parser.add_argument("--require-all", action="store_true")
    parser.add_argument("--tokens", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("reports/generation-audit.json"))
    args = parser.parse_args()
    topics = catalog()
    paths = sorted(Path("data/shards").glob("T*.jsonl"))
    rows = validate_shards(paths, require_all=args.require_all)
    tokenizer = None
    if args.tokens:
        from openjev.model import get_tokenizer, encode_candidates
        tokenizer = get_tokenizer("Qwen3.5-0.8B")
    maximum = json.loads(Path("configs/train.json").read_text(encoding="utf-8"))["max_length"]
    domains, groups = defaultdict(Counter), defaultdict(Counter)
    types, positions, vectors = Counter(), Counter(), defaultdict(set)
    normalized_states = defaultdict(list)
    alerts = []
    source_files = []
    draft_checks = Counter()
    for path in paths:
        draft_path = Path('data/drafts') / path.name
        source = {'topic_id': path.stem, 'shard_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        if not draft_path.exists():
            draft_checks['missing'] += 1
            alerts.append({'topic_id': path.stem, 'kind': 'missing_source_draft'})
        else:
            source['draft_sha256'] = hashlib.sha256(draft_path.read_bytes()).hexdigest()
            reconstructed = []
            for draft in read_jsonl(draft_path):
                item = dict(draft)
                labels = item.pop('labels')
                item['response'] = build_response(item['request'], labels)
                reconstructed.append(item)
            if reconstructed == read_jsonl(path):
                draft_checks['matching'] += 1
            else:
                draft_checks['mismatched'] += 1
                alerts.append({'topic_id': path.stem, 'kind': 'draft_shard_mismatch'})
        source_files.append(source)
    lengths = []
    length_rows = []
    token_totals = Counter()
    topic_structured = Counter()
    for row in rows:
        topic = topics[row["topic_id"]]
        domain, group = domains[topic["domain"]], groups[str(topic["group"])]
        domain["requests"] += 1
        group["requests"] += 1
        state = row["request"]["state"]
        if isinstance(state, (dict, list)):
            topic_structured[row["topic_id"]] += 1
        if isinstance(state, list):
            group["array_states"] += 1
        state_text = text_values(state)
        if re.search(r"待扩展|待生成|【占位|正确答案|本训练样本", state_text):
            alerts.append({"sample_id": row["sample_id"], "kind": "placeholder_or_answer_leak_marker"})
        if row["topic_id"] in state_text:
            alerts.append({"sample_id": row["sample_id"], "kind": "topic_id_in_state"})
        normalized = re.sub(r"\d+", "#", state_text)
        normalized = re.sub(r"\s+", "", normalized)
        normalized_states[hashlib.sha256(normalized.encode()).hexdigest()].append(row["sample_id"])
        sentences = [s.strip() for s in re.split(r"[。！？\n]", state_text) if len(s.strip()) >= 25]
        if len(sentences) >= 8 and len(set(sentences)) / len(sentences) < .75:
            alerts.append({"sample_id": row["sample_id"], "kind": "repeated_long_sentences",
                           "unique": len(set(sentences)), "total": len(sentences)})
        request = parse_request(row["request"])
        longest = 0
        request_tokens = 0
        for qid, question in request.questions.items():
            types[question.type] += 1
            domain["questions"] += 1
            if isinstance(question.instructions, (dict, list)):
                group["structured_instructions"] += 1
            choices = question.criteria
            values = list(choices.values()) if isinstance(choices, dict) else choices or []
            group["structured_candidate_descriptions"] += sum(isinstance(v, (dict, list)) for v in values)
            answer = row["response"]["answers"][qid]
            if question.type == "choice":
                keys = list(question.criteria)
                positions[str(keys.index(answer["choice"]))] += 1
                vectors["choice"].add(tuple(answer["probabilities"][k] for k in keys))
            elif question.type == "score":
                vectors["score"].add(tuple(answer["probabilities"].values()))
            else:
                vectors["noul"].add(answer["noul"])
                if row["slot"] == 10 and .35 <= answer["noul"] <= .65:
                    group["mixed_uncertain_noul"] += 1
                    instruction = text_values(question.instructions)
                    if re.search(r"是否.*(足以|足够|已确认|已记录|有.*证据)|能否.*确认", instruction):
                        alerts.append({"sample_id": row["sample_id"], "question_id": qid,
                                       "kind": "check_evidence_sufficiency_vs_unknown_fact",
                                       "instructions": instruction})
            if tokenizer is not None:
                encoded, candidate_tokens = encode_candidates(tokenizer, request.state, question, maximum)
                request_tokens += candidate_tokens
                longest = max(longest, max(map(len, encoded["input_ids"])))
        if tokenizer is not None:
            lengths.append(longest)
            label = "short" if longest <= 1024 else "medium" if longest <= 4096 else "long"
            domain[label] += 1
            group[label] += 1
            token_totals[label] += request_tokens
            domain["candidate_input_tokens"] += request_tokens
            length_rows.append({"sample_id": row["sample_id"], "max_candidate_tokens": longest,
                                "length_band": label, "candidate_input_tokens": request_tokens})
    for values in normalized_states.values():
        if len(values) > 1:
            splits = {topics[s.split("-")[0]]["split"] for s in values}
            alerts.append({"kind": "states_differ_only_in_numbers_or_whitespace",
                           "samples": values, "cross_split": len(splits) > 1})
    for path in paths:
        if not topic_structured[path.stem]:
            alerts.append({"topic_id": path.stem, "kind": "missing_structured_state"})
    for group, counts in groups.items():
        if counts["requests"] == 250:
            for key in ["array_states", "structured_instructions", "structured_candidate_descriptions", "mixed_uncertain_noul"]:
                if counts[key] < 5:
                    alerts.append({"group": int(group), "kind": "group_coverage_gap", "metric": key, "actual": counts[key]})
    if tokenizer is not None:
        for domain, counts in domains.items():
            if counts["requests"] == 100:
                for key in ["medium", "long"]:
                    if counts[key] < 1:
                        alerts.append({"domain": domain, "kind": "domain_length_coverage_gap", "metric": key})
    report = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "complete": len(paths) == 500, "shards": len(paths), "requests": len(rows),
        "questions": sum(types.values()), "question_types": dict(types),
        "splits": dict(Counter(topics[r["topic_id"]]["split"] for r in rows)),
        "topic_families": dict(Counter(topics[p.stem]["task_family"] for p in paths)),
        "domains": dict(domains), "groups": dict(groups),
        "choice_winner_positions_zero_based": dict(positions),
        "distinct_label_vectors": {k: len(v) for k, v in vectors.items()},
        "max_input_tokens": max(lengths) if lengths else None,
        "candidate_tokens_by_request_length_band": dict(token_totals),
        "total_candidate_input_tokens": sum(token_totals.values()) if lengths else None,
        "input_limit": maximum, "alerts_for_semantic_review": alerts,
        "draft_consistency": dict(draft_checks),
        "topic_catalog_sha256": hashlib.sha256(Path('data/topics.jsonl').read_bytes()).hexdigest(),
        "source_files": source_files,
        "scope": "Structure, exact duplication, selected heuristics and optional token lengths; semantic correctness requires separate review.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if length_rows:
        args.output.with_suffix(".lengths.jsonl").write_text(
            "".join(compact(r) + "\n" for r in length_rows), encoding="utf-8")
    print(compact({k: report[k] for k in ["complete", "shards", "requests", "questions", "max_input_tokens"]}
                  | {"review_alerts": len(alerts)}))


if __name__ == "__main__":
    main()
