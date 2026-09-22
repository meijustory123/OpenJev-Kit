"""Shared, predeclared metrics for local checkpoints and the official Jev API."""
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from openjev.protocol import candidates, label_vector, labels_from_response, parse_request, probability

METRIC_VERSION = "synthetic-agreement-v1"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def prediction_vectors(request, response):
    """Ignore official confidence/usage extensions, validate the decision payload.

    Official probabilities are rounded to two decimal places. Allow at most
    0.005 per candidate of accumulated rounding error, then normalize.
    Official confidence is neither a prediction nor a training target here.
    """
    if set(response["answers"]) != set(request.questions):
        raise ValueError("Response question IDs differ from the request")
    vectors = {}
    for key, question in request.questions.items():
        answer = response["answers"][key]
        if answer["type"] != question.type:
            raise ValueError("Response question type differs from the request")
        if question.type == "noul":
            yes = probability(answer["noul"])
            vectors[key] = [1 - yes, yes]
            continue
        keys, _ = candidates(question)
        probabilities = answer["probabilities"]
        if set(probabilities) != set(keys):
            raise ValueError("Response probability keys differ from criteria")
        values = [probability(probabilities[k]) for k in keys]
        total = sum(values)
        if total <= 0 or abs(total - 1) > len(values) * 0.005 + 1e-6:
            raise ValueError("Response probabilities do not sum to one")
        values = [v / total for v in values]
        if question.type == "choice":
            if answer["choice"] not in keys or max(values) - values[keys.index(answer["choice"])] > 0.010001 / total:
                raise ValueError("Response choice disagrees with probability argmax")
        else:
            score = answer["score"]
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
                raise ValueError("Invalid score")
            if not 0 <= score <= len(values) - 1:
                raise ValueError("Response score is outside the level range")
            # The API rounds score and probabilities separately: use its returned score.
        vectors[key] = values
    return vectors


def summarize(rows, predictions):
    by_id = {p["sample_id"]: p for p in predictions}
    if len(by_id) != len(predictions):
        raise ValueError("Duplicate predictions")
    if set(by_id) != {r["sample_id"] for r in rows}:
        raise ValueError("A complete prediction for every fixed test request is required")
    stats = defaultdict(lambda: defaultdict(list))
    counts = Counter()
    excluded = Counter()
    for row in rows:
        request = parse_request(row["request"])
        labels = labels_from_response(request, row["response"])
        response = by_id[row["sample_id"]]["response"]
        vectors = prediction_vectors(request, response)
        for key, question in request.questions.items():
            target = label_vector(question, labels[key])
            pred = vectors[key]
            values = stats[question.type]
            counts[question.type] += 1
            values["brier_vs_soft_labels"].append(sum((p - t) ** 2 for p, t in zip(pred, target)))
            values["soft_cross_entropy"].append(-sum(t * math.log(max(p, 1e-12)) for p, t in zip(pred, target)))
            if question.type in {"choice", "score"}:
                winners = [i for i, t in enumerate(target) if abs(t - max(target)) < 1e-8]
                if len(winners) == 1:
                    if question.type == "choice":
                        keys, _ = candidates(question)
                        correct = keys.index(response["answers"][key]["choice"]) == winners[0]
                    else:
                        # Predicted ties are abstentions, not correct by insertion order.
                        correct = pred[winners[0]] == max(pred) and sum(abs(p - max(pred)) < 1e-8 for p in pred) == 1
                    values["top1_agreement"].append(float(correct))
                else:
                    excluded[question.type + "_gold_ties"] += 1
            if question.type == "score":
                error = abs(response["answers"][key]["score"] - sum(i * t for i, t in enumerate(target)))
                values["score_mae"].append(error)
                values["normalized_score_mae"].append(error / (len(target) - 1))
            elif question.type == "noul":
                values["noul_mae"].append(abs(pred[1] - target[1]))
                if target[1] <= 0.1 or target[1] >= 0.9:
                    correct = (pred[1] > 0.5 and target[1] >= 0.9) or (pred[1] < 0.5 and target[1] <= 0.1)
                    values["definite_binary_agreement"].append(float(correct))
                else:
                    excluded["noul_nondefinite"] += 1
    latencies = sorted(p["seconds"] for p in predictions)
    hard = (stats["choice"].get("top1_agreement", []) + stats["score"].get("top1_agreement", [])
            + stats["noul"].get("definite_binary_agreement", []))
    return {
        "requests": len(rows), "questions": sum(counts.values()), "question_types": dict(counts),
        "actual_models": dict(Counter(p["response"]["model"] for p in predictions)),
        "excluded_from_hard_agreement": dict(excluded),
        "combined_hard_agreement": {"mean": sum(hard) / len(hard) if hard else None,
                                    "count": len(hard), "correct": int(sum(hard))},
        "metrics": {kind: {name: {"mean": sum(v) / len(v), "count": len(v),
                                  **({"correct": int(sum(v))} if "agreement" in name else {})}
                           for name, v in metrics.items()} for kind, metrics in stats.items()},
        "request_seconds": {"total": sum(latencies), "mean": sum(latencies) / len(latencies),
                            "median": latencies[len(latencies) // 2],
                            "p95": latencies[math.ceil(0.95 * len(latencies)) - 1]},
        "usage": {key: sum(p["response"].get("usage", {}).get(key, 0) for p in predictions)
                  for key in ["input_tokens", "output_tokens"]},
    }


METRIC_DEFINITIONS = {
    "reference": "agent_synthetic soft labels; agreement is not real-world or human-verified accuracy",
    "combined_hard_agreement": "Micro-average over eligible Choice top1, Score top1 and definite Noul binary judgments; ambiguous references excluded as specified below; not an average of the three type percentages",
    "choice_top1": "Returned choice equals the unique highest-probability reference option; reference ties excluded",
    "score_top1": "Unique predicted most-probable level equals the unique reference most-probable level; reference ties excluded; predicted ties count as wrong",
    "score_mae": "Mean absolute error of returned expected score vs reference expected score (zero-based levels)",
    "normalized_score_mae": "Score absolute error divided by the question's maximum level index",
    "noul_mae": "Mean absolute error of noul vs reference noul over all Noul questions",
    "noul_definite_binary": "Only reference <=0.1 or >=0.9; prediction <0.5 means false, >0.5 true, exact 0.5 counts as wrong",
    "soft_cross_entropy": "-sum(target * log(max(prediction, 1e-12))); natural logarithm; lower is better",
    "brier_vs_soft_labels": "Sum of squared probability errors per question, averaged; lower is better",
    "rounding": "Official API rounds probabilities/score separately. Allow sum error <=0.005 per candidate and normalize distributions; use returned choice and score for their metrics. Rounded prediction ties count as wrong for Score top1. Score is validated against level bounds, not rounded probability expectation.",
    "confidence": "Official confidence ignored; no confidence metric or local output field",
}
