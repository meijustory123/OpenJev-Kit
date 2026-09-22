import copy

import pytest

from openjev.benchmark import prediction_vectors, summarize
from openjev.protocol import build_response, parse_request


def sample():
    request = {"model": "jev-latest", "state": "test", "questions": {
        "choice": {"type": "choice", "instructions": "classify", "criteria": {"a": "A", "b": "B"}},
        "score": {"type": "score", "instructions": "rate", "criteria": ["low", "mid", "high"]},
        "noul": {"type": "noul", "instructions": "true?"},
    }}
    response = build_response(request, {
        "choice": {"probabilities": {"a": 0.8, "b": 0.2}},
        "score": {"probabilities": {"0": 0.1, "1": 0.2, "2": 0.7}},
        "noul": {"noul": 0.98},
    })
    return {"sample_id": "test-01", "request": request, "response": response}


def test_shared_metrics_use_returned_score_and_ignore_official_confidence():
    row = sample()
    response = copy.deepcopy(row["response"])
    response["model"] = "jev-test"
    response["answers"]["score"].update(probabilities={"0": 0.04, "1": 0.23, "2": 0.73}, score=1.7, confidence=0.55)
    response["answers"]["choice"]["confidence"] = 0.2
    response["answers"]["noul"]["noul"] = 0.9
    metrics = summarize([row], [{"sample_id": "test-01", "response": response, "seconds": 1}])["metrics"]
    assert metrics["choice"]["top1_agreement"] == {"mean": 1, "count": 1, "correct": 1}
    assert metrics["score"]["score_mae"]["mean"] == pytest.approx(0.1)
    assert metrics["score"]["normalized_score_mae"]["mean"] == pytest.approx(0.05)
    assert metrics["noul"]["noul_mae"]["mean"] == pytest.approx(0.08)
    assert metrics["noul"]["definite_binary_agreement"]["correct"] == 1


def test_rounding_is_bounded_and_bad_ids_are_rejected():
    row = sample()
    request = parse_request(row["request"])
    response = copy.deepcopy(row["response"])
    response["answers"]["score"]["probabilities"] = {"0": 0.03, "1": 0.23, "2": 0.73}
    assert sum(prediction_vectors(request, response)["score"]) == pytest.approx(1)
    response["answers"]["score"]["probabilities"]["0"] = 0.3
    with pytest.raises(ValueError, match="sum"):
        prediction_vectors(request, response)
    response["answers"].pop("score")
    with pytest.raises(ValueError, match="IDs"):
        prediction_vectors(request, response)


def test_ambiguous_reference_is_excluded_only_from_hard_metrics():
    row = sample()
    row["response"]["answers"]["choice"].update(probabilities={"a": 0.5, "b": 0.5})
    row["response"]["answers"]["noul"]["noul"] = 0.5
    result = summarize([row], [{"sample_id": "test-01", "response": row["response"], "seconds": 1}])
    assert result["excluded_from_hard_agreement"] == {"choice_gold_ties": 1, "noul_nondefinite": 1}
    assert "top1_agreement" not in result["metrics"]["choice"]
    assert "definite_binary_agreement" not in result["metrics"]["noul"]
    assert result["metrics"]["noul"]["noul_mae"]["count"] == 1


def test_report_rejects_missing_and_duplicate_predictions():
    row = sample()
    with pytest.raises(ValueError, match="complete"):
        summarize([row], [])
    prediction = {"sample_id": "test-01", "response": row["response"], "seconds": 1}
    with pytest.raises(ValueError, match="Duplicate"):
        summarize([row], [prediction, prediction])


def test_exact_half_is_not_a_correct_definite_noul_prediction():
    row = sample()
    response = copy.deepcopy(row["response"])
    response["answers"]["noul"]["noul"] = 0.5
    metrics = summarize([row], [{"sample_id": "test-01", "response": response, "seconds": 1}])["metrics"]
    assert metrics["noul"]["definite_binary_agreement"]["correct"] == 0
