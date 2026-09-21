import copy
import json
from pathlib import Path

import pytest
from openjev.protocol import parse_request, build_response, labels_from_response, strict_loads
from openjev.data import catalog


def example():
    request = json.loads(Path("examples/request.json").read_text(encoding="utf-8"))
    labels = {"处理部门": {"probabilities": {"退换货": .9, "物流": .08, "财务": .02}},
              "信息完整度": {"probabilities": {"0": .1, "1": .2, "2": .7}},
              "要求换货": {"noul": .95}}
    return request, labels


def test_response_has_no_confidence_and_correct_expected_score():
    request, labels = example()
    response = build_response(request, labels)
    assert response["answers"]["处理部门"]["choice"] == "退换货"
    assert response["answers"]["信息完整度"]["score"] == pytest.approx(1.6)
    assert response["answers"]["信息完整度"]["legend"] == {str(i): c for i, c in enumerate(request["questions"]["信息完整度"]["criteria"])}
    assert "confidence" not in json.dumps(response)
    recovered = labels_from_response(parse_request(request), response)
    assert recovered["要求换货"] == labels["要求换货"]


@pytest.mark.parametrize("bad", [True, "0.9", -0.1, 1.1, float("nan"), float("inf")])
def test_invalid_probabilities_rejected(bad):
    request, labels = example()
    labels["要求换货"]["noul"] = bad
    with pytest.raises(ValueError):
        build_response(request, labels)


def test_wrong_probability_keys_sum_and_response_extra_rejected():
    request, labels = example()
    labels["信息完整度"]["probabilities"]["1"] = .5
    with pytest.raises(ValueError):
        build_response(request, labels)
    request, labels = example()
    response = build_response(request, labels)
    response["answers"]["处理部门"]["confidence"] = .9
    with pytest.raises(ValueError):
        labels_from_response(parse_request(request), response)


def test_float_roundoff_is_accepted_but_wrong_choice_is_not():
    request, labels = example()
    response = build_response(request, labels)
    response["answers"]["处理部门"]["probabilities"]["物流"] += 1e-10
    labels_from_response(parse_request(request), response)
    response["answers"]["处理部门"]["choice"] = "财务"
    with pytest.raises(ValueError, match="choice"):
        labels_from_response(parse_request(request), response)


def test_duplicate_keys_and_non_json_numbers_rejected():
    with pytest.raises(ValueError):
        strict_loads('{"a":1,"a":2}')
    with pytest.raises(ValueError):
        strict_loads('{"a":NaN}')


def test_typesafe_limits_and_structured_values():
    request, _ = example()
    request["questions"]["处理部门"]["criteria"] = {str(i): None for i in range(255)}
    parse_request(request)
    request["questions"]["处理部门"]["criteria"]["256"] = None
    with pytest.raises(ValueError):
        parse_request(request)
    request, _ = example()
    request["questions"]["信息完整度"]["criteria"] = ["单级"]
    with pytest.raises(ValueError):
        parse_request(request)
    request, _ = example()
    request["questions"]["要求换货"]["instructions"] = {"问题": "是否要求更换？", "说明": ["按原话"]}
    request["questions"]["要求换货"]["criteria"] = {"true": "要求更换", "false": "明确拒绝更换"}
    parse_request(request)


def test_catalog_exact_counts_and_no_split_leak():
    topics = catalog()
    assert len(topics) == 500
    assert len({r["title"] for r in topics.values()}) == 500
    assert len({r["domain"] for r in topics.values()}) == 50
    for domain in {r["domain"] for r in topics.values()}:
        rows = [r for r in topics.values() if r["domain"] == domain]
        assert [sum(r["split"] == s for r in rows) for s in ["train", "validation", "test"]] == [8, 1, 1]
