"""Jev HTTP field contract, omitting confidence at the user's request."""
import json
import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator

MODEL_ID = "openjev-qwen3.5-0.8b-v1"
Structured = str | dict[str, JsonValue] | list[JsonValue]


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def strict_loads(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"重复 JSON 键: {key}")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError(f"非法 JSON 数字: {value}")
    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Choice(StrictModel):
    type: Literal["choice"]
    instructions: Structured
    criteria: dict[str, Structured | None] = Field(min_length=1, max_length=255)


class Score(StrictModel):
    type: Literal["score"]
    instructions: Structured
    criteria: list[Structured] = Field(min_length=2, max_length=10)


class NoulCriteria(StrictModel):
    true: Structured | None = None
    false: Structured | None = None


class Noul(StrictModel):
    type: Literal["noul"]
    instructions: Structured
    criteria: NoulCriteria | None = None


Question = Annotated[Choice | Score | Noul, Field(discriminator="type")]


class Request(StrictModel):
    model: str = Field(min_length=1)
    state: Structured
    questions: dict[str, Question] = Field(min_length=1)

    @field_validator("state", "questions")
    @classmethod
    def finite_json(cls, value):
        # Pydantic permits floats inside JsonValue, but JSON must not carry NaN/Inf.
        if isinstance(value, dict) and value and isinstance(next(iter(value.values())), BaseModel):
            value_to_check = {k: v.model_dump(exclude_none=True) for k, v in value.items()}
        else:
            value_to_check = value
        compact(value_to_check)
        return value


def parse_request(value):
    request = Request.model_validate(value)
    # Also catch non-finite values inside instructions / nested criteria.
    compact(request.model_dump())
    return request


def candidates(question):
    """Never put question IDs or Score indices in the model's input."""
    if isinstance(question, Choice):
        return list(question.criteria), [
            {"选项": key, "定义": value} for key, value in question.criteria.items()
        ]
    if isinstance(question, Score):
        return [str(i) for i in range(len(question.criteria))], list(question.criteria)
    criteria = question.criteria or NoulCriteria()
    return ["false", "true"], [
        {"判断": "否", "定义": criteria.false},
        {"判断": "是", "定义": criteria.true},
    ]


def probability(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("概率必须是数字，不能是布尔值或字符串")
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("概率必须是 [0, 1] 内有限值")
    return float(value)


def label_vector(question, label):
    keys, _ = candidates(question)
    if isinstance(question, Noul):
        if set(label) != {"noul"}:
            raise ValueError("Noul 标签只能包含 noul")
        yes = probability(label["noul"])
        return [1 - yes, yes]
    if set(label) != {"probabilities"} or set(label["probabilities"]) != set(keys):
        raise ValueError("概率键必须与 Choice 选项或 Score 下标完全一致")
    values = [probability(label["probabilities"][key]) for key in keys]
    if abs(sum(values) - 1) > 1e-6:
        raise ValueError(f"概率和不等于 1: {sum(values)}")
    return values


def answer_from_vector(question, values):
    keys, _ = candidates(question)
    values = [probability(v) for v in values]
    if len(values) != len(keys) or abs(sum(values) - 1) > 1e-5:
        raise ValueError("预测概率长度或概率和无效")
    # Normalize floating-point roundoff only; malformed labels never reach here.
    total = sum(values)
    values = [v / total for v in values]
    if isinstance(question, Noul):
        return {"type": "noul", "noul": values[1]}
    probabilities = dict(zip(keys, values))
    answer = {"type": question.type, "probabilities": probabilities}
    if isinstance(question, Choice):
        # Ties use the request's insertion order; this is a local policy.
        answer["choice"] = keys[max(range(len(values)), key=values.__getitem__)]
    else:
        answer["score"] = sum(i * value for i, value in enumerate(values))
        answer["legend"] = {str(i): c if isinstance(c, str) else compact(c)
                            for i, c in enumerate(question.criteria)}
    return answer


def build_response(request, labels, input_tokens=0, output_tokens=0):
    if isinstance(request, dict):
        request = parse_request(request)
    if set(labels) != set(request.questions):
        raise ValueError("标签问题 ID 与请求不一致")
    return {"model": MODEL_ID,
            "answers": {key: answer_from_vector(q, label_vector(q, labels[key]))
                        for key, q in request.questions.items()},
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens}}


def labels_from_response(request, response):
    if set(response) != {"model", "answers", "usage"}:
        raise ValueError("响应顶层必须为 model / answers / usage")
    if not isinstance(response["model"], str) or not response["model"]:
        raise ValueError("响应 model 必须为非空字符串")
    usage = response["usage"]
    if set(usage) != {"input_tokens", "output_tokens"} or any(
        type(v) is not int or v < 0 for v in usage.values()
    ):
        raise ValueError("usage 计数必须为非负整数")
    if set(response["answers"]) != set(request.questions):
        raise ValueError("响应的问题 ID 不一致")
    labels = {}
    for key, question in request.questions.items():
        answer = response["answers"][key]
        labels[key] = ({"noul": answer["noul"]} if isinstance(question, Noul)
                       else {"probabilities": answer["probabilities"]})
        expected = answer_from_vector(question, label_vector(question, labels[key]))
        if set(answer) != set(expected):
            raise ValueError(f"{key}: 响应字段不正确")
        for field, value in expected.items():
            actual = answer[field]
            if field == "probabilities":
                if set(actual) != set(value) or any(abs(probability(actual[k]) - v) > 1e-6 for k, v in value.items()):
                    raise ValueError(f"{key}.probabilities: 与问题或概率不一致")
            elif isinstance(value, float):
                if isinstance(actual, bool) or not isinstance(actual, (float, int)) or not math.isfinite(actual) or abs(value - actual) > 1e-6:
                    raise ValueError(f"{key}.{field}: 数值不符合本地派生规则")
            elif actual != value:
                raise ValueError(f"{key}.{field}: 与问题或概率不一致")
    return labels
