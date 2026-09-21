import json
from pathlib import Path
from openjev.protocol import Request


def main():
    directory = Path("schemas")
    directory.mkdir(exist_ok=True)
    (directory / "request.schema.json").write_text(
        json.dumps(Request.model_json_schema(), ensure_ascii=False, indent=2), encoding="utf-8")
    probability = {"type": "number", "minimum": 0, "maximum": 1}
    def object_schema(properties, required=None):
        return {"type": "object", "properties": properties,
                "required": list(properties) if required is None else required,
                "additionalProperties": False}
    probabilities = {"type": "object", "minProperties": 1, "additionalProperties": probability}
    answer = {"oneOf": [
        object_schema({"type": {"const": "choice"}, "choice": {"type": "string"}, "probabilities": probabilities}),
        object_schema({"type": {"const": "score"}, "score": {"type": "number", "minimum": 0, "maximum": 9},
                       "legend": {"type": "object", "additionalProperties": {"type": "string"}},
                       "probabilities": probabilities}),
        object_schema({"type": {"const": "noul"}, "noul": probability}),
    ]}
    schema = object_schema({"model": {"type": "string"}, "answers": {"type": "object", "additionalProperties": answer},
                            "usage": object_schema({key: {"type": "integer", "minimum": 0}
                                                    for key in ["input_tokens", "output_tokens"]})})
    schema.update({"$schema": "https://json-schema.org/draft/2020-12/schema",
                   "title": "Jev-style response without confidence",
                   "$comment": "概率和、问题ID、score期望值等关联约束由 openjev.protocol 校验。"})
    (directory / "response.schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    print("schemas/request.schema.json; schemas/response.schema.json")


if __name__ == "__main__":
    main()
