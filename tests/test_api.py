from fastapi.testclient import TestClient
from scripts.serve import create_app
from tests.test_protocol import example
from openjev.protocol import build_response, parse_request


class StubEngine:
    def predict(self, value):
        parse_request(value)
        _, labels = example()
        return build_response(value, labels)


def test_http_contract_and_bad_input():
    client = TestClient(create_app(StubEngine()))
    request, _ = example()
    result = client.post("/v1/systemone", json=request)
    assert result.status_code == 200
    assert set(result.json()) == {"model", "answers", "usage"}
    assert "confidence" not in result.text
    assert client.post("/v1/systemone", json={"state": "缺少问题"}).status_code == 422
    assert client.post("/v1/systemone", content='{"model":"a","model":"b"}').status_code == 422
