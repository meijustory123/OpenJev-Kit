import json
from pathlib import Path
import torch

from openjev.model import build_model, get_tokenizer, encode_candidates, score_encoded
from openjev.protocol import MODEL_ID, parse_request, answer_from_vector


class Engine:
    def __init__(self, checkpoint, device="cuda", max_length=None, micro_batch_size=1):
        checkpoint = Path(checkpoint)
        if not (checkpoint / "COMPLETE").exists():
            raise ValueError("请提供完整训练断点目录")
        config = json.loads((checkpoint / "decision_config.json").read_text(encoding="utf-8"))
        self.max_length = max_length or config["training"]["max_length"]
        self.micro_batch_size = micro_batch_size
        self.device = device
        if micro_batch_size < 1:
            raise ValueError("micro_batch_size 必须为正整数")
        self.tokenizer = get_tokenizer(checkpoint / "tokenizer")
        self.model = build_model(None, checkpoint, training=False)
        if device == "cpu":
            self.model = self.model.float()
        self.model = self.model.to(device).eval()

    @torch.inference_mode()
    def predict(self, raw_request):
        request = parse_request(raw_request)
        if request.model not in {MODEL_ID, "jev-latest"}:
            raise ValueError(f"本地仅接受 {MODEL_ID} 或兼容别名 jev-latest")
        answers, input_tokens = {}, 0
        # Each question sees the same state and none of the other questions.
        for key, question in request.questions.items():
            encoded, count = encode_candidates(self.tokenizer, request.state, question, self.max_length)
            input_tokens += count  # Actual encoded tokens, including repeated state.
            logits = score_encoded(self.model, self.tokenizer, encoded, self.device,
                                   self.micro_batch_size)
            values = logits.softmax(dim=0).cpu().tolist()
            answers[key] = answer_from_vector(question, values)
        return {"model": MODEL_ID, "answers": answers,
                "usage": {"input_tokens": input_tokens, "output_tokens": 0}}
