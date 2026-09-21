import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from openjev.inference import Engine
from openjev.data import read_jsonl
from openjev.protocol import parse_request, candidates, label_vector, labels_from_response, compact


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", type=Path, default=Path("data/prepared/test_requests.jsonl"))
    p.add_argument("--output", type=Path, default=Path("outputs/evaluation.json"))
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    rows = read_jsonl(args.data)
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        raise ValueError("没有评估样本")
    engine = Engine(args.checkpoint)
    stats = defaultdict(lambda: defaultdict(list))
    predictions = []
    for row in rows:
        request = parse_request(row["request"])
        expected = labels_from_response(request, row["response"])
        response = engine.predict(row["request"])
        labels = labels_from_response(request, response)
        for key, q in request.questions.items():
            target, pred = label_vector(q, expected[key]), label_vector(q, labels[key])
            values = stats[q.type]
            values["brier_vs_soft_labels"].append(sum((a-b)**2 for a, b in zip(pred, target)))
            values["soft_cross_entropy"].append(-sum(t * math.log(max(p, 1e-12)) for p, t in zip(pred, target)))
            if q.type == "choice":
                gold = max(target)
                # Ambiguous ties are excluded from hard-label agreement.
                if sum(abs(v-gold) < 1e-8 for v in target) == 1:
                    values["top1_agreement"].append(float(max(range(len(pred)), key=pred.__getitem__) == target.index(gold)))
            elif q.type == "score":
                values["score_mae"].append(abs(sum(i * (a-b) for i, (a, b) in enumerate(zip(pred, target)))))
            else:
                values["noul_mae"].append(abs(pred[1]-target[1]))
        predictions.append({"sample_id": row["sample_id"], "response": response})
    report = {"requests": len(rows), "reference": "agent_synthetic; agreement is not real-world accuracy",
              "metrics": {kind: {name: {"mean": sum(v)/len(v), "count": len(v)}
                                 for name, v in metrics.items()} for kind, metrics in stats.items()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    from openjev.data import write_jsonl
    write_jsonl(args.output.with_suffix(".predictions.jsonl"), predictions)
    print(compact(report))


if __name__ == "__main__":
    main()
