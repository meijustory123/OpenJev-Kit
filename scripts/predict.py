import argparse
from pathlib import Path
from openjev.inference import Engine
from openjev.protocol import strict_loads, compact


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--request", type=Path, default=Path("examples/request.json"))
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    engine = Engine(args.checkpoint, device=args.device)
    print(compact(engine.predict(strict_loads(args.request.read_text(encoding="utf-8")))))


if __name__ == "__main__":
    main()
