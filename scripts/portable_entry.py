"""Run the distribution using only its bundled Python and dependencies."""
import json
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    if "--check" in sys.argv:
        from scripts.bootstrap_environment import inspect_environment
        state = inspect_environment()
        if not state["ready"]:
            print(json.dumps(state, ensure_ascii=False, indent=2))
            raise SystemExit(1)
        from openjev.webapp import available_checkpoints
        checkpoints = available_checkpoints(ROOT / "outputs/decision-full")
        if not checkpoints or checkpoints[-1].name != "checkpoint-001800":
            raise RuntimeError("缺少完整的第 1800 步模型，请完整解压压缩包。")
        print("运行环境和第 1800 步模型检查通过，正在打开网页。")
        return
    runpy.run_module("scripts.launch_web", run_name="__main__")


if __name__ == "__main__":
    main()
