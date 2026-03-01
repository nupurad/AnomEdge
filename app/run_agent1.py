from __future__ import annotations

import argparse
import json

from app.agent1_stub import agent1_stub_from_scenario


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Agent1 using stub or model (if available).")
    parser.add_argument("--scenario", type=str, default="smoke", help="Stub scenario: normal|smoke|leak|jam")
    parser.add_argument("--image", type=str, default=None, help="Image path for model inference")
    parser.add_argument("--model-ref", type=str, default=None, help="Model id/path for Agent1ModelRunner")
    args = parser.parse_args()

    # Optional model path: only used if app.agent1 exists and imports cleanly.
    if args.image and args.model_ref:
        try:
            from app.agent1 import Agent1ModelRunner, agent1  # type: ignore

            runner = Agent1ModelRunner(args.model_ref)
            result = agent1(image_path=args.image, runner=runner, scenario=args.scenario)
            print(json.dumps(result, indent=2))
            return
        except Exception as e:
            print(f"Model path unavailable; falling back to stub. Reason: {e}")

    result = agent1_stub_from_scenario(scenario=args.scenario)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
