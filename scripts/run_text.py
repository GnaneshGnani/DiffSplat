#!/usr/bin/env python3
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffsplat_tools.common import add_inference_args
from diffsplat_tools.constants import MODEL_SPECS
from diffsplat_tools.inference import run_text_inference


def main() -> None:
    parser = argparse.ArgumentParser(description="Run text-conditioned DiffSplat inference.")
    add_inference_args(parser, MODEL_SPECS.keys())
    parser.add_argument("--prompt")
    parser.add_argument("--prompt-file")
    parser.add_argument("--eval", action="store_true", help="Enable CLIP / R-Precision / ImageReward evaluation.")
    args = parser.parse_args()
    run_text_inference(args)


if __name__ == "__main__":
    main()
