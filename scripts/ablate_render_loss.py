#!/usr/bin/env python3
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffsplat_tools.common import add_inference_args
from diffsplat_tools.constants import MODEL_SPECS
from diffsplat_tools.inference import run_render_loss_ablation


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare no-render and render DiffSplat checkpoints.")
    add_inference_args(parser, MODEL_SPECS.keys())
    parser.add_argument("--prompt", default="a_toy_robot")
    args = parser.parse_args()
    run_render_loss_ablation(args)


if __name__ == "__main__":
    main()
