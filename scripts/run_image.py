#!/usr/bin/env python3
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffsplat_tools.common import add_inference_args
from diffsplat_tools.constants import MODEL_SPECS
from diffsplat_tools.inference import run_image_inference


def main() -> None:
    parser = argparse.ArgumentParser(description="Run image-conditioned DiffSplat inference.")
    add_inference_args(parser, MODEL_SPECS.keys())
    parser.add_argument("--image-path")
    parser.add_argument("--image-dir")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--elevation", type=float, default=20.0)
    parser.add_argument("--rembg-and-center", action="store_true")
    parser.add_argument("--triangle-cfg-scaling", action="store_true")
    parser.add_argument("--use-elevest", action="store_true")
    args = parser.parse_args()
    run_image_inference(args)


if __name__ == "__main__":
    main()
