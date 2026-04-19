#!/usr/bin/env python3
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffsplat_tools.evaluation import evaluate_image_dirs


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute PSNR / SSIM / LPIPS on matching prediction and ground-truth images.")
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--json", help="Optional path to write a JSON summary.")
    parser.add_argument("--device", default="auto", help="`auto`, `cpu`, or a torch device like `cuda:0`.")
    parser.add_argument("--skip-lpips", action="store_true")
    parser.add_argument("--save-per-image", action="store_true")
    args = parser.parse_args()
    evaluate_image_dirs(args)


if __name__ == "__main__":
    main()
