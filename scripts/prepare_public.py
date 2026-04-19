#!/usr/bin/env python3
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffsplat_tools.cli import prepare_public
from diffsplat_tools.common import add_dry_run, add_hf_args, add_out_dir, add_repo_dir
from diffsplat_tools.constants import MODEL_SPECS


def main() -> None:
    parser = argparse.ArgumentParser(description="Clone DiffSplat, install deps, and download the public project assets.")
    add_repo_dir(parser)
    add_out_dir(parser)
    add_hf_args(parser, include_torch_home=True)
    add_dry_run(parser)
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--ref", help="Optional branch, tag, or commit to checkout after cloning.")
    parser.add_argument("--torch-channel", default="cu121", help="PyTorch wheel channel, for example cu121 or cu118.")
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), default="sd15")
    parser.add_argument("--variant", choices=["text", "image", "both"], default="both")
    parser.add_argument("--skip-xformers", action="store_true")
    parser.add_argument("--skip-rasterizer", action="store_true")
    parser.add_argument("--skip-t3bench", action="store_true")
    parser.add_argument("--skip-gso", action="store_true")
    args = parser.parse_args()
    prepare_public(args)


if __name__ == "__main__":
    main()
