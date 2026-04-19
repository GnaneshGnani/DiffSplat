#!/usr/bin/env python3
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffsplat_tools.common import add_dry_run, add_hf_args, add_out_dir, add_repo_dir, command_env, expand
from diffsplat_tools.constants import MODEL_SPECS
from diffsplat_tools.downloads import download_checkpoints


def main() -> None:
    parser = argparse.ArgumentParser(description="Download pretrained DiffSplat checkpoints.")
    add_repo_dir(parser)
    add_out_dir(parser)
    add_hf_args(parser, include_torch_home=True)
    add_dry_run(parser)
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), default="sd15")
    parser.add_argument("--variant", choices=["text", "image", "both"], default="both")
    args = parser.parse_args()
    download_checkpoints(
        expand(args.repo_dir),
        expand(args.out_dir),
        model=args.model,
        variant=args.variant,
        env=command_env(args.hf_home, args.torch_home, args.hf_endpoint),
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
