#!/usr/bin/env python3
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffsplat_tools.common import add_dry_run, add_repo_dir, expand
from diffsplat_tools.setup import install_repo


def main() -> None:
    parser = argparse.ArgumentParser(description="Install the public DiffSplat inference dependencies.")
    add_repo_dir(parser)
    add_dry_run(parser)
    parser.add_argument("--torch-channel", default="cu121", help="PyTorch wheel channel, for example cu121 or cu118.")
    parser.add_argument("--skip-xformers", action="store_true")
    parser.add_argument("--skip-rasterizer", action="store_true")
    args = parser.parse_args()
    install_repo(
        expand(args.repo_dir),
        torch_channel=args.torch_channel,
        skip_xformers=args.skip_xformers,
        skip_rasterizer=args.skip_rasterizer,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
