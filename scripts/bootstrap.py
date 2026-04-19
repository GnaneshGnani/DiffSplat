#!/usr/bin/env python3
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffsplat_tools.common import add_dry_run, add_repo_dir, expand
from diffsplat_tools.setup import clone_repo


def main() -> None:
    parser = argparse.ArgumentParser(description="Clone the official DiffSplat repository.")
    add_repo_dir(parser)
    parser.add_argument("--ref", help="Optional branch, tag, or commit to checkout after cloning.")
    add_dry_run(parser)
    args = parser.parse_args()
    clone_repo(expand(args.repo_dir), args.ref, args.dry_run)


if __name__ == "__main__":
    main()
