#!/usr/bin/env python3
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffsplat_tools.common import add_dry_run, add_hf_args, expand
from diffsplat_tools.downloads import download_gso, download_t3bench


def main() -> None:
    parser = argparse.ArgumentParser(description="Download T3Bench prompts and the rendered GSO benchmark subset.")
    add_hf_args(parser)
    add_dry_run(parser)
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--skip-t3bench", action="store_true")
    parser.add_argument("--skip-gso", action="store_true")
    args = parser.parse_args()

    data_dir = expand(args.data_dir)
    if not args.skip_t3bench:
        download_t3bench(data_dir, dry_run=args.dry_run)
    if not args.skip_gso:
        download_gso(
            data_dir,
            dry_run=args.dry_run,
            hf_home=args.hf_home,
            hf_endpoint=args.hf_endpoint,
        )


if __name__ == "__main__":
    main()
