#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a small prompt subset for fast DiffSplat experiments.")
    parser.add_argument("--input", default="data/t3bench/t3bench_prompt.txt")
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", choices=["first", "random", "stride"], default="first")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.input).expanduser().resolve()
    target = Path(args.output).expanduser().resolve()
    prompts = [line.strip() for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.count <= 0:
        raise SystemExit("--count must be positive")
    if args.count > len(prompts):
        raise SystemExit(f"Requested {args.count} prompts, but only {len(prompts)} exist in {source}")

    if args.mode == "first":
        selected = prompts[: args.count]
    elif args.mode == "random":
        rng = random.Random(args.seed)
        selected = rng.sample(prompts, args.count)
    else:
        step = max(1, len(prompts) // args.count)
        selected = prompts[::step][: args.count]

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(selected) + "\n", encoding="utf-8")
    print(f"Wrote {len(selected)} prompts to {target}")


if __name__ == "__main__":
    main()
