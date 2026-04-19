from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path
from typing import Sequence


def expand(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def print_command(command: Sequence[str], cwd: Path | None) -> None:
    where = f" (cwd={cwd})" if cwd else ""
    print(f"$ {shlex.join(command)}{where}")


def run(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    dry_run: bool = False,
) -> None:
    print_command(command, cwd)
    if dry_run:
        return
    subprocess.run(command, cwd=str(cwd) if cwd else None, env=env, check=True)


def command_env(hf_home: str, torch_home: str, hf_endpoint: str | None) -> dict[str, str]:
    env = os.environ.copy()
    env["HF_HOME"] = str(expand(hf_home))
    env["TORCH_HOME"] = str(expand(torch_home))
    if hf_endpoint:
        env["HF_ENDPOINT"] = hf_endpoint
    return env


def apply_hf_env(hf_home: str | None = None, hf_endpoint: str | None = None) -> None:
    if hf_home:
        os.environ["HF_HOME"] = str(expand(hf_home))
    if hf_endpoint:
        os.environ["HF_ENDPOINT"] = hf_endpoint


def require_repo(repo_dir: Path, dry_run: bool = False) -> None:
    if not repo_dir.exists() and not dry_run:
        raise SystemExit(
            f"DiffSplat repo not found at {repo_dir}. Run bootstrap first."
        )


def add_repo_dir(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-dir", default="./DiffSplat", help="Where to clone the official DiffSplat repo.")


def add_out_dir(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out-dir", default="./out", help="Directory for downloaded checkpoints and generated outputs.")


def add_hf_args(parser: argparse.ArgumentParser, *, include_torch_home: bool = False) -> None:
    parser.add_argument("--hf-home", default="~/.cache/huggingface", help="Hugging Face cache directory.")
    parser.add_argument("--hf-endpoint", default=os.environ.get("HF_ENDPOINT"), help="Optional Hugging Face endpoint mirror.")
    if include_torch_home:
        parser.add_argument("--torch-home", default="~/.cache/torch", help="Torch cache directory.")


def add_dry_run(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")


def add_shared_paths(parser: argparse.ArgumentParser) -> None:
    add_repo_dir(parser)
    add_out_dir(parser)
    add_hf_args(parser, include_torch_home=True)
    add_dry_run(parser)


def add_inference_args(parser: argparse.ArgumentParser, model_choices: Sequence[str]) -> None:
    add_shared_paths(parser)
    parser.add_argument("--model", choices=sorted(model_choices), default="sd15")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--half-precision", action="store_true")
    parser.add_argument("--output-video-type", default="gif")
    parser.add_argument("--guidance-scale", type=float)
    parser.add_argument("--num-inference-timesteps", type=int)
    parser.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="Extra upstream flags. Put them after `--`, for example `-- --save_ply`.",
    )
