from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from diffsplat_tools.common import command_env, expand, require_repo, run
from diffsplat_tools.constants import MODEL_SPECS, ModelSpec


def run_inference(
    repo_dir: Path,
    out_dir: Path,
    spec: ModelSpec,
    tag: str,
    *,
    seed: int,
    gpu_id: int,
    half_precision: bool,
    common_args: Sequence[str],
    env: dict[str, str],
    dry_run: bool,
) -> None:
    require_repo(repo_dir, dry_run=dry_run)
    command = [
        sys.executable,
        spec.script,
        "--config_file",
        spec.config,
        "--tag",
        tag,
        "--output_dir",
        str(out_dir),
        "--seed",
        str(seed),
        "--gpu_id",
        str(gpu_id),
        "--allow_tf32",
    ]
    if half_precision:
        command.append("--half_precision")
    command.extend(common_args)
    run(command, cwd=repo_dir, env=env, dry_run=dry_run)


def text_args_to_command(args: argparse.Namespace) -> list[str]:
    command: list[str] = []
    if args.prompt:
        command.extend(["--prompt", args.prompt])
    if args.prompt_file:
        command.extend(["--prompt_file", str(expand(args.prompt_file))])
    if args.output_video_type:
        command.extend(["--output_video_type", args.output_video_type])
    if args.num_inference_timesteps is not None:
        command.extend(["--num_inference_timesteps", str(args.num_inference_timesteps)])
    if args.guidance_scale is not None:
        command.extend(["--guidance_scale", str(args.guidance_scale)])
    if getattr(args, "eval", False):
        command.append("--eval_text_cond")
    command.extend(args.extra)
    return command


def image_args_to_command(args: argparse.Namespace) -> list[str]:
    command: list[str] = []
    if args.image_path:
        command.extend(["--image_path", str(expand(args.image_path))])
    if args.image_dir:
        command.extend(["--image_dir", str(expand(args.image_dir))])
    if args.prompt is not None:
        command.extend(["--prompt", args.prompt])
    if args.elevation is not None:
        command.extend(["--elevation", str(args.elevation)])
    if args.output_video_type:
        command.extend(["--output_video_type", args.output_video_type])
    if args.guidance_scale is not None:
        command.extend(["--guidance_scale", str(args.guidance_scale)])
    if args.num_inference_timesteps is not None:
        command.extend(["--num_inference_timesteps", str(args.num_inference_timesteps)])
    if args.rembg_and_center:
        command.append("--rembg_and_center")
    if args.triangle_cfg_scaling:
        command.append("--triangle_cfg_scaling")
    if args.use_elevest:
        command.append("--use_elevest")
    command.extend(args.extra)
    return command


def run_text_inference(args: argparse.Namespace) -> None:
    repo_dir = expand(args.repo_dir)
    out_dir = expand(args.out_dir)
    spec = MODEL_SPECS[args.model]
    env = command_env(args.hf_home, args.torch_home, args.hf_endpoint)
    run_inference(
        repo_dir,
        out_dir,
        spec,
        spec.text_tag,
        seed=args.seed,
        gpu_id=args.gpu_id,
        half_precision=args.half_precision,
        common_args=text_args_to_command(args),
        env=env,
        dry_run=args.dry_run,
    )


def run_image_inference(args: argparse.Namespace) -> None:
    repo_dir = expand(args.repo_dir)
    out_dir = expand(args.out_dir)
    spec = MODEL_SPECS[args.model]
    env = command_env(args.hf_home, args.torch_home, args.hf_endpoint)
    run_inference(
        repo_dir,
        out_dir,
        spec,
        spec.image_tag,
        seed=args.seed,
        gpu_id=args.gpu_id,
        half_precision=args.half_precision,
        common_args=image_args_to_command(args),
        env=env,
        dry_run=args.dry_run,
    )


def run_render_loss_ablation(args: argparse.Namespace) -> None:
    repo_dir = expand(args.repo_dir)
    out_dir = expand(args.out_dir)
    spec = MODEL_SPECS[args.model]
    no_render_dir = out_dir / spec.no_render_tag
    if not no_render_dir.exists():
        raise SystemExit(
            "\n".join(
                [
                    f"The no-render checkpoint `{spec.no_render_tag}` was not found in {out_dir}.",
                    "The public Hugging Face release exposes the `__render` checkpoints, but not the no-render ablation weights.",
                    "If you obtain or train the no-render checkpoint, place it under the same out directory and rerun this command.",
                ]
            )
        )

    env = command_env(args.hf_home, args.torch_home, args.hf_endpoint)
    common_args = text_args_to_command(args)

    run_inference(
        repo_dir,
        out_dir,
        spec,
        spec.no_render_tag,
        seed=args.seed,
        gpu_id=args.gpu_id,
        half_precision=args.half_precision,
        common_args=common_args,
        env=env,
        dry_run=args.dry_run,
    )
    run_inference(
        repo_dir,
        out_dir,
        spec,
        spec.text_tag,
        seed=args.seed,
        gpu_id=args.gpu_id,
        half_precision=args.half_precision,
        common_args=common_args,
        env=env,
        dry_run=args.dry_run,
    )

