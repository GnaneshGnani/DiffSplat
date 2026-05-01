#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffsplat_tools.common import command_env
from diffsplat_tools.constants import MODEL_SPECS
from scripts.run_text_ablation_matrix import GSVAE_TAGS, GSRECON_TAG, csv_list, slug, symlink_dir


def prepare_checkpoint_layout(output_dir: Path, checkpoint_out_dir: Path, model: str, run_tag: str) -> None:
    spec = MODEL_SPECS[model]
    symlink_dir(checkpoint_out_dir / spec.image_tag / "checkpoints", output_dir / run_tag / "checkpoints")
    symlink_dir(checkpoint_out_dir / GSRECON_TAG, output_dir / GSRECON_TAG)
    symlink_dir(checkpoint_out_dir / GSVAE_TAGS[model], output_dir / GSVAE_TAGS[model])


def variant_flags(variant: str, prompt: str) -> tuple[str, bool, bool]:
    if variant == "baseline":
        return prompt, True, True
    if variant == "no_triangle":
        return prompt, True, False
    if variant == "no_rembg":
        return prompt, False, True
    if variant == "empty_prompt":
        return "", True, True
    raise SystemExit(f"Unknown image variant: {variant}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run image-conditioned DiffSplat ablations.")
    parser.add_argument("--model", default="sd15", choices=sorted(MODEL_SPECS))
    parser.add_argument("--image-path", default="DiffSplat/assets/grm/frog.png")
    parser.add_argument("--prompt", default="a_frog")
    parser.add_argument("--cfgs", default="1.5,2,3")
    parser.add_argument("--steps", default="20")
    parser.add_argument("--schedulers", default="sde-dpmsolver++")
    parser.add_argument("--elevations", default="20")
    parser.add_argument("--variants", default="baseline,no_triangle,no_rembg,empty_prompt")
    parser.add_argument("--output-dir", default="out_ablation")
    parser.add_argument("--checkpoint-out-dir", default="out")
    parser.add_argument("--repo-dir", default="DiffSplat")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--video-type", default="", help="Optional gif/mp4/fancy output; empty is fastest.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-experiments", type=int)
    parser.add_argument("--job-index", type=int, default=os.environ.get("SLURM_ARRAY_TASK_ID"))
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfgs = csv_list(args.cfgs, float)
    steps = csv_list(args.steps, int)
    schedulers = csv_list(args.schedulers)
    elevations = csv_list(args.elevations, float)
    variants = csv_list(args.variants)
    experiments = list(itertools.product(cfgs, steps, schedulers, elevations, variants))
    if args.max_experiments is not None:
        experiments = experiments[: args.max_experiments]
    if args.job_index is not None:
        index = int(args.job_index)
        if index < 0 or index >= len(experiments):
            raise SystemExit(f"job index {index} outside 0..{len(experiments) - 1}")
        experiments = [experiments[index]]

    output_dir = (ROOT / args.output_dir).resolve()
    checkpoint_out_dir = (ROOT / args.checkpoint_out_dir).resolve()
    repo_dir = (ROOT / args.repo_dir).resolve()
    image_path = (ROOT / args.image_path).resolve()
    env = command_env(
        str(ROOT / "out" / ".cache" / "huggingface"),
        str(ROOT / "out" / ".cache" / "torch"),
        os.environ.get("HF_ENDPOINT"),
    )

    spec = MODEL_SPECS[args.model]
    print(f"Running {len(experiments)} image experiments")
    for cfg, num_steps, scheduler, elevation, variant in experiments:
        prompt, rembg, triangle = variant_flags(variant, args.prompt)
        run_tag = (
            f"ablate_image_{args.model}_{variant}_cfg{slug(cfg)}_steps{num_steps}_"
            f"elev{slug(elevation)}_{slug(scheduler)}"
        )
        prepare_checkpoint_layout(output_dir, checkpoint_out_dir, args.model, run_tag)
        command = [
            sys.executable,
            spec.script,
            "--config_file",
            spec.config,
            "--tag",
            run_tag,
            "--output_dir",
            str(output_dir),
            "--seed",
            "0",
            "--gpu_id",
            str(args.gpu_id),
            "--allow_tf32",
            "--image_path",
            str(image_path),
            "--prompt",
            prompt,
            "--elevation",
            str(elevation),
            "--guidance_scale",
            str(cfg),
            "--num_inference_steps",
            str(num_steps),
            "--scheduler_type",
            scheduler,
            "--name_by_id",
        ]
        if rembg:
            command.append("--rembg_and_center")
        if triangle:
            command.append("--triangle_cfg_scaling")
        if args.video_type:
            command.extend(["--output_video_type", args.video_type])

        exp_dir = output_dir / run_tag
        manifest = {
            "kind": "image",
            "model": args.model,
            "variant": variant,
            "cfg": cfg,
            "steps": num_steps,
            "scheduler": scheduler,
            "elevation": elevation,
            "prompt": prompt,
            "image_path": str(image_path),
            "tag": run_tag,
            "command": command,
            "start_time": datetime.now(timezone.utc).isoformat(),
        }
        print("$ " + shlex.join(command))
        start = time.monotonic()
        if args.dry_run:
            returncode = 0
        else:
            try:
                completed = subprocess.run(command, cwd=repo_dir, env=env, check=not args.continue_on_error)
                returncode = completed.returncode
            except subprocess.CalledProcessError as exc:
                returncode = exc.returncode
                if not args.continue_on_error:
                    raise
        manifest["end_time"] = datetime.now(timezone.utc).isoformat()
        manifest["runtime_sec"] = round(time.monotonic() - start, 3)
        manifest["returncode"] = returncode
        exp_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
