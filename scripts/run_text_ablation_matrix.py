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

GSRECON_TAG = "gsrecon_gobj265k_cnp_even4"
GSVAE_TAGS = {
    "sd15": "gsvae_gobj265k_sd",
    "pas": "gsvae_gobj265k_sdxl_fp16",
    "sd35m": "gsvae_gobj265k_sd3",
}


def csv_list(value: str, cast=str) -> list:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def slug(value: object) -> str:
    return str(value).replace(".", "p").replace("+", "p").replace("/", "-").replace("_", "-")


def symlink_dir(target: Path, link: Path) -> None:
    target = target.resolve()
    if not target.exists():
        raise SystemExit(f"Required checkpoint path is missing: {target}")
    if link.is_symlink():
        if link.resolve() != target:
            raise SystemExit(f"Refusing to replace symlink {link} -> {link.resolve()} with {target}")
        return
    if link.exists():
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target, target_is_directory=True)


def prepare_checkpoint_layout(output_dir: Path, checkpoint_out_dir: Path, model: str, run_tag: str, text: bool = True) -> None:
    spec = MODEL_SPECS[model]
    source_tag = spec.text_tag if text else spec.image_tag
    symlink_dir(checkpoint_out_dir / source_tag / "checkpoints", output_dir / run_tag / "checkpoints")
    symlink_dir(checkpoint_out_dir / GSRECON_TAG, output_dir / GSRECON_TAG)
    symlink_dir(checkpoint_out_dir / GSVAE_TAGS[model], output_dir / GSVAE_TAGS[model])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a text-conditioned DiffSplat ablation matrix.")
    parser.add_argument("--models", default="sd15", help="Comma list: sd15,pas,sd35m")
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--cfgs", default="3,5,7.5")
    parser.add_argument("--steps", default="10,20,30")
    parser.add_argument("--schedulers", default="sde-dpmsolver++")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--output-dir", default="out_ablation")
    parser.add_argument("--checkpoint-out-dir", default="out")
    parser.add_argument("--repo-dir", default="DiffSplat")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--eval", action="store_true", help="Enable CLIP/R-Precision/ImageReward.")
    parser.add_argument("--video-type", default="", help="Optional gif/mp4/fancy output; empty is fastest.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-experiments", type=int)
    parser.add_argument("--job-index", type=int, default=os.environ.get("SLURM_ARRAY_TASK_ID"))
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = csv_list(args.models)
    cfgs = csv_list(args.cfgs, float)
    steps = csv_list(args.steps, int)
    schedulers = csv_list(args.schedulers)
    seeds = csv_list(args.seeds, int)

    experiments = list(itertools.product(models, cfgs, steps, schedulers, seeds))
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
    prompt_file = (ROOT / args.prompt_file).resolve()
    env = command_env(
        str(ROOT / "out" / ".cache" / "huggingface"),
        str(ROOT / "out" / ".cache" / "torch"),
        os.environ.get("HF_ENDPOINT"),
    )

    print(f"Running {len(experiments)} text experiments")
    for model, cfg, num_steps, scheduler, seed in experiments:
        spec = MODEL_SPECS[model]
        run_tag = f"ablate_text_{model}_cfg{slug(cfg)}_steps{num_steps}_{slug(scheduler)}_seed{seed}"
        prepare_checkpoint_layout(output_dir, checkpoint_out_dir, model, run_tag, text=True)
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
            str(seed),
            "--gpu_id",
            str(args.gpu_id),
            "--allow_tf32",
            "--prompt_file",
            str(prompt_file),
            "--guidance_scale",
            str(cfg),
            "--num_inference_steps",
            str(num_steps),
            "--scheduler_type",
            scheduler,
            "--name_by_id",
        ]
        if args.eval:
            command.append("--eval_text_cond")
        if args.video_type:
            command.extend(["--output_video_type", args.video_type])

        exp_dir = output_dir / run_tag
        manifest = {
            "kind": "text",
            "model": model,
            "cfg": cfg,
            "steps": num_steps,
            "scheduler": scheduler,
            "seed": seed,
            "prompt_file": str(prompt_file),
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
