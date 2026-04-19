from __future__ import annotations

import argparse

from diffsplat_tools.common import add_inference_args, add_shared_paths, command_env, expand
from diffsplat_tools.constants import MODEL_SPECS
from diffsplat_tools.downloads import download_checkpoints, download_gso, download_t3bench
from diffsplat_tools.evaluation import evaluate_image_dirs
from diffsplat_tools.inference import run_image_inference, run_render_loss_ablation, run_text_inference
from diffsplat_tools.setup import clone_repo, install_repo


def prepare_public(args: argparse.Namespace) -> None:
    repo_dir = expand(args.repo_dir)
    out_dir = expand(args.out_dir)
    data_dir = expand(args.data_dir)
    env = command_env(args.hf_home, args.torch_home, args.hf_endpoint)

    clone_repo(repo_dir, args.ref, args.dry_run)
    install_repo(
        repo_dir,
        torch_channel=args.torch_channel,
        skip_xformers=args.skip_xformers,
        skip_rasterizer=args.skip_rasterizer,
        dry_run=args.dry_run,
    )
    download_checkpoints(
        repo_dir,
        out_dir,
        model=args.model,
        variant=args.variant,
        env=env,
        dry_run=args.dry_run,
    )
    if not args.skip_t3bench:
        download_t3bench(data_dir, dry_run=args.dry_run)
    if not args.skip_gso:
        download_gso(
            data_dir,
            dry_run=args.dry_run,
            hf_home=args.hf_home,
            hf_endpoint=args.hf_endpoint,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal end-to-end wrapper for the DiffSplat course reproduction project.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    bootstrap = subparsers.add_parser("bootstrap", help="Clone the official DiffSplat repository.")
    bootstrap.add_argument("--repo-dir", default="./DiffSplat")
    bootstrap.add_argument("--ref", help="Optional branch, tag, or commit to checkout after cloning.")
    bootstrap.add_argument("--dry-run", action="store_true")

    install = subparsers.add_parser("install", help="Install the public DiffSplat inference dependencies.")
    add_shared_paths(install)
    install.add_argument("--torch-channel", default="cu121", help="PyTorch wheel channel, for example cu121 or cu118.")
    install.add_argument("--skip-xformers", action="store_true")
    install.add_argument("--skip-rasterizer", action="store_true")

    checkpoints = subparsers.add_parser("download-checkpoints", help="Download pretrained checkpoints from Hugging Face.")
    add_shared_paths(checkpoints)
    checkpoints.add_argument("--model", choices=sorted(MODEL_SPECS), default="sd15")
    checkpoints.add_argument("--variant", choices=["text", "image", "both"], default="both")

    data = subparsers.add_parser("download-data", help="Download T3Bench prompts and the rendered GSO benchmark subset.")
    add_shared_paths(data)
    data.add_argument("--data-dir", default="./data")
    data.add_argument("--skip-t3bench", action="store_true")
    data.add_argument("--skip-gso", action="store_true")

    prepare = subparsers.add_parser("prepare-public", help="Clone DiffSplat, install deps, and download the public project assets.")
    add_shared_paths(prepare)
    prepare.add_argument("--data-dir", default="./data")
    prepare.add_argument("--ref", help="Optional branch, tag, or commit to checkout after cloning.")
    prepare.add_argument("--torch-channel", default="cu121", help="PyTorch wheel channel, for example cu121 or cu118.")
    prepare.add_argument("--model", choices=sorted(MODEL_SPECS), default="sd15")
    prepare.add_argument("--variant", choices=["text", "image", "both"], default="both")
    prepare.add_argument("--skip-xformers", action="store_true")
    prepare.add_argument("--skip-rasterizer", action="store_true")
    prepare.add_argument("--skip-t3bench", action="store_true")
    prepare.add_argument("--skip-gso", action="store_true")

    text = subparsers.add_parser("text", help="Run text-conditioned inference.")
    add_inference_args(text, MODEL_SPECS.keys())
    text.add_argument("--prompt")
    text.add_argument("--prompt-file")
    text.add_argument("--eval", action="store_true", help="Enable CLIP / R-Precision / ImageReward evaluation.")

    image = subparsers.add_parser("image", help="Run image-conditioned inference.")
    add_inference_args(image, MODEL_SPECS.keys())
    image.add_argument("--image-path")
    image.add_argument("--image-dir")
    image.add_argument("--prompt", default="")
    image.add_argument("--elevation", type=float, default=20.0)
    image.add_argument("--rembg-and-center", action="store_true")
    image.add_argument("--triangle-cfg-scaling", action="store_true")
    image.add_argument("--use-elevest", action="store_true")

    ablation = subparsers.add_parser("ablate-render-loss", help="Compare no-render and render checkpoints, when both are available locally.")
    add_inference_args(ablation, MODEL_SPECS.keys())
    ablation.add_argument("--prompt", default="a_toy_robot")

    eval_image = subparsers.add_parser("eval-image", help="Compute PSNR / SSIM / LPIPS on matching prediction and ground-truth images.")
    eval_image.add_argument("--pred-dir", required=True)
    eval_image.add_argument("--gt-dir", required=True)
    eval_image.add_argument("--limit", type=int)
    eval_image.add_argument("--json", help="Optional path to write a JSON summary.")
    eval_image.add_argument("--device", default="auto", help="`auto`, `cpu`, or a torch device like `cuda:0`.")
    eval_image.add_argument("--skip-lpips", action="store_true")
    eval_image.add_argument("--save-per-image", action="store_true")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "bootstrap":
        clone_repo(expand(args.repo_dir), args.ref, args.dry_run)
        return
    if args.command == "install":
        install_repo(
            expand(args.repo_dir),
            torch_channel=args.torch_channel,
            skip_xformers=args.skip_xformers,
            skip_rasterizer=args.skip_rasterizer,
            dry_run=args.dry_run,
        )
        return
    if args.command == "download-checkpoints":
        download_checkpoints(
            expand(args.repo_dir),
            expand(args.out_dir),
            model=args.model,
            variant=args.variant,
            env=command_env(args.hf_home, args.torch_home, args.hf_endpoint),
            dry_run=args.dry_run,
        )
        return
    if args.command == "download-data":
        data_dir = expand(args.data_dir)
        if not args.skip_t3bench:
            download_t3bench(data_dir, dry_run=args.dry_run)
        if not args.skip_gso:
            download_gso(data_dir, dry_run=args.dry_run)
        return
    if args.command == "prepare-public":
        prepare_public(args)
        return
    if args.command == "text":
        run_text_inference(args)
        return
    if args.command == "image":
        run_image_inference(args)
        return
    if args.command == "ablate-render-loss":
        run_render_loss_ablation(args)
        return
    if args.command == "eval-image":
        evaluate_image_dirs(args)
        return
    raise SystemExit(f"Unknown command: {args.command}")
