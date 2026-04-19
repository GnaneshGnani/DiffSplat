# DiffSplat Course Project

This repo is a minimal, modular wrapper around the official [DiffSplat](https://github.com/chenguolin/DiffSplat) codebase for the scope described in `DiffSplat_Proposal_Final.pdf`
The project is split into:

- `diffsplat_tools/`: shared Python modules
- `scripts/`: one focused script per task
- `diffsplat_project.py`: optional single-entry compatibility CLI

## What This Project Covers

- Public inference reproduction with pretrained checkpoints
- T3Bench text-prompt evaluation workflow
- GSO rendered benchmark download
- Text-conditioned and image-conditioned generation
- Image metric evaluation: PSNR, SSIM, LPIPS

## What It Does Not Cover

- Full local retraining of GSRecon, GSVAE, or DiffSplat from scratch
- G-Objaverse training data ingestion

The official training code depends on an internal HDFS-backed data layout, so this wrapper stays focused on the public reproduction path.

## Repo Layout

```text
diffsplat_tools/
  common.py
  constants.py
  downloads.py
  evaluation.py
  inference.py
  setup.py
scripts/
  bootstrap.py
  install.py
  download_checkpoints.py
  download_data.py
  prepare_public.py
  run_text.py
  run_image.py
  ablate_render_loss.py
  evaluate_image.py
diffsplat_project.py
```

## Quickstart

```bash
python3 scripts/prepare_public.py --model sd15
```

That will:

- clone `DiffSplat/`
- install the public dependencies
- download SD1.5 text and image checkpoints into `out/`
- download T3Bench prompts into `data/t3bench/`
- download the rendered GSO subset into `data/gso_rendered/`

The GSO download is large. If you want a lighter first setup, use `--skip-gso` and download it later with `download-data`.

If your CUDA wheels are not `cu121`, pick another channel:

```bash
python3 scripts/prepare_public.py --model sd15 --torch-channel cu118
```

If Hugging Face is slow:

```bash
python3 scripts/prepare_public.py --model sd15 --hf-endpoint https://hf-mirror.com
```

## Common Commands

Text-conditioned inference:

```bash
python3 scripts/run_text.py --model sd15 --prompt a_toy_robot
```

T3Bench batch evaluation:

```bash
python3 scripts/run_text.py \
  --model sd15 \
  --prompt-file data/t3bench/t3bench_prompt.txt \
  --eval
```

Image-conditioned inference:

```bash
python3 scripts/run_image.py \
  --model sd15 \
  --image-path DiffSplat/assets/grm/frog.png \
  --prompt a_frog \
  --elevation 20 \
  --rembg-and-center \
  --triangle-cfg-scaling \
  --guidance-scale 2
```

Pass extra upstream flags after `--`:

```bash
python3 scripts/run_text.py --model sd15 --prompt a_toy_robot -- --save_ply
```

Image metrics on matched prediction and ground-truth views:

```bash
python3 scripts/evaluate_image.py \
  --pred-dir path/to/pred_views \
  --gt-dir path/to/gt_views \
  --json results/gso_metrics.json
```

## Ablation Note

The public Hugging Face release includes the `__render` checkpoints, but the no-render ablation checkpoints are not publicly exposed in the same way. The command below will work if you already have the no-render checkpoint locally under `out/`:

```bash
python3 scripts/ablate_render_loss.py --model sd15 --prompt a_toy_robot
```

## Scripts

```bash
python3 scripts/bootstrap.py
python3 scripts/install.py --torch-channel cu121
python3 scripts/download_checkpoints.py --model sd15 --variant both
python3 scripts/download_data.py
python3 scripts/prepare_public.py --model sd15
python3 scripts/run_text.py --model sd15 --prompt a_toy_robot
python3 scripts/run_image.py --model sd15 --image-path DiffSplat/assets/grm/frog.png
python3 scripts/ablate_render_loss.py --model sd15 --prompt a_toy_robot
python3 scripts/evaluate_image.py --pred-dir PREDS --gt-dir GT
```

## Optional Combined CLI

If you still want one command entrypoint, the old interface is preserved:

```bash
python3 diffsplat_project.py prepare-public --model sd15
```
