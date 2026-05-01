# DiffSplat One-Week Experiment Plan

This repo now has small scripts for a presentation-ready DiffSplat benchmark:

- text ablations over CFG, denoising steps, scheduler, seed
- image ablations over prompt/rembg/triangle-CFG/elevation
- model comparison across released DiffSplat backbones
- mini-training entrypoint for the no-render vs render-loss extension

Always start on a compute node or inside `srun`, then load modules and source the scratch env:

```bash
module load gcc/11.2.0 Python3/3.11.11 cuda/12.1.1
cd /fs/nexus-scratch/gnanesh/DiffSplat
source scripts/env_scratch.sh
source .venv/bin/activate
bash scripts/apply_upstream_patches.sh
```

## Prompt Subsets

```bash
python scripts/make_prompt_subset.py \
  --input data/t3bench/t3bench_prompt.txt \
  --output experiments/prompts/t3bench_30.txt \
  --count 30 \
  --mode first
```

## Text Ablations

Fast grid for tables:

```bash
python scripts/run_text_ablation_matrix.py \
  --models sd15 \
  --prompt-file experiments/prompts/t3bench_30.txt \
  --cfgs 3,5,7.5 \
  --steps 10,20,30 \
  --schedulers sde-dpmsolver++ \
  --seeds 0 \
  --output-dir out_ablation \
  --checkpoint-out-dir out \
  --gpu-id 0
```

Metrics run with CLIP/R-Precision/ImageReward:

```bash
python scripts/run_text_ablation_matrix.py \
  --models sd15 \
  --prompt-file experiments/prompts/t3bench_20.txt \
  --cfgs 7.5 \
  --steps 20 \
  --schedulers sde-dpmsolver++ \
  --seeds 0 \
  --output-dir out_ablation \
  --checkpoint-out-dir out \
  --gpu-id 0 \
  --eval
```

## Image Ablations

```bash
python scripts/run_image_ablation_matrix.py \
  --model sd15 \
  --image-path DiffSplat/assets/grm/frog.png \
  --prompt a_frog \
  --cfgs 1.5,2,3 \
  --steps 20 \
  --schedulers sde-dpmsolver++ \
  --elevations 10,20,30 \
  --variants baseline,no_triangle,no_rembg,empty_prompt \
  --output-dir out_ablation \
  --checkpoint-out-dir out \
  --gpu-id 0
```

## Model Comparison

Download extra checkpoints first:

```bash
python scripts/download_checkpoints.py --model pas --variant both
python scripts/download_checkpoints.py --model sd35m --variant both
```

Then run:

```bash
python scripts/run_text_ablation_matrix.py \
  --models sd15,pas,sd35m \
  --prompt-file experiments/prompts/t3bench_10.txt \
  --cfgs 7.5 \
  --steps 20 \
  --schedulers sde-dpmsolver++ \
  --seeds 0 \
  --output-dir out_ablation \
  --checkpoint-out-dir out \
  --gpu-id 0 \
  --continue-on-error
```

## Collect Results

```bash
python scripts/collect_ablation_results.py \
  --root out_ablation \
  --json results/ablation_summary.json \
  --csv results/ablation_summary.csv
```

## Slurm Shortcuts

```bash
sbatch experiments/slurm_text_ablation.sh
sbatch experiments/slurm_text_metrics.sh
sbatch experiments/slurm_image_ablation.sh
sbatch experiments/slurm_model_compare.sh
```

You can override the grids:

```bash
sbatch --export=ALL,PROMPT_COUNT=50,CFGS=3,5,7.5,STEPS=10,20,30 experiments/slurm_text_ablation.sh
```

## Mini Training

The official training code needs local GObjaverse-style parquet chunks. Once those exist, run:

```bash
sbatch --export=ALL,STAGE=no_render,TRAIN_DIR=/path/to/train_parquet,VAL_DIR=/path/to/val_parquet,MAX_STEPS=1000 \
  experiments/slurm_train_sd15_mini.sh

sbatch --export=ALL,STAGE=render,TRAIN_DIR=/path/to/train_parquet,VAL_DIR=/path/to/val_parquet,MAX_STEPS=1000 \
  experiments/slurm_train_sd15_mini.sh
```

For a presentation in one week, treat this as a mini-training feasibility extension unless the data is already prepared.
