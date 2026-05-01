#!/usr/bin/env bash
# Run a smaller text suite with CLIP/R-Precision/ImageReward enabled.
# Submit with: sbatch experiments/slurm_text_metrics.sh

#SBATCH --job-name=diffsplat-text-metrics
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtx6000ada:1
#SBATCH --mem=100G
#SBATCH --time=08:00:00
#SBATCH --output=/fs/nexus-scratch/gnanesh/DiffSplat/logs/slurm/text_metrics_%j.log

set -euo pipefail

module load gcc/11.2.0 Python3/3.11.11 cuda/12.1.1

cd /fs/nexus-scratch/gnanesh/DiffSplat
mkdir -p logs/slurm results experiments/prompts
source scripts/env_scratch.sh
source .venv/bin/activate
bash scripts/apply_upstream_patches.sh

PROMPT_COUNT="${PROMPT_COUNT:-20}"
PROMPT_FILE="experiments/prompts/t3bench_metrics_${PROMPT_COUNT}.txt"

python scripts/make_prompt_subset.py \
  --input data/t3bench/t3bench_prompt.txt \
  --output "${PROMPT_FILE}" \
  --count "${PROMPT_COUNT}" \
  --mode first

python scripts/run_text_ablation_matrix.py \
  --models "${MODELS:-sd15}" \
  --prompt-file "${PROMPT_FILE}" \
  --cfgs "${CFGS:-7.5}" \
  --steps "${STEPS:-20}" \
  --schedulers "${SCHEDULERS:-sde-dpmsolver++}" \
  --seeds "${SEEDS:-0}" \
  --output-dir out_ablation \
  --checkpoint-out-dir out \
  --gpu-id 0 \
  --eval

python scripts/collect_ablation_results.py \
  --root out_ablation \
  --json results/text_metrics_summary.json \
  --csv results/text_metrics_summary.csv
