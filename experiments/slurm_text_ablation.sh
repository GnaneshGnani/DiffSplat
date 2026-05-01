#!/usr/bin/env bash
# Run a compact text-conditioned ablation suite for presentation tables.
# Submit with: sbatch experiments/slurm_text_ablation.sh

#SBATCH --job-name=diffsplat-text-ablate
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtx6000ada:1
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --output=/fs/nexus-scratch/gnanesh/DiffSplat/logs/slurm/text_ablation_%j.log

set -euo pipefail

module load gcc/11.2.0 Python3/3.11.11 cuda/12.1.1

cd /fs/nexus-scratch/gnanesh/DiffSplat
mkdir -p logs/slurm results experiments/prompts
source scripts/env_scratch.sh
source .venv/bin/activate
bash scripts/apply_upstream_patches.sh

PROMPT_COUNT="${PROMPT_COUNT:-30}"
MODELS="${MODELS:-sd15}"
CFGS="${CFGS:-3,5,7.5}"
STEPS="${STEPS:-10,20,30}"
SCHEDULERS="${SCHEDULERS:-sde-dpmsolver++}"
SEEDS="${SEEDS:-0}"
PROMPT_FILE="experiments/prompts/t3bench_${PROMPT_COUNT}.txt"

python scripts/make_prompt_subset.py \
  --input data/t3bench/t3bench_prompt.txt \
  --output "${PROMPT_FILE}" \
  --count "${PROMPT_COUNT}" \
  --mode first

python scripts/run_text_ablation_matrix.py \
  --models "${MODELS}" \
  --prompt-file "${PROMPT_FILE}" \
  --cfgs "${CFGS}" \
  --steps "${STEPS}" \
  --schedulers "${SCHEDULERS}" \
  --seeds "${SEEDS}" \
  --output-dir out_ablation \
  --checkpoint-out-dir out \
  --gpu-id 0

python scripts/collect_ablation_results.py \
  --root out_ablation \
  --json results/text_ablation_summary.json \
  --csv results/text_ablation_summary.csv
