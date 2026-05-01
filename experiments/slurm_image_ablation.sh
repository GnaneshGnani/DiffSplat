#!/usr/bin/env bash
# Run image-conditioned ablations: rembg, triangle CFG, prompt, elevation, CFG.
# Submit with: sbatch experiments/slurm_image_ablation.sh

#SBATCH --job-name=diffsplat-image-ablate
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtx6000ada:1
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=/fs/nexus-scratch/gnanesh/DiffSplat/logs/slurm/image_ablation_%j.log

set -euo pipefail

module load gcc/11.2.0 Python3/3.11.11 cuda/12.1.1

cd /fs/nexus-scratch/gnanesh/DiffSplat
mkdir -p logs/slurm results
source scripts/env_scratch.sh
source .venv/bin/activate
bash scripts/apply_upstream_patches.sh

python scripts/run_image_ablation_matrix.py \
  --model "${MODEL:-sd15}" \
  --image-path "${IMAGE_PATH:-DiffSplat/assets/grm/frog.png}" \
  --prompt "${PROMPT:-a_frog}" \
  --cfgs "${CFGS:-1.5,2,3}" \
  --steps "${STEPS:-20}" \
  --schedulers "${SCHEDULERS:-sde-dpmsolver++}" \
  --elevations "${ELEVATIONS:-10,20,30}" \
  --variants "${VARIANTS:-baseline,no_triangle,no_rembg,empty_prompt}" \
  --output-dir out_ablation \
  --checkpoint-out-dir out \
  --gpu-id 0

python scripts/collect_ablation_results.py \
  --root out_ablation \
  --json results/image_ablation_summary.json \
  --csv results/image_ablation_summary.csv
