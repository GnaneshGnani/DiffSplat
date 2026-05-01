#!/usr/bin/env bash
# Mini-training entrypoint for the ambitious extension.
#
# This expects public GObjaverse-style parquet chunks to already exist locally.
# Run no-render first, then render fine-tuning:
#
#   sbatch --export=ALL,STAGE=no_render,TRAIN_DIR=/path/train,VAL_DIR=/path/val experiments/slurm_train_sd15_mini.sh
#   sbatch --export=ALL,STAGE=render,TRAIN_DIR=/path/train,VAL_DIR=/path/val experiments/slurm_train_sd15_mini.sh

#SBATCH --job-name=diffsplat-train-mini
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtx6000ada:4
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --output=/fs/nexus-scratch/gnanesh/DiffSplat/logs/slurm/train_sd15_mini_%j.log

set -euo pipefail

: "${TRAIN_DIR:?Set TRAIN_DIR to local GObjaverse parquet train directory}"
: "${VAL_DIR:?Set VAL_DIR to local GObjaverse parquet validation directory}"

module load gcc/11.2.0 Python3/3.11.11 cuda/12.1.1

cd /fs/nexus-scratch/gnanesh/DiffSplat
mkdir -p logs/slurm out_training wandb /tmp/test_dataset
printf 'offline\n' > wandb/token
source scripts/env_scratch.sh
source .venv/bin/activate
bash scripts/apply_upstream_patches.sh

export NUM_LOCAL_GPUS="${NUM_LOCAL_GPUS:-4}"
export MAIN_MACHINE_IP="${MAIN_MACHINE_IP:-127.0.0.1}"
export MAIN_MACHINE_PROT="${MAIN_MACHINE_PROT:-29500}"
export STAGE="${STAGE:-no_render}"
export MAX_STEPS="${MAX_STEPS:-1000}"
export NUM_WORKERS="${NUM_WORKERS:-8}"
export BATCH_SIZE="${BATCH_SIZE:-2}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-2}"
export GRAD_ACCUM="${GRAD_ACCUM:-2}"
export TRAIN_NAME="${TRAIN_NAME:-GObjaverse-train}"
export VAL_NAME="${VAL_NAME:-GObjaverse-val}"
export SAVE_FREQ="${SAVE_FREQ:-250}"
export EVAL_FREQ="${EVAL_FREQ:-250}"

cd DiffSplat

if [[ "${STAGE}" == "no_render" ]]; then
  TAG="${TAG:-gsdiff_gobj83k_sd15_mini_norender}"
  RENDER_ARGS=(opt.rendering_loss_prob=0)
elif [[ "${STAGE}" == "render" ]]; then
  TAG="${TAG:-gsdiff_gobj83k_sd15_mini_render}"
  LOAD_PRETRAINED_MODEL="${LOAD_PRETRAINED_MODEL:-gsdiff_gobj83k_sd15_mini_norender}"
  RENDER_ARGS=(
    opt.rendering_loss_prob=1
    opt.use_tiny_decoder=true
    --load_pretrained_model "${LOAD_PRETRAINED_MODEL}"
  )
else
  echo "Unknown STAGE=${STAGE}; expected no_render or render" >&2
  exit 2
fi

NUM_LOCAL_GPUS="${NUM_LOCAL_GPUS}" bash scripts/train.sh \
  src/train_gsdiff_sd.py configs/gsdiff_sd15.yaml "${TAG}" \
  --output_dir ../out_training \
  --max_train_steps "${MAX_STEPS}" \
  --max_val_steps 1 \
  --num_workers "${NUM_WORKERS}" \
  --gradient_accumulation_steps "${GRAD_ACCUM}" \
  --mixed_precision bf16 \
  --use_ema \
  --offline_wandb \
  train.batch_size_per_gpu="${BATCH_SIZE}" \
  val.batch_size_per_gpu="${VAL_BATCH_SIZE}" \
  train.save_freq="${SAVE_FREQ}" \
  train.eval_freq="${EVAL_FREQ}" \
  train.early_eval_freq="${EVAL_FREQ}" \
  opt.file_dir_train="${TRAIN_DIR}" \
  opt.file_name_train="${TRAIN_NAME}" \
  opt.file_dir_test="${VAL_DIR}" \
  opt.file_name_test="${VAL_NAME}" \
  "${RENDER_ARGS[@]}"
