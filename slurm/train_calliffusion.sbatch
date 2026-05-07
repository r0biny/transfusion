#!/bin/bash
#SBATCH --job-name=calliffusion
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/home/ruohan.yu/Code/transfusion_results/slurm_out/%x-%j.out
#SBATCH --error=/home/ruohan.yu/Code/transfusion_results/slurm_out/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/Code/transfusion}"
DATASET_ROOT="${DATASET_ROOT:-$HOME/Code/calligraphy_project}"
RES_REPO="${RES_REPO:-$HOME/Code/transfusion_results}"
VAE_MODEL="${VAE_MODEL:-stabilityai/sd-vae-ft-mse}"

TRAIN_CSV="${TRAIN_CSV:-${DATASET_ROOT}/dataset/train_260402.csv}"
VAL_CSV="${VAL_CSV:-${DATASET_ROOT}/dataset/validation.csv}"

mkdir -p "${RES_REPO}" "${RES_REPO}/slurm_out"

source /apps/local/conda_init.sh
conda activate tf

cd "${PROJECT_ROOT}"

python -u train_calliffusion.py \
  --dataset_root "${DATASET_ROOT}" \
  --train_csv "${TRAIN_CSV}" \
  --val_csv "${VAL_CSV}" \
  --image_column path \
  --caption_template "一幅取法{source}的{style}单字书法作品，书写内容为{character}。" \
  --image_backend vae \
  --vae_model_name_or_path "${VAE_MODEL}" \
  --vae_downsample_factor 8 \
  --image_size 64 \
  --dim 512 \
  --depth 8 \
  --heads 8 \
  --dim_head 64 \
  --dropout 0.0 \
  --train_batch_size 256 \
  --learning_rate 3e-4 \
  --weight_decay 1e-2 \
  --gradient_accumulation_steps 1 \
  --max_grad_norm 0.5 \
  --mixed_precision bf16 \
  --num_workers 8 \
  --epochs 3000 \
  --ema_beta 0.99 \
  --sample_every 1000 \
  --sample_count 8 \
  --modality_steps 16 \
  --checkpoint_every 2000 \
  --output_root "${RES_REPO}"
