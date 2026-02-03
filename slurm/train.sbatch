#!/bin/bash
#SBATCH --job-name=mnist_sheer
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/home/ruohan.yu/Code/transfusion/slurm/%x-%j.out
#SBATCH --error=/home/ruohan.yu/Code/transfusion/slurm/%x-%j.err

set -euo pipefail

source /apps/local/conda_init.sh
conda activate tf

cd /home/ruohan.yu/Code/transfusion
python train_mnist_augment.py