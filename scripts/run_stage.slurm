#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

source ~/.bashrc
conda activate shift

python -m src.main \
  --mode train \
  --job-mode single \
  --job-type ${JOB_TYPE} \
  --job-index ${SLURM_ARRAY_TASK_ID} \
  --stage ${STAGE} \
  --config ${CONFIG}