#!/bin/bash
#SBATCH --job-name=quantus_shift
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mem=16G
#SBATCH --array=0-0
#SBATCH --output=logs/quantus_%A_%a.out
#SBATCH --error=logs/quantus_%A_%a.err

source ~/.bashrc
conda activate shift

python -m src.main \
  --mode train \
  --job-mode single \
  --job-type condition \
  --job-index ${SLURM_ARRAY_TASK_ID} \
  --stage quantus \
  --config src/configs/experiment_config.py