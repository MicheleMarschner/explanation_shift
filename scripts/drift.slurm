#!/bin/bash
#SBATCH --job-name=drift_shift
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --array=0-0
#SBATCH --output=logs/drift_%A_%a.out
#SBATCH --error=logs/drift_%A_%a.err

source ~/.bashrc
conda activate shift

python -m src.main \
  --mode train \
  --job-mode single \
  --job-type condition \
  --job-index ${SLURM_ARRAY_TASK_ID} \
  --stage drift \
  --config src/configs/experiment_config.py