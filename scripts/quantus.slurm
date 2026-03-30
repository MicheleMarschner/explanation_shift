#!/bin/bash
#SBATCH --job-name=quantus_shift
#SBATCH --account=sci-demelo-computer-vision
#SBATCH --partition=gpu-batch
#SBATCH --constraint=ARCH:X86
#####SBATCH --nodelist=gx24
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --array=0-79
#SBATCH --output=logs/quantus_%A_%a.out
#SBATCH --error=logs/quantus_%A_%a.err

set -euxo pipefail

source /sc/home/michele.marschner/conda3/etc/profile.d/conda.sh
conda activate shift

python -m src.main \
  --mode train \
  --job-mode single \
  --job-type condition \
  --job-index ${SLURM_ARRAY_TASK_ID} \
  --stage quantus \
  --config src/configs/experiments_config.py