#!/bin/bash
#SBATCH --job-name=ref_shift
#SBATCH --account=sci-demelo-computer-vision
#SBATCH --partition=gpu-batch
#SBATCH --constraint=ARCH:X86
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --array=0-2
#SBATCH --output=logs/reference_%A_%a.out
#SBATCH --error=logs/reference_%A_%a.err

set -euxo pipefail

source /sc/home/michele.marschner/conda3/etc/profile.d/conda.sh
conda activate shift

python -m src.main \
  --mode train \
  --job-mode single \
  --job-type reference \
  --job-index ${SLURM_ARRAY_TASK_ID} \
  --stage reference \
  --config src/configs/experiments_config.py