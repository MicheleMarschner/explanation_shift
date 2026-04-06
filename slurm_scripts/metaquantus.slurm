#!/bin/bash
#SBATCH --job-name=metaquantus_test
#SBATCH --account=sci-demelo-computer-vision
#SBATCH --partition=gpu-batch
#SBATCH --constraint=ARCH:X86
#SBATCH --gres=gpu:b200:4
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/sc/home/michele.marschner/project/explanation_shift/logs/metaquantus_%j.out
#SBATCH --error=/sc/home/michele.marschner/project/explanation_shift/logs/metaquantus_%j.err

set -euxo pipefail

source /sc/home/michele.marschner/conda3/etc/profile.d/conda.sh
conda activate shift

cd /sc/home/michele.marschner/project/explanation_shift
mkdir -p logs

echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true
python --version

python -m src.main \
  --mode train \
  --job-mode single \
  --job-type metaquantus \
  --job-index 4 \
  --stage metaquantus \
  --config src/configs/experiments_config.py