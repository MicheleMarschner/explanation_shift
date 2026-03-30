#!/bin/bash
#SBATCH --job-name=fix-opencv
#SBATCH --account=sci-demelo-computer-vision
#SBATCH --partition=gpu-batch
#SBATCH --constraint=ARCH:X86
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/sc/home/michele.marschner/project/explanation_shift/logs/fix_opencv_%j.out
#SBATCH --error=/sc/home/michele.marschner/project/explanation_shift/logs/fix_opencv_%j.err

set -euxo pipefail

source /sc/home/michele.marschner/conda3/etc/profile.d/conda.sh
conda activate shift

cd /sc/home/michele.marschner/project/explanation_shift
mkdir -p logs

python -m pip uninstall -y opencv-python opencv-python-headless
python -m pip install "opencv-python-headless>=4.10,<4.11"

python -c "import cv2; print(cv2.__version__)"