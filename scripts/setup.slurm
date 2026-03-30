#!/bin/bash
#SBATCH -J anlp_setup
#SBATCH -p cpu-batch
#SBATCH --account=sci-demelo-computer-vision
#SBATCH -t 01:00:00
#SBATCH -o logs/setup_%j.out
#SBATCH -e logs/setup_%j.err

source /sc/home/michele.marschner/conda3/etc/profile.d/conda.sh
#conda create -n shift python=3.11
conda activate shift
cd ~/project/explanation_shift
python -m pip install -e .