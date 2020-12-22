#!/bin/bash
#SBATCH --nodes=1
#SBATCH --requeue
#SBATCH --output=logs/log.out
#SBATCH --job-name=log
#SBATCH --gres=gpu:1
#SBATCH --mem=14000
#SBATCH --time=10000

export PATH="$HOME/miniconda/bin:$PATH"

nvidia-smi
source activate bertie
python logger.py
