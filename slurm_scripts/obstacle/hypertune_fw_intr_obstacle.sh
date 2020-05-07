#!/bin/bash
#SBATCH --job-name=hypertune__fw_PAML__obstacle
#SBATCH --output=./outputs/hypertune__fw_PAML__obstacle__output.txt
#SBATCH --error=./errors/hypertune__fw_PAML__obstacle__error.txt
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1

source ~/.bashrc
cd ~/jax_models
python hyper_tune.py --agent=fw_PAML --env=obstacle