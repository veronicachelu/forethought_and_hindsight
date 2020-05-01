#!/bin/bash
#SBATCH --job-name=hypertune__mult_fw__obstacle
#SBATCH --output=./outputs/hypertune__mult_fw__obstacle__output.txt
#SBATCH --error=./errors/hypertune__mult_fw__obstacle__error.txt
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1

source ~/.bashrc
cd ~/jax_models
python hyper_tune.py --agent=mult_fw --env=obstacle