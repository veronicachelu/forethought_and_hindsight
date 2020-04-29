#!/bin/bash
#SBATCH --job-name=hypertune__vanilla__random_linear
#SBATCH --output=./outputs/hypertune__vanilla__random_linear__output.txt
#SBATCH --error=./errors/hypertune__vanilla__random_linear__error.txt
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1

source ~/.bashrc
cd ~/jax_models
python hyper_tune.py --agent=vanilla --env=random_linear