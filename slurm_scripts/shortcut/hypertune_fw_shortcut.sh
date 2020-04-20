#!/bin/bash
#SBATCH --job-name=hypertune__fw__shortcut
#SBATCH --output=./outputs/hypertune__fw__shortcut__output.txt
#SBATCH --error=./errors/hypertune__fw__shortcut__error.txt
#SBATCH --ntasks=1
#SBATCH --time=12:00:00

source ~/.bashrc
cd ~/jax_models
python hyper_tune.py --agent=fw --env=shortcut