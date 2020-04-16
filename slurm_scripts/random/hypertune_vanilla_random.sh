#!/bin/bash
#SBATCH --job-name=hypertune__vanilla__random
#SBATCH --output=./outputs/hypertune__vanilla__random__output.txt
#SBATCH --error=./errors/hypertune__vanilla__random__error.txt
#SBATCH --ntasks=1
#SBATCH --time=10000
#SBATCH --mem=100Gb

source ~/.bashrc
cd ~/jax_models
python hyper_tune.py --agent=vanilla --env=random