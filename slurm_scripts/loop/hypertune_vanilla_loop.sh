#!/bin/bash
#SBATCH --job-name=hypertune__vanilla__loop
#SBATCH --output=./outputs/hypertune__vanilla__loop__output.txt
#SBATCH --error=./errors/hypertune__vanilla__loop__error.txt
#SBATCH --ntasks=1
#SBATCH --time=10000
#SBATCH --mem=100Gb

source ~/.bashrc
cd ~/jax_models
python hyper_tune.py --agent=vanilla --env=loop