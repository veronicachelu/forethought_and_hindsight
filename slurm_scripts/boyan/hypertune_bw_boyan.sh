#!/bin/bash
#SBATCH --job-name=hypertune__bw__boyan
#SBATCH --output=./outputs/hypertune__bw__boyan__output.txt
#SBATCH --error=./errors/hypertune__bw__boyan__error.txt
#SBATCH --ntasks=1
#SBATCH --time=12:00:00

source ~/.bashrc
cd ~/jax_models
python hyper_tune.py --agent=bw --env=boyan