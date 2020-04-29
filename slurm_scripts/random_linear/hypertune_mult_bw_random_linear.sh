#!/bin/bash
#SBATCH --job-name=hypertune__mult_bw__random_linear
#SBATCH --output=./outputs/hypertune__mult_bw__random_linear__output.txt
#SBATCH --error=./errors/hypertune__mult_bw__random_linear__error.txt
#SBATCH --ntasks=1
#SBATCH --time=12:00:00

source ~/.bashrc
cd ~/jax_models
python hyper_tune.py --agent=mult_bw --env=random_linear