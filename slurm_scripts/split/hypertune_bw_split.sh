#!/bin/bash
#SBATCH --job-name=hypertune__bw__split
#SBATCH --output=./outputs/hypertune__bw__split__output.txt
#SBATCH --error=./errors/hypertune__bw__split__error.txt
#SBATCH --ntasks=1
#SBATCH --time=12:00:00

source ~/.bashrc
cd ~/jax_models
python hyper_tune.py --agent=bw --env=split