#!/bin/bash
#SBATCH --job-name=allbest__random_linear
#SBATCH --output=./outputs/allbest__random_linear__output.txt
#SBATCH --error=./errors/allbest__random_linear__error.txt
#SBATCH --ntasks=1
#SBATCH --time=12:00:00

source ~/.bashrc
cd ~/jax_models
python run_all_best.py --env=random_linear