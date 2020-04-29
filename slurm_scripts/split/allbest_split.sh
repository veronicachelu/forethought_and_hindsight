#!/bin/bash
#SBATCH --job-name=allbest__split
#SBATCH --output=./outputs/allbest__split__output.txt
#SBATCH --error=./errors/allbest__split__error.txt
#SBATCH --ntasks=1
#SBATCH --time=12:00:00

source ~/.bashrc
cd ~/jax_models
python run_all_best.py --env=split