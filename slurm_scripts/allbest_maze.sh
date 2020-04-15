#!/bin/bash
#SBATCH --job-name=allbest__maze
#SBATCH --output=./outputs/allbest__maze__output.txt
#SBATCH --error=./errors/allbest__maze__error.txt
#SBATCH --ntasks=1
#SBATCH --time=10000
#SBATCH --mem=100Gb

source ~/.bashrc
cd ~/jax_models
python run_all_best.py --env=maze