#!/bin/bash
#SBATCH --job-name=allbest__repeat
#SBATCH --output=./outputs/allbest__repeat__output.txt
#SBATCH --error=./errors/allbest__repeat__error.txt
#SBATCH --ntasks=1
#SBATCH --time=10000
#SBATCH --mem=100Gb

source ~/.bashrc
cd ~/jax_models
python run_all_best.py --env=repeat