#!/bin/bash
#SBATCH --job-name=allbest__boyan
#SBATCH --output=./outputs/allbest__boyan__output.txt
#SBATCH --error=./errors/allbest__boyan__error.txt
#SBATCH --ntasks=1
#SBATCH --time=10000
#SBATCH --mem=100Gb

source ~/.bashrc
cd ~/jax_models
python run_all_best.py --env=boyan