#!/bin/bash
#SBATCH --job-name=allbest__shortcut
#SBATCH --output=./outputs/allbest__shortcut__output.txt
#SBATCH --error=./errors/allbest__shortcut__error.txt
#SBATCH --ntasks=1
#SBATCH --time=10000
#SBATCH --mem=100Gb

source ~/.bashrc
cd ~/jax_models
python run_all_best.py --env=shortcut