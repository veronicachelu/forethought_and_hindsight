#!/bin/bash
#SBATCH --job-name=allbest__boyan
#SBATCH --output=./outputs/allbest__boyan__output.txt
#SBATCH --error=./errors/allbest__boyan__error.txt
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1

source ~/.bashrc
cd ~/jax_models
python run_all_best.py --env=boyan