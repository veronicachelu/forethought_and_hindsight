#!/bin/bash
#SBATCH --job-name=allbest__stoch_linear_maze
#SBATCH --output=./outputs/allbest__stoch_linear_maze__output.txt
#SBATCH --error=./errors/allbest__stoch_linear_maze__error.txt
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1

source ~/.bashrc
cd ~/jax_models
python run_all_best.py --env=stoch_linear_maze