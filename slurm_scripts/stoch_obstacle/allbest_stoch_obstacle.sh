#!/bin/bash
#SBATCH --job-name=allbest__stoch_obstacle
#SBATCH --output=./outputs/allbest__stoch_obstacle__output.txt
#SBATCH --error=./errors/allbest__stoch_obstacle__error.txt
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1

source ~/.bashrc
cd ~/jax_models
python run_all_best.py --env=stoch_obstacle