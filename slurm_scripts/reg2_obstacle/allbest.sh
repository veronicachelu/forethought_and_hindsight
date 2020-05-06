#!/bin/bash
#SBATCH --job-name=allbest__reg2_obstacle
#SBATCH --output=./outputs/allbest__reg2_obstacle__output.txt
#SBATCH --error=./errors/allbest__reg2_obstacle__error.txt
#SBATCH --ntasks=1
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=2gb

source ~/.bashrc
cd ~/jax_models
python run_all_best.py --env=reg2_obstacle