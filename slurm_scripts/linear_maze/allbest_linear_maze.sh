#!/bin/bash
#SBATCH --job-name=allbest__linear_maze
#SBATCH --output=./outputs/allbest__linear_maze__output.txt
#SBATCH --error=./errors/allbest__linear_maze__error.txt
#SBATCH --ntasks=1
#SBATCH --time=12:00:00

source ~/.bashrc
cd ~/jax_models
python run_all_best.py --env=linear_maze