#!/bin/bash
#SBATCH --job-name=hypertune__bw_vaware__linear_maze
#SBATCH --output=./outputs/hypertune__bw_vaware__linear_maze__output.txt
#SBATCH --error=./errors/hypertune__bw_vaware__linear_maze__error.txt
#SBATCH --ntasks=1
#SBATCH --time=12:00:00

source ~/.bashrc
cd ~/jax_models
python hyper_tune.py --agent=bw_vaware --env=linear_maze