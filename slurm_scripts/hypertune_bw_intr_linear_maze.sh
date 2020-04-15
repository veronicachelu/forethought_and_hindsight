#!/bin/bash
#SBATCH --job-name=hypertune__bw_intr__maze
#SBATCH --output=./outputs/hypertune__bw_intr_maze__output.txt
#SBATCH --error=./errors/hypertune__bw_intr_maze__error.txt
#SBATCH --ntasks=1
#SBATCH --time=10000
#SBATCH --mem=100Gb

source ~/.bashrc
cd ~/jax_models
python hyper_tune.py --agent=bw_intr --env=linear_maze