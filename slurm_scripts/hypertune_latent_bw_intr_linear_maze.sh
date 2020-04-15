#!/bin/bash
#SBATCH --job-name=hypertune__latent_bw_intr__maze
#SBATCH --output=./outputs/hypertune__latent_bw_intr__maze_output.txt
#SBATCH --error=./errors/hypertune__latent_bw_intr__maze_error.txt
#SBATCH --ntasks=1
#SBATCH --time=10000
#SBATCH --mem=100Gb

source ~/.bashrc
cd ~/jax_models
python hyper_tune.py --agent=latent_bw_intr --env=linear_maze