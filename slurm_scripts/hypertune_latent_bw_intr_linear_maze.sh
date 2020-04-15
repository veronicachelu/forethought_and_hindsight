#!/bin/bash
#SBATCH --job-name=hypertune_MG_maze
#SBATCH --output=./outputs/hypertune_MG_maze_output.txt
#SBATCH --error=./errors/hypertune_MG_maze_error.txt
#SBATCH --ntasks=1
#SBATCH --time=10:00
#SBATCH --mem=100Gb

source ~/.bashrc
cd ~/jax_models
python hyper_tune.py --agent=latent_bw_intr --env=linear_maze