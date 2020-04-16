#!/bin/bash
#SBATCH --job-name=hypertune__latent_vanilla_intr__linear_maze
#SBATCH --output=./outputs/hypertune__latent_vanilla_intr__linear_maze__output.txt
#SBATCH --error=./errors/hypertune__latent_vanilla_intr__linear_maze__error.txt
#SBATCH --ntasks=1
#SBATCH --time=10000
#SBATCH --mem=100Gb

source ~/.bashrc
cd ~/jax_models
python hyper_tune.py --agent=latent_vanilla_intr --env=linear_maze