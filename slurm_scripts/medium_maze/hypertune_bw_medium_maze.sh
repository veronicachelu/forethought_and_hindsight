#!/bin/bash
#SBATCH --job-name=hypertune__bw__medium_maze
#SBATCH --output=./outputs/hypertune__bw__medium_maze__output.txt
#SBATCH --error=./errors/hypertune__bw__medium_maze__error.txt
#SBATCH --ntasks=1
#SBATCH --time=10000
#SBATCH --mem=100Gb

source ~/.bashrc
cd ~/jax_models
python hyper_tune.py --agent=bw --env=medium_maze