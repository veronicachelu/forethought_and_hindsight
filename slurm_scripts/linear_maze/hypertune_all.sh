#!/bin/bash
#SBATCH --job-name=hypertune__all__linear_maze
#SBATCH --output=./outputs/hypertune__all__linear_maze__output.txt
#SBATCH --error=./errors/hypertune__all__linear_maze__error.txt
#SBATCH --ntasks=1
#SBATCH --time=10000
#SBATCH --mem=100Gb

source ~/.bashrc
cd ~/jax_models
python hyper_tune.py --agent=bw --env=linear_maze
python hyper_tune.py --agent=fw --env=linear_maze
python hyper_tune.py --agent=fw_rnd --env=linear_maze