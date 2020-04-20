#!/bin/bash
#SBATCH --job-name=hypertune__fw_rnd__medium_maze
#SBATCH --output=./outputs/hypertune__fw_rnd__medium_maze__output.txt
#SBATCH --error=./errors/hypertune__fw_rnd__medium_maze__error.txt
#SBATCH --ntasks=1
#SBATCH --time=12:00:00

source ~/.bashrc
cd ~/jax_models
python hyper_tune.py --agent=fw_rnd --env=medium_maze