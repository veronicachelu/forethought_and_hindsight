#!/bin/bash
#SBATCH --job-name=hypertune_MG_maze
#SBATCH --output=hypertune_MG_maze_output.txt
#SBATCH --error=hypertune_MG_maze_error.txt
#SBATCH --ntasks=1
#SBATCH --time=10:00
#SBATCH --mem=100Gb

source ~/.bashrc
cd ~/jax_models
python hyper_tune.py --agent=bw_fw_MG --env=maze