#!/bin/bash
#SBATCH --job-name=hypertune__MG__maze
#SBATCH --output=./outputs/hypertune__MG__maze_output.txt
#SBATCH --error=./errors/hypertune__MG__maze_error.txt
#SBATCH --ntasks=1
#SBATCH --time=10:00
#SBATCH --mem=100Gb

source ~/.bashrc
cd ~/jax_models
python hyper_tune.py --agent=bw_fw_MG --env=maze