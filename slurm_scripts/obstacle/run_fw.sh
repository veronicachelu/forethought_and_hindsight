#!/bin/bash
#SBATCH --job-name=allbest__obstacle
#SBATCH --output=./outputs/allbest__obstacle__output.txt
#SBATCH --error=./errors/allbest__obstacle__error.txt
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1

source ~/.bashrc
cd ~/jax_models
python run.py --agent=fw --env=obstacle --lr=0.01 --lr_p=0.01 --lr_m=0.01 --planning_depth=1
