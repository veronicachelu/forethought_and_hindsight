#!/bin/bash
#SBATCH --job-name=allbest__reg1_obstacle_mb_bw_iter
#SBATCH --output=./outputs/allbest__reg1_obstacle_mb_bw_iter__output.txt
#SBATCH --error=./errors/allbest__reg1_obstacle_mb_bw_iter__error.txt
#SBATCH --ntasks=1
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=2gb

source ~/.bashrc
cd ~/jax_models
python run.py --env=reg1_obstacle --agent=mb_bw_iter --lr=0.001 --lr_p=0.001 --lr_m=0.005 --planning_depth=1