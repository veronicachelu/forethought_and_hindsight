#!/bin/bash
#SBATCH --job-name=hypertune__bw_mult_intr__stoch_obstacle
#SBATCH --output=./outputs/hypertune__bw_mult_intr__stoch_obstacle__output.txt
#SBATCH --error=./errors/hypertune__bw_mult_intr__stoch_obstacle__error.txt
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1

source ~/.bashrc
cd ~/jax_models
python hyper_tune.py --agent=bw_mult_intr --env=stoch_obstacle