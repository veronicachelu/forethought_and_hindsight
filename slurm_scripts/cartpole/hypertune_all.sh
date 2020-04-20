#!/bin/bash
#SBATCH --job-name=hypertune__all__cartpole
#SBATCH --output=./outputs/hypertune__all__cartpole__output.txt
#SBATCH --error=./errors/hypertune__all__cartpole__error.txt
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1

source ~/.bashrc
cd ~/jax_models
python hyper_tune.py --agent=bw --env=cartpole
python hyper_tune.py --agent=fw --env=cartpole
python hyper_tune.py --agent=fw_rnd --env=cartpole