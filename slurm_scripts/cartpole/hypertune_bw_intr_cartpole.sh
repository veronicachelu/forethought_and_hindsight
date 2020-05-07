#!/bin/bash
#SBATCH --job-name=hypertune__bw_PAML__cartpole
#SBATCH --output=./outputs/hypertune__bw_PAML__cartpole__output.txt
#SBATCH --error=./errors/hypertune__bw_PAML__cartpole__error.txt
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1

source ~/.bashrc
conda activate new
module load cuda/10.0
module load cuda/10.0/cudnn/7.6

cd ~/jax_models
python hyper_tune.py --agent=bw_PAML --env=cartpole