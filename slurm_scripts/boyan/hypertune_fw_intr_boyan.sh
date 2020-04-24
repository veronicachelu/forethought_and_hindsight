#!/bin/bash
#SBATCH --job-name=hypertune__fw_intr__boyan
#SBATCH --output=./outputs/hypertune__bw_intr__boyan__output.txt
#SBATCH --error=./errors/hypertune__fw_intr__boyan__error.txt
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1

source ~/.bashrc
cd ~/jax_models
python hyper_tune.py --agent=fw_intr --env=boyan