#!/bin/bash
#SBATCH --job-name=hypertune__fw_pri__random
#SBATCH --output=./outputs/hypertune__fw_pri__random__output.txt
#SBATCH --error=./errors/hypertune__fw_pri__random__error.txt
#SBATCH --ntasks=1
#SBATCH --time=10000
#SBATCH --mem=100Gb

source ~/.bashrc
cd ~/jax_models
python hyper_tune.py --agent=fw_pri --env=random