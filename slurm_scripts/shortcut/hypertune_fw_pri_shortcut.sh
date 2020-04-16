#!/bin/bash
#SBATCH --job-name=hypertune__fw_pri__shortcut
#SBATCH --output=./outputs/hypertune__fw_pri__shortcut__output.txt
#SBATCH --error=./errors/hypertune__fw_pri__shortcut__error.txt
#SBATCH --ntasks=1
#SBATCH --time=10000
#SBATCH --mem=100Gb

source ~/.bashrc
cd ~/jax_models
python hyper_tune.py --agent=fw_pri --env=shortcut