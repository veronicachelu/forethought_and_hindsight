#!/bin/bash
#SBATCH --job-name=hypertune__bw_fw_PWMA__repeat
#SBATCH --output=./outputs/hypertune__bw_fw_PWMA__repeat__output.txt
#SBATCH --error=./errors/hypertune__bw_fw_PWMA__repeat__error.txt
#SBATCH --ntasks=1
#SBATCH --time=10000
#SBATCH --mem=100Gb

source ~/.bashrc
cd ~/jax_models
python hyper_tune.py --agent=bw_fw_PWMA --env=repeat