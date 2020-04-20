#!/bin/bash
#SBATCH --job-name=hypertune__bw_fw_PWMA__loop
#SBATCH --output=./outputs/hypertune__bw_fw_PWMA__loop__output.txt
#SBATCH --error=./errors/hypertune__bw_fw_PWMA__loop__error.txt
#SBATCH --ntasks=1
#SBATCH --time=12:00:00

source ~/.bashrc
cd ~/jax_models
python hyper_tune.py --agent=bw_fw_PWMA --env=loop