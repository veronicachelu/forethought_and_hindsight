#!/bin/bash
#SBATCH --job-name=hypertune__bw_fw_MG__boyan
#SBATCH --output=./outputs/hypertune__bw_fw_MG__boyan__output.txt
#SBATCH --error=./errors/hypertune__bw_fw_Mg__boyan__error.txt
#SBATCH --ntasks=1
#SBATCH --time=10000
#SBATCH --mem=100Gb

source ~/.bashrc
cd ~/jax_models
python hyper_tune.py --agent=bw_fw_MG --env=boyan