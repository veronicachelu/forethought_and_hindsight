#!/bin/bash
#SBATCH --job-name=hypertune__all__boyan
#SBATCH --output=./outputs/hypertune__all__boyan__output.txt
#SBATCH --error=./errors/hypertune__all__boyan__error.txt
#SBATCH --ntasks=1
#SBATCH --time=12:00:00

source ~/.bashrc
cd ~/jax_models
python hyper_tune.py --agent=bw --env=boyan
python hyper_tune.py --agent=fw --env=boyan
python hyper_tune.py --agent=fw_rnd --env=boyan