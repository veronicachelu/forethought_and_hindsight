#!/bin/bash
#SBATCH --job-name=hypertune__bw__loop
#SBATCH --output=./outputs/hypertune__bw__loop__output.txt
#SBATCH --error=./errors/hypertune__bw__loop__error.txt
#SBATCH --ntasks=1
#SBATCH --time=10000
#SBATCH --mem=100Gb

source ~/.bashrc
cd ~/jax_models
python hyper_tune.py --agent=bw --env=loop