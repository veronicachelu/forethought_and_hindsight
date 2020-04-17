#!/bin/bash
#SBATCH --job-name=hypertune__all__boyan
#SBATCH --output=./outputs/hypertune__all__boyan__output.txt
#SBATCH --error=./errors/hypertune__all__boyan__error.txt
#SBATCH --ntasks=1
#SBATCH --time=10000
#SBATCH --mem=100Gb

source ~/.bashrc
cd ~/jax_models
python hyper_tune.py --agent=bw --env=boyan
python hyper_tune.py --agent=fw --env=boyan
python hyper_tune.py --agent=bw_fw --env=boyan
python hyper_tune.py --agent=fw_rnd --env=boyan
python hyper_tune.py --agent=fw_pri --env=boyan
python hyper_tune.py --agent=bw_fw_PWMA --env=boyan
python hyper_tune.py --agent=bw_fw_MG --env=boyan