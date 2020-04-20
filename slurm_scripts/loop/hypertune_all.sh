#!/bin/bash
#SBATCH --job-name=hypertune__all__loop
#SBATCH --output=./outputs/hypertune__all__loop__output.txt
#SBATCH --error=./errors/hypertune__all__loop__error.txt
#SBATCH --ntasks=1
#SBATCH --time=12:00:00

source ~/.bashrc
cd ~/jax_models
python hyper_tune.py --agent=bw --env=loop
python hyper_tune.py --agent=fw --env=loop
python hyper_tune.py --agent=bw_fw --env=loop
python hyper_tune.py --agent=fw_rnd --env=loop
python hyper_tune.py --agent=fw_pri --env=loop
python hyper_tune.py --agent=bw_fw_PWMA --env=loop
python hyper_tune.py --agent=bw_fw_MG --env=loop