#!/bin/bash
#SBATCH --job-name=hypertune__all__repeat
#SBATCH --output=./outputs/hypertune__all__repeat__output.txt
#SBATCH --error=./errors/hypertune__all__repeat__error.txt
#SBATCH --ntasks=1
#SBATCH --time=10000
#SBATCH --mem=100Gb

source ~/.bashrc
cd ~/jax_models
python hyper_tune.py --agent=bw --env=repeat
python hyper_tune.py --agent=fw --env=repeat
python hyper_tune.py --agent=bw_fw --env=repeat
python hyper_tune.py --agent=fw_rnd --env=repeat
python hyper_tune.py --agent=fw_pri --env=repeat
python hyper_tune.py --agent=bw_fw_PWMA --env=repeat
python hyper_tune.py --agent=bw_fw_MG --env=repeat