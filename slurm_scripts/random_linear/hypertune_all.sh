#!/bin/bash
#SBATCH --job-name=hypertune__all__random_linear
#SBATCH --output=./outputs/hypertune__all__random_linear__output.txt
#SBATCH --error=./errors/hypertune__all__random_linear__error.txt
#SBATCH --ntasks=1
#SBATCH --time=12:00:00

source ~/.bashrc
cd ~/jax_models
python hyper_tune.py --agent=bw --env=random_linear
python hyper_tune.py --agent=fw --env=random_linear
python hyper_tune.py --agent=bw_fw --env=random_linear
python hyper_tune.py --agent=fw_rnd --env=random_linear
python hyper_tune.py --agent=fw_pri --env=random_linear
python hyper_tune.py --agent=bw_fw_PWMA --env=random_linear
python hyper_tune.py --agent=bw_fw_MG --env=random_linear