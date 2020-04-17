#!/bin/bash
#SBATCH --job-name=hypertune__all__random
#SBATCH --output=./outputs/hypertune__all__random__output.txt
#SBATCH --error=./errors/hypertune__all__random__error.txt
#SBATCH --ntasks=1
#SBATCH --time=10000
#SBATCH --mem=100Gb

source ~/.bashrc
cd ~/jax_models
python hyper_tune.py --agent=bw --env=random
python hyper_tune.py --agent=fw --env=random
python hyper_tune.py --agent=bw_fw --env=random
python hyper_tune.py --agent=fw_rnd --env=random
python hyper_tune.py --agent=fw_pri --env=random
python hyper_tune.py --agent=bw_fw_PWMA --env=random
python hyper_tune.py --agent=bw_fw_MG --env=random