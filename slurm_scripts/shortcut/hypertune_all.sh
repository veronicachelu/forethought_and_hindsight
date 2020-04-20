#!/bin/bash
#SBATCH --job-name=hypertune__all__shortcut
#SBATCH --output=./outputs/hypertune__all__shortcut__output.txt
#SBATCH --error=./errors/hypertune__all__shortcut__error.txt
#SBATCH --ntasks=1
#SBATCH --time=12:00:00

source ~/.bashrc
cd ~/jax_models
python hyper_tune.py --agent=bw --env=shortcut
python hyper_tune.py --agent=fw --env=shortcut
python hyper_tune.py --agent=bw_fw --env=shortcut
python hyper_tune.py --agent=fw_rnd --env=shortcut
python hyper_tune.py --agent=fw_pri --env=shortcut
python hyper_tune.py --agent=bw_fw_PWMA --env=shortcut
python hyper_tune.py --agent=bw_fw_MG --env=shortcut