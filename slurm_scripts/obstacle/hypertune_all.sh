#!/bin/bash
#SBATCH --job-name=hypertune__all__obstacle
#SBATCH --output=./outputs/hypertune__all__obstacle__output.txt
#SBATCH --error=./errors/hypertune__all__obstacle__error.txt
#SBATCH --ntasks=1
#SBATCH --time=12:00:00

source ~/.bashrc
cd ~/jax_models
python hyper_tune.py --agent=bw --env=obstacle
python hyper_tune.py --agent=fw --env=obstacle
python hyper_tune.py --agent=bw_fw --env=obstacle
python hyper_tune.py --agent=fw_rnd --env=obstacle
python hyper_tune.py --agent=fw_pri --env=obstacle
python hyper_tune.py --agent=bw_fw_PWMA --env=obstacle
python hyper_tune.py --agent=bw_fw_MG --env=obstacle