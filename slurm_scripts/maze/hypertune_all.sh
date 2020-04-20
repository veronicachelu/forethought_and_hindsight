#!/bin/bash
#SBATCH --job-name=hypertune__all__maze
#SBATCH --output=./outputs/hypertune__all__maze__output.txt
#SBATCH --error=./errors/hypertune__all__maze__error.txt
#SBATCH --ntasks=1
#SBATCH --time=12:00:00

source ~/.bashrc
cd ~/jax_models
python hyper_tune.py --agent=bw --env=maze
python hyper_tune.py --agent=fw --env=maze
python hyper_tune.py --agent=bw_fw --env=maze
python hyper_tune.py --agent=fw_rnd --env=maze
python hyper_tune.py --agent=fw_pri --env=maze
python hyper_tune.py --agent=bw_fw_PWMA --env=maze
python hyper_tune.py --agent=bw_fw_MG --env=maze