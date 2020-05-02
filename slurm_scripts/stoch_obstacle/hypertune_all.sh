#!/bin/bash
#SBATCH --job-name=hypertune__all__stoch_obstacle
#SBATCH --output=./outputs/hypertune__all__stoch_obstacle__output.txt
#SBATCH --error=./errors/hypertune__all__stoch_obstacle__error.txt
#SBATCH --ntasks=1
#SBATCH --time=12:00:00

source ~/.bashrc
cd ~/jax_models
python hyper_tune.py --agent=bw --env=stoch_obstacle
python hyper_tune.py --agent=fw --env=stoch_obstacle
python hyper_tune.py --agent=bw_fw --env=stoch_obstacle
python hyper_tune.py --agent=fw_rnd --env=stoch_obstacle
python hyper_tune.py --agent=fw_pri --env=stoch_obstacle
python hyper_tune.py --agent=bw_fw_PWMA --env=stoch_obstacle
python hyper_tune.py --agent=bw_fw_MG --env=stoch_obstacle