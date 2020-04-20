#!/bin/bash
#SBATCH --job-name=hypertune__all__medium_maze
#SBATCH --output=./outputs/hypertune__all__medium_maze__output.txt
#SBATCH --error=./errors/hypertune__all__medium_maze__error.txt
#SBATCH --ntasks=1
#SBATCH --time=12:00:00

source ~/.bashrc
cd ~/jax_models
python hyper_tune.py --agent=bw --env=medium_maze
python hyper_tune.py --agent=fw --env=medium_maze
python hyper_tune.py --agent=bw_fw --env=medium_maze
python hyper_tune.py --agent=fw_rnd --env=medium_maze
python hyper_tune.py --agent=fw_pri --env=medium_maze
python hyper_tune.py --agent=bw_fw_PWMA --env=medium_maze
python hyper_tune.py --agent=bw_fw_MG --env=medium_maze