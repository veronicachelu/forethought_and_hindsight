#!/bin/bash
#SBATCH --job-name=hypertune
#SBATCH --output=hypertune_output.txt
#SBATCH --error=hypertune_error.txt
#SBATCH --ntasks=1
#SBATCH --time=10:00
#SBATCH --mem=100Gb

### Print total arguments and their values

echo "Total Arguments:" $#
echo "All Arguments values:" $@

### Command arguments can be accessed as

echo "agent->"  $1
echo "env->" $2

source ~/.bashrc
cd ~/jax_models
python hyper_tune.py --agent=$1 --env=$2