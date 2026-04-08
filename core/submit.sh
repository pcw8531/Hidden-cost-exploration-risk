#!/bin/bash
#SBATCH --job-name=hidden_cost
#SBATCH --array=0-8
#SBATCH --time=48:00:00
#SBATCH --mem=8G
#SBATCH --output=log_%A_%a.out

# Each array index maps to pr = [0.1, 0.2, ..., 0.9]
python3 model_hpc.py $SLURM_ARRAY_TASK_ID
