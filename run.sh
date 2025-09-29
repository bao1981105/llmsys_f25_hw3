#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 24:00:00
#SBATCH --gpus=v100-32:4
#SBATCH --output=jupyter_output_%j.log      # Standard output file (%j will be replaced with job ID)
#SBATCH --error=jupyter_error_%j.log        # Standard error file (%j will be replaced with job ID)

# load conda
module load anaconda3/2024.10-1

# activate environment
conda activate env_name
nvidia-smi
python3 project/run_machine_translation.py


