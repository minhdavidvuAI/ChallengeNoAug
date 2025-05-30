#!/bin/bash
#SBATCH --job-name=esc50_run
#SBATCH --partition=spotvm-t4
#SBATCH --time=7:00:00
#SBATCH --ntasks=1
#SBATCH --output=logs/esc50_%j.out  # %j = job ID

python train_crossval.py
