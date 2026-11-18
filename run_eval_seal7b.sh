#!/bin/bash
#SBATCH --job-name=eval_seal7b
#SBATCH --output=eval_seal7b_%j.out
#SBATCH --error=eval_seal7b_%j.err
#SBATCH --partition=a100-short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

source ~/.bashrc
conda activate teta_eval

python -u evaluate_guard.py --model "MickyMike/SEALGuard-7B" --iterations 1
