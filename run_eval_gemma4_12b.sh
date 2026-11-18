#!/bin/bash
#SBATCH --job-name=eval_g4_12b
#SBATCH --output=eval_g4_12b_%j.out
#SBATCH --error=eval_g4_12b_%j.err
#SBATCH --partition=a100-short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

source ~/.bashrc
conda activate teta_eval

python -u evaluate_guard.py --model "unsloth/gemma-4-12b-it" --iterations 1
