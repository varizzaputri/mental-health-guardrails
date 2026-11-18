#!/bin/bash
#SBATCH --job-name=eval_qwen4b
#SBATCH --output=eval_qwen4b_%j.out
#SBATCH --error=eval_qwen4b_%j.err
#SBATCH --partition=a100-short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

source ~/.bashrc
conda activate teta_eval

python evaluate_guard.py --model "Qwen/Qwen3Guard-Gen-4B" --iterations 1
