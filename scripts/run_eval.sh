#!/bin/bash
#SBATCH --job-name=eval_guard
#SBATCH --output=results/eval_%j.out
#SBATCH --error=results/eval_%j.err
#SBATCH --partition=a100-short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

source ~/.bashrc
conda activate teta_eval

# Ambil parameter dari input environment atau pakai nilai default
DATA_FILE=${DATA:-"data/dataset_sample.csv"}
MODEL_NAME=${MODEL:-"gemma4:12b"}

echo "Menjalankan evaluasi pada GPU A100:"
echo "Data : $DATA_FILE"
echo "Model: $MODEL_NAME"

# Jalankan evaluasi
python -u evaluate_guard.py --data "$DATA_FILE" --model "$MODEL_NAME" --iterations 1
