#!/bin/bash
#SBATCH --job-name=eval_qwen8b
#SBATCH --account=student
#SBATCH --partition=a100-long
#SBATCH --qos=a100_1gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=eval_qwen8b_%j.out
#SBATCH --error=eval_qwen8b_%j.err

# Inisialisasi Environment
source /home/paulia/miniconda3/etc/profile.d/conda.sh
conda activate teta_eval
export CUDA_VISIBLE_DEVICES=0

# Pindah ke direktori kerja
cd /home/paulia/05_guardrails

# Jalankan evaluasi model (Ganti nama model di bawah ini sesuai urutan pengujian)
python evaluate_guard.py --model "Qwen/Qwen3Guard-Gen-8B"
