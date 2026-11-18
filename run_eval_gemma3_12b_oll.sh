#!/bin/bash
#SBATCH --job-name=eval_g3_12b
#SBATCH --output=eval_g3_12b_%j.out
#SBATCH --error=eval_g3_12b_%j.err
#SBATCH --partition=a100-short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

source ~/.bashrc
conda activate teta_eval

export PATH="$HOME/ollama_local/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/ollama_local/lib/ollama:$LD_LIBRARY_PATH"
export OLLAMA_HOST="127.0.0.1:11457"
export OLLAMA_URL="http://127.0.0.1:11457/api/chat"
export OLLAMA_MODELS="$HOME/.ollama/models"

$HOME/ollama_local/bin/ollama serve > ollama_a100_g3_12b.log 2>&1 &
OLLAMA_PID=$!
sleep 8

python -u evaluate_guard.py --model "gemma3:12b" --iterations 1

kill $OLLAMA_PID 2>/dev/null
