#!/bin/bash
# Run the layer-wise fine-tuning editing experiment for all 27 models
# (9 weight decays x 3 seeds), one model per array task.
#
# Adjust the partition (and account/constraint if required) for your cluster,
# activate your Python environment, then submit from 03_editing/ with:
#   sbatch run_sweep.sh
# After all tasks finish, build the summary CSV with:
#   python summarize_results.py
#
#SBATCH --job-name=editing
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:1
#SBATCH --mem=125000
#SBATCH --time=24:00:00
#SBATCH --array=0-26
#SBATCH --output=logs/slurm-%A_%a.out

set -euo pipefail
export PYTHONUNBUFFERED=1

WD_VALUES=(0.0 0.1 0.5 1.0 2.0 3.0 4.0 5.0 6.0)
WD=${WD_VALUES[$((SLURM_ARRAY_TASK_ID % 9))]}
SEED=$((SLURM_ARRAY_TASK_ID / 9))

echo "Task ${SLURM_ARRAY_TASK_ID}: weight_decay=${WD}, seed=${SEED}"

python run_all_baselines.py --wd "${WD}" --seed "${SEED}"
