#!/bin/bash
# Weight-decay x seed sweep from the paper (arXiv:2509.21993, Appendix A):
# weight decay in {0, 0.1, 0.5, 1, 2, 3, 4, 5, 6} x seeds {0, 1, 2} = 27 runs,
# each on 4 GPUs with a global batch size of 64.
#
# Adjust the partition (and account/constraint if required) for your cluster,
# activate your Python environment, then submit from 01_train/ with:  sbatch train_sweep.sh
#
#SBATCH --job-name=family-neox
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --array=0-26
#SBATCH --output=logs/slurm-%A_%a.out

set -euo pipefail
export PYTHONUNBUFFERED=1

WD_VALUES=(0.0 0.1 0.5 1.0 2.0 3.0 4.0 5.0 6.0)
WD=${WD_VALUES[$((SLURM_ARRAY_TASK_ID % 9))]}
SEED=$((SLURM_ARRAY_TASK_ID / 9))

echo "Task ${SLURM_ARRAY_TASK_ID}: weight_decay=${WD}, seed=${SEED}"

torchrun --standalone --nproc_per_node=4 train.py \
    --weight_decay "${WD}" \
    --seed "${SEED}" \
    --out_dir "runs/wd${WD}_seed${SEED}"
