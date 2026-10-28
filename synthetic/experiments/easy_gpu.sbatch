#!/usr/bin/env bash
#SBATCH --job-name=easy_gpu
#SBATCH --partition=brl
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=00:45:00
#SBATCH --chdir=/mnt/data1/calvinkim1027/BOMIS_new_수만이형_runtime
#SBATCH --output=logs/%x_%A_%a.log
set -euo pipefail
trap 'rm -rf "${SLURM_TMPDIR:-/tmp}/bl_deps_${SLURM_JOB_ID}"' EXIT
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
echo "job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID node=$SLURMD_NODENAME"; nvidia-smi --query-gpu=name --format=csv,noheader
/mnt/data1/calvinkim1027/miniconda3/envs/bomis/bin/python -I scripts/run_easy.py --group gpu ${EXTRA_ARGS:-}
