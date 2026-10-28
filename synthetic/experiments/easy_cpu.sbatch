#!/usr/bin/env bash
#SBATCH --job-name=easy_cpu
#SBATCH --partition=brl
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=01:00:00
#SBATCH --chdir=/mnt/data1/calvinkim1027/BOMIS_new_수만이형_runtime
#SBATCH --output=logs/%x_%A_%a.log
set -euo pipefail
trap 'rm -rf "${SLURM_TMPDIR:-/tmp}/bl_deps_${SLURM_JOB_ID}"' EXIT
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
echo "job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID node=$SLURMD_NODENAME"; sha256sum scripts/run_easy.py baselines/external.py dgp/*.py | sha256sum
/mnt/data1/calvinkim1027/miniconda3/envs/bomis/bin/python -I scripts/run_easy.py --group cpu ${EXTRA_ARGS:-}
