#!/usr/bin/env bash
#SBATCH --job-name=bl_smoke_cpu
#SBATCH --partition=brl
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --chdir=/mnt/data1/calvinkim1027/BOMIS_new_수만이형_runtime
#SBATCH --output=logs/%x_%j.log
set -euo pipefail
echo "job=$SLURM_JOB_ID node=$SLURMD_NODENAME"; sha256sum scripts/baseline_smoke.py baselines/external.py
/mnt/data1/calvinkim1027/miniconda3/envs/bomis/bin/python -I scripts/baseline_smoke.py --models smoten smote_enc
