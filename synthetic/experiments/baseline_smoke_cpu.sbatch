#!/usr/bin/env bash
#SBATCH --job-name=bl_smoke_cpu
#SBATCH --partition=brl
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR" && source config/paths.sh  # 런타임 폴더($BOMIS_RUNTIME)에서 제출한다
echo "job=$SLURM_JOB_ID node=$SLURMD_NODENAME"; sha256sum scripts/baseline_smoke.py baselines/external.py
"$BOMIS_PY" -I scripts/baseline_smoke.py --models smoten smote_enc
