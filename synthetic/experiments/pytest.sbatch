#!/usr/bin/env bash
#SBATCH --job-name=dgp_pytest
#SBATCH --partition=brl
#SBATCH --ntasks=1
#SBATCH --qos=normal
#SBATCH --exclude=brl1,brl2
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.log
# C20, D20.1: 계획 T1.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR" && source config/paths.sh  # 런타임 폴더($BOMIS_RUNTIME)에서 제출한다
"$BOMIS_PY" -I -m pytest -v
