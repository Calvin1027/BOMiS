#!/usr/bin/env bash
#SBATCH --job-name=dgp_pytest
#SBATCH --partition=brl
#SBATCH --ntasks=1
#SBATCH --qos=normal
#SBATCH --exclude=brl1,brl2
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --chdir=/mnt/data1/calvinkim1027/BOMIS_new_수만이형_runtime
#SBATCH --output=logs/%x_%j.log
# C20, D20.1: 계획 T1.
set -euo pipefail
/mnt/data1/calvinkim1027/miniconda3/envs/bomis/bin/python -I -m pytest -v
