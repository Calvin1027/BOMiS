#!/usr/bin/env bash
#SBATCH --job-name=abl_time
#SBATCH --partition=brl
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=01:00:00
#SBATCH --chdir=/mnt/data1/calvinkim1027/BOMIS_new_수만이형_runtime
#SBATCH --output=logs/%x_%j.log
# Chao1 상한 ablation 칸 1개 시간 측정 (fav, pop 0, n1=400, seed 0: 기존 BOMiS 650초, fav에서 가장 느린 칸)
set -euo pipefail
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONNOUSERSITE=1
echo "job=$SLURM_JOB_ID node=$SLURMD_NODENAME"; sha256sum scripts/run_ablation.py dgp/*.py | sha256sum
/usr/bin/env /mnt/data1/calvinkim1027/miniconda3/envs/bomis/bin/python -I scripts/run_ablation.py --env frequent__fav --pop 0 --n1 400 --seed 0 --out results/ablation_cap_time 2>&1
