#!/usr/bin/env bash
#SBATCH --job-name=abl_cap_mr
#SBATCH --partition=brl
#SBATCH --qos=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=2G
#SBATCH --time=00:40:00
#SBATCH --array=0-359%10
#SBATCH --chdir=/mnt/data1/calvinkim1027/BOMIS_new_수만이형_runtime
#SBATCH --output=logs/%x_%A_%a.log
# Chao1 상한 ablation mid·rare 360칸 (6환경 × 모집단 3 × n1 2 × seed 10). 줄 번호 = 작업 번호 + 1.
# 0번 = 중간 길이 칸, 나머지는 기존 BOMiS 시간 긴 순. 자원 근거: frequent ablation 22752 실측 평균 7.8분·최장 12.8분·0.58G, mid·rare BOMiS 최장 칸 1001초.
set -euo pipefail
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONNOUSERSITE=1
read -r ENV POP N1 SEED < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" experiments/ablation_cells_midrare.txt)
echo "job=$SLURM_ARRAY_JOB_ID task=$SLURM_ARRAY_TASK_ID node=$SLURMD_NODENAME cell=$ENV $POP $N1 $SEED"
sha256sum scripts/run_ablation.py dgp/*.py | sha256sum
/mnt/data1/calvinkim1027/miniconda3/envs/bomis/bin/python -I scripts/run_ablation.py --env "$ENV" --pop "$POP" --n1 "$N1" --seed "$SEED" --out results/ablation_cap
