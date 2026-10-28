#!/usr/bin/env bash
#SBATCH --job-name=easy_dl
#SBATCH --partition=brl
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --exclude=brl2
#SBATCH --chdir=/mnt/data1/calvinkim1027/BOMIS_new_수만이형_runtime
#SBATCH --output=logs/%x_%A_%a.log
# 작업 하나 = 한 DL 모델의 모집단 3칸을 GPU 1장에서 동시에(slot 0–2). 칸 배치는 scripts/run_dl.py.
set -euo pipefail
deps="${SLURM_TMPDIR:-/tmp}/bl_deps_${SLURM_JOB_ID}"
trap 'rm -rf "$deps"' EXIT
# slot 셋이 같은 폴더에 동시에 풀지 않도록 extras는 여기서 한 번 푼다
mkdir -p "$deps" && tar -xzf /mnt/data1/calvinkim1027/categorical_baselines_runtime_20260929/extras.tar.gz -C "$deps" && touch "$deps/.unpacked"
export BL_DEPS="$deps" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
echo "job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID node=$SLURMD_NODENAME"; nvidia-smi --query-gpu=name --format=csv,noheader
pids=()
for slot in 0 1 2; do
    /mnt/data1/calvinkim1027/miniconda3/envs/bomis/bin/python -I scripts/run_dl.py --slot "$slot" ${EXTRA_ARGS:-} \
        > "logs/easy_dl_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_s${slot}.log" 2>&1 &
    pids+=($!)
done
rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
exit $rc
