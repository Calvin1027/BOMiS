#!/usr/bin/env bash
# C27, D27.2: baseline 프로젝트 코드(수정 없음)를 실행 사본으로 복사한다.
set -euo pipefail
src="/data1/calvinkim1027/과거 BOMIS/TDG_baseline_categorical"
dst=/mnt/data1/calvinkim1027/BOMIS_new_수만이형_runtime/baselines_src
mkdir -p "$dst"
for directory in src configs third_party; do
    rsync -a --exclude "__pycache__" --exclude ".git" "$src/$directory/" "$dst/$directory/"
done
