#!/usr/bin/env bash
# C20, D20.1: 계획 Global Constraints. 실행 코드만 동기화한다.
set -euo pipefail
source_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_root=/mnt/data1/calvinkim1027/BOMIS_new_수만이형_runtime
mkdir -p "$runtime_root"/{logs,results,reports}
for directory in dgp scripts tests experiments vendor baselines; do
    mkdir -p "$runtime_root/$directory"
    rsync -a --delete --exclude "__pycache__" "$source_root/$directory/" "$runtime_root/$directory/"
done
rsync -a "$source_root/pytest.ini" "$runtime_root/pytest.ini"
