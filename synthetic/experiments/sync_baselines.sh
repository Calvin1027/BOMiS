#!/usr/bin/env bash
# C27, D27.2: baseline 프로젝트 코드(수정 없음, $BL_CATEGORICAL_SRC)를 실행 사본($BOMIS_RUNTIME/baselines_src)으로 복사한다.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/config/paths.sh"
src="$BL_CATEGORICAL_SRC"
dst="$BOMIS_RUNTIME/baselines_src"
mkdir -p "$dst"
for directory in src configs third_party; do
    rsync -a --exclude "__pycache__" --exclude ".git" "$src/$directory/" "$dst/$directory/"
done
