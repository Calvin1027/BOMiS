#!/usr/bin/env bash
# C20, D20.1: 실행 코드만 런타임 폴더($BOMIS_RUNTIME, config/paths.sh)로 동기화한다.
set -euo pipefail
source_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$source_root/config/paths.sh"
runtime_root="$BOMIS_RUNTIME"
mkdir -p "$runtime_root"/{logs,results,reports}
for directory in dgp scripts tests experiments vendor baselines config third_party; do
    mkdir -p "$runtime_root/$directory"
    rsync -a --delete --exclude "__pycache__" "$source_root/$directory/" "$runtime_root/$directory/"
done
rsync -a "$source_root/pytest.ini" "$runtime_root/pytest.ini"
