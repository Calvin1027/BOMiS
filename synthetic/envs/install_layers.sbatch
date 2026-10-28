#!/usr/bin/env bash
#SBATCH --job-name=bl_layers
#SBATCH --partition=brl
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=bl_layers_%j.log
# baseline env 위에 PYTHONPATH로 얹는 패키지 폴더 3개(generation, tabddpm_extras, extras)를 원래 서버와 같은 구성으로 만든다.
# base env(envs/baselines/base.txt)를 먼저 install_env.sbatch로 설치하고, wheel은 fetch_wheels.sh로 받아 둔다.
# 사용: sbatch --export=ALL,BL_PY=<base env python>,LAYERS=<만들 폴더>,WHEELS=<wheel 상위 폴더>,REQ_DIR=<envs/baselines 절대 경로> envs/install_layers.sbatch
#   WHEELS 아래에 generation/, tabddpm_extras/, extras/ 폴더가 있어야 한다(fetch_wheels.sh 출력).
set -euo pipefail
: "${BL_PY:?}" "${LAYERS:?}" "${WHEELS:?}" "${REQ_DIR:?}"
export PYTHONNOUSERSITE=1 PIP_NO_CACHE_DIR=1
pipt() {  # $1 = 대상 폴더, $2 = wheel 폴더, 나머지 = requirements 파일 또는 패키지
  local target=$1 wheels=$2; shift 2
  "$BL_PY" -m pip install --no-index --no-deps --no-build-isolation --find-links "$wheels" --target "$target" "$@"
}
echo "node=$(hostname) layers=$LAYERS"
mkdir -p "$LAYERS"

for L in tabddpm_extras extras; do
  rm -rf "${LAYERS:?}/$L"
  pipt "$LAYERS/$L" "$WHEELS/$L" -r "$REQ_DIR/$L.txt"
done

# generation: zero 0.9.2와 libzero 0.0.8이 같은 zero/ 폴더를 쓴다. 원래 폴더는 zero를 먼저 깔고 libzero가 겹쳐 덮은 상태라,
# libzero를 따로 받아 파일 단위로 덮어쓴다(pip --target --upgrade는 폴더 전체를 바꿔 버려서 쓰지 않는다).
rm -rf "${LAYERS:?}/generation"
grep -v '^libzero==' "$REQ_DIR/generation.txt" > "$LAYERS/.generation_wo_libzero.txt"
pipt "$LAYERS/generation" "$WHEELS/generation" -r "$LAYERS/.generation_wo_libzero.txt"
tmp=$(mktemp -d)
pipt "$tmp" "$WHEELS/generation" "$(grep '^libzero==' "$REQ_DIR/generation.txt")"
cp -a "$tmp"/. "$LAYERS/generation"/
rm -rf "$tmp" "$LAYERS/.generation_wo_libzero.txt"

echo "== 확인: import"
PYTHONPATH="$LAYERS/extras:$LAYERS/generation" "$BL_PY" -c "import ctgan, imblearn, rdt, timm, ml_collections, zero; print('categorical path ok, zero', zero.__version__)"
PYTHONPATH="$LAYERS/tabddpm_extras:$LAYERS/generation" "$BL_PY" -c "import rtdl, zero, pkg_resources; print('tabddpm path ok, zero', zero.__version__)"
echo "LAYERS OK"
