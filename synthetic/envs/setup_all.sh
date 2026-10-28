#!/usr/bin/env bash
# 실험 환경 전체 설치. 로그인 노드에서 synthetic/ 폴더 안에서 실행한다:  bash envs/setup_all.sh
#   1) (로그인 노드) conda env 두 개 만들기, wheel 받기 — 인터넷이 필요한 단계
#   2) (계산 노드, sbatch 3개를 차례로 제출) 오프라인 설치와 확인
# 경로는 config/paths.sh에서 읽는다. 이미 있는 env와 받은 wheel은 건너뛰므로 중간에 끊겨도 다시 실행하면 된다.
set -euo pipefail
cd "$(dirname "$0")/.."
SYN=$(pwd)
source config/paths.sh
CONDA=${CONDA:-/mnt/data1/$USER/miniconda3/bin/conda}
WHEELS=${BOMIS_WHEELS:-/mnt/data1/$USER/bomis_wheels}
BOMIS_ENV=$(dirname "$(dirname "$BOMIS_PY")")
BL_ENV=$(dirname "$(dirname "$BL_PYTHON")")
BL_LAYERS=$(dirname "$BL_GEN_SITE")
SBATCH_OPTS=${SBATCH_OPTS:-}   # 예: "--qos=normal --exclude=brl1,brl2"
LOGS=${BOMIS_SETUP_LOGS:-/mnt/data1/$USER/bomis_setup_logs}
[ -x "$CONDA" ] || { echo "conda가 없음: $CONDA (envs/README.md 0단계)"; exit 1; }
command -v jq >/dev/null && command -v curl >/dev/null || { echo "jq, curl이 필요함"; exit 1; }
mkdir -p "$WHEELS" "$LOGS"

echo "== 1. conda env"
[ -x "$BOMIS_ENV/bin/python" ] || "$CONDA" create -q -y -p "$BOMIS_ENV" python=3.11.17 pip
[ -x "$BL_ENV/bin/python" ] || "$CONDA" create -q -y -p "$BL_ENV" python=3.10.20 pip

echo "== 2. wheel 받기 (약 6GB)"
bash envs/fetch_wheels.sh envs/bomis/requirements.txt "$WHEELS/bomis" cp311
for L in base generation tabddpm_extras extras; do
  bash envs/fetch_wheels.sh "envs/baselines/$L.txt" "$WHEELS/baselines/$L" cp310
done

echo "== 3. 계산 노드에서 설치 (sbatch)"
# 계산 노드는 /mnt/data1만 보므로 정의 파일을 wheel 폴더 옆으로 복사해서 넘긴다(저장소가 /data1에 있어도 되게).
DEFS="$WHEELS/defs"
rm -rf "$DEFS" && mkdir -p "$DEFS" && cp -r envs/bomis envs/baselines "$DEFS"/
cd "$LOGS"
j1=$(sbatch --parsable $SBATCH_OPTS --export=ALL,ENV_PREFIX="$BOMIS_ENV",WHEELS="$WHEELS/bomis",REQ="$DEFS/bomis/requirements.txt" "$SYN/envs/install_env.sbatch")
j2=$(sbatch --parsable $SBATCH_OPTS --export=ALL,ENV_PREFIX="$BL_ENV",WHEELS="$WHEELS/baselines/base",REQ="$DEFS/baselines/base.txt" "$SYN/envs/install_env.sbatch")
j3=$(sbatch --parsable $SBATCH_OPTS --dependency=afterok:$j2 --export=ALL,BL_PY="$BL_PYTHON",LAYERS="$BL_LAYERS",WHEELS="$WHEELS/baselines",REQ_DIR="$DEFS/baselines" "$SYN/envs/install_layers.sbatch")
echo "제출: bomis env $j1, baseline env $j2, 덧댄 폴더 $j3 (로그 $LOGS)"
echo "끝나면 로그 끝줄 확인: INSTALL OK (두 개), LAYERS OK"
