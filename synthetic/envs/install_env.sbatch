#!/usr/bin/env bash
#SBATCH --job-name=env_install
#SBATCH --partition=brl
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=env_install_%j.log
# 계산 노드에서 미리 받은 wheel로 env에 고정 버전을 오프라인 설치하고, 설치 결과가 requirements와 같은지 확인한다.
# 먼저 로그인 노드에서: conda create -y -p <ENV_PREFIX> python=<버전> pip, bash envs/fetch_wheels.sh ...
# 사용: sbatch --export=ALL,ENV_PREFIX=<env 경로>,WHEELS=<wheel 폴더>,REQ=<requirements.txt 절대 경로> envs/install_env.sbatch
set -euo pipefail
: "${ENV_PREFIX:?ENV_PREFIX 필요}" "${WHEELS:?WHEELS 필요}" "${REQ:?REQ 필요}"
export PYTHONNOUSERSITE=1 PIP_NO_CACHE_DIR=1
PY="$ENV_PREFIX/bin/python"
echo "node=$(hostname) env=$ENV_PREFIX req=$REQ"
"$PY" --version

# --no-deps: requirements에 적힌 패키지와 버전만 설치한다(의존 패키지도 모두 목록에 있다).
"$PY" -m pip install --no-index --no-deps --no-build-isolation --find-links "$WHEELS" -r "$REQ"

echo "== pip check"
"$PY" -m pip check || echo "(pip check 경고: 원래 env와 같은 상태인지 아래 비교로 판단)"

echo "== requirements와 설치 결과 비교"
norm() { tr 'A-Z_' 'a-z-' | sed 's/\.dist-info//' | sort; }
"$PY" -m pip list --format=freeze | norm > "$ENV_PREFIX/installed.txt"
missing=$(grep -vE '^\s*(#|$)' "$REQ" | norm | comm -23 - "$ENV_PREFIX/installed.txt")
if [ -n "$missing" ]; then echo "설치되지 않았거나 버전이 다른 패키지:"; echo "$missing"; exit 1; fi
echo "INSTALL OK: requirements의 패키지 $(grep -cvE '^\s*(#|$)' "$REQ")개가 모두 같은 버전으로 설치됨"
