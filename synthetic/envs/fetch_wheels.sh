#!/usr/bin/env bash
# 로그인 노드에서 실행: requirements의 고정 버전 wheel을 PyPI(와 PyTorch 저장소)에서 curl로 받는다.
# 계산 노드는 인터넷이 안 되므로, install.sbatch가 이 폴더를 --find-links로 써서 오프라인 설치한다.
# Python을 실행하지 않는다(로그인 노드에서는 python 실행 금지). curl과 jq만 쓴다.
#
# 사용: bash envs/fetch_wheels.sh <requirements.txt> <wheel 폴더> <cp 태그, 예: cp311>
# 결과: <wheel 폴더>/plan.txt (패키지별로 고른 파일), 받은 wheel들. 이미 받은 파일은 건너뛴다.
set -euo pipefail
REQ=$1 OUT=$2 CP=$3
GLIBC_MAX=231   # 서버 glibc 2.31. manylinux_2_XX에서 XX가 31 이하인 것만 쓴다.
mkdir -p "$OUT"
PLAN="$OUT/plan.txt"
: > "$PLAN"

ok_tag() {  # 파일 이름이 이 서버에서 설치 가능한 wheel인지
  local f=$1
  local plat
  if [[ $f =~ -(${CP}-${CP}|py3-none|py2\.py3-none)-([^/]+)\.whl$ ]]; then
    plat=${BASH_REMATCH[2]}
  elif [[ $f =~ -cp3([0-9]+)-abi3-([^/]+)\.whl$ ]] && (( BASH_REMATCH[1] <= ${CP#cp3} )); then
    plat=${BASH_REMATCH[2]}  # 안정 ABI(abi3) wheel: 표시된 버전 이상의 모든 CPython에서 쓴다
  else
    return 1
  fi
  [[ $plat == any ]] && return 0
  [[ $plat == *x86_64* ]] || return 1
  [[ $plat == linux_x86_64 ]] && return 0
  local best=0 t
  for t in $(grep -oE 'manylinux_2_[0-9]+' <<<"$plat" | sed 's/manylinux_2_//'); do (( 200 + t <= GLIBC_MAX )) && best=1; done
  [[ $plat =~ manylinux(1|2010|2014) ]] && best=1
  (( best == 1 ))
}

wheel_rank() {  # pip의 선호 순서: glibc 요구 버전이 높은 빌드 > 낮은 빌드 > linux_x86_64 > any
  local f=$1 r=0 t
  [[ $f == *-any.whl ]] && { echo 0; return; }
  r=1
  [[ $f == *manylinux1_* ]] && r=5
  [[ $f == *manylinux2010_* ]] && r=12
  [[ $f == *manylinux2014_* ]] && r=17
  for t in $(grep -oE 'manylinux_2_[0-9]+' <<<"$f" | sed 's/manylinux_2_//'); do
    (( 200 + t <= GLIBC_MAX && t > r )) && r=$t
  done
  echo "$r"
}

grep -vE '^\s*(#|$)' "$REQ" | while read -r spec; do
  name=${spec%%==*} ver=${spec##*==}
  if [[ $ver == *+* ]]; then  # PyTorch 저장소의 로컬 버전(예: 2.5.1+cu121)
    cu=${ver##*+}
    url="https://download.pytorch.org/whl/${cu}/${name//-/_}-${ver/+/%2B}-${CP}-${CP}-linux_x86_64.whl"
    echo "$url $name $ver pytorch" >> "$PLAN"
    continue
  fi
  json=$(curl -sf "https://pypi.org/pypi/$name/$ver/json") || { echo "NONE $name $ver (PyPI에 없음)" >> "$PLAN"; continue; }
  pick="" best=-1
  while read -r url; do  # 설치 가능한 wheel 중 pip처럼 가장 최신 glibc 빌드를 고른다
    [[ -z $url ]] && continue
    f=$(basename "$url")
    ok_tag "$f" || continue
    s=$(wheel_rank "$f")
    (( s > best )) && { best=$s; pick=$url; }
  done < <(jq -r '.urls[] | select(.packagetype=="bdist_wheel") | .url' <<<"$json")
  if [[ -z $pick ]]; then  # 맞는 wheel이 없으면 소스 배포판
    pick=$(jq -r '[.urls[] | select(.packagetype=="sdist")][0].url // empty' <<<"$json")
    [[ -n $pick ]] && echo "$pick $name $ver sdist" >> "$PLAN" || echo "NONE $name $ver" >> "$PLAN"
  else
    echo "$pick $name $ver wheel" >> "$PLAN"
  fi
done

if grep -q '^NONE' "$PLAN"; then grep '^NONE' "$PLAN"; echo "받을 파일을 찾지 못한 패키지가 있음"; exit 1; fi
cd "$OUT"
awk '{print $1}' plan.txt | while read -r url; do
  f=$(basename "${url//%2B/+}")
  [ -s "$f" ] || curl -sfL --retry 3 -o "$f" "$url"
done
echo "패키지 $(wc -l < plan.txt)개 (sdist $(grep -c ' sdist$' plan.txt || true)개), 폴더 크기 $(du -sh . | cut -f1)"
