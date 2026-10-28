# 사용자별 경로 설정. 이 파일을 config/paths.sh로 복사해서 쓴다(paths.sh는 git에 올라가지 않는다).
#   cp config/paths.example.sh config/paths.sh
# sbatch 파일과 sync 스크립트가 이 파일을 source한다. Python 코드는 여기서 export한 환경변수를 읽고,
# 값이 없으면 원래 서버(calvinkim1027)의 경로를 기본값으로 쓴다. 값을 빈 문자열로 두면 그 경로는 쓰지 않는다.
# 아래 값은 envs/README.md의 설치 절차를 그대로 따랐을 때의 경로다.

# 실행 폴더: 계산 노드가 보는 곳(/mnt/data1 아래). sync_runtime.sh가 코드를 여기로 복사하고, 결과·로그가 여기에 쌓인다.
export BOMIS_RUNTIME=/mnt/data1/$USER/BOMiS_synthetic_runtime

# BOMiS·DGP·채점용 env (Python 3.11, envs/bomis/requirements.txt)
export BOMIS_PY=/mnt/data1/$USER/miniconda3/envs/bomis/bin/python

# baseline 기반 env (Python 3.10, envs/baselines/base.txt)
export BL_PYTHON=/mnt/data1/$USER/miniconda3/envs/bomis_baselines/bin/python

# baseline env 위에 PYTHONPATH로 얹는 패키지 폴더 3개 (envs/baselines/{generation,tabddpm_extras,extras}.txt)
BL_LAYERS=/mnt/data1/$USER/bomis_baseline_layers
export BL_GEN_SITE=$BL_LAYERS/generation
export BL_TABDDPM_SITE=$BL_LAYERS/tabddpm_extras
export BL_DEPS=$BL_LAYERS/extras
export BL_EXTRAS_TAR=          # extras를 압축 파일로 풀던 옛 방식. 위 BL_DEPS를 쓰므로 비워 둔다

# baseline 코드: 저장소 synthetic/third_party/ (sync_runtime.sh가 런타임 폴더로 복사)
export BL_SRC=$BOMIS_RUNTIME/third_party/categorical_baselines
export BL_V4_SRC=$BOMIS_RUNTIME/third_party/baseline_exp_v4/src
export BL_TABDDPM_SOURCE=$BOMIS_RUNTIME/third_party/tab-ddpm
export BL_FF_SOURCE=$BOMIS_RUNTIME/third_party/ForestDiffusion
export BL_FF_PATCH=$BOMIS_RUNTIME/third_party/patches/forest_flow_target_label.patch

# sync_baselines.sh 전용(옛 방식: 연구실 categorical_baselines 작업 폴더를 런타임으로 복사). third_party/를 쓰면 필요 없다.
export BL_CATEGORICAL_SRC=
