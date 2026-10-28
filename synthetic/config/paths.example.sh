# 사용자별 경로 설정. 이 파일을 config/paths.sh로 복사해서 자기 경로로 고친다(paths.sh는 git에 올라가지 않는다).
#   cp config/paths.example.sh config/paths.sh
# sbatch 파일과 sync 스크립트가 이 파일을 source한다. Python 코드는 여기서 export한 환경변수를 읽고,
# 값이 없으면 원래 서버(calvinkim1027)의 경로를 기본값으로 쓴다. 값을 빈 문자열로 두면 그 경로는 쓰지 않는다.

# 실행 폴더: 계산 노드가 보는 곳(/mnt/data1 아래). 코드 사본, 결과, 로그가 여기에 쌓인다. 사람마다 따로 둔다.
export BOMIS_RUNTIME=/mnt/data1/$USER/BOMiS_synthetic_runtime

# BOMiS·DGP·채점용 env의 python (Python 3.11, envs/bomis 참고)
export BOMIS_PY=/mnt/data1/$USER/miniconda3/envs/bomis/bin/python

# baseline용 env의 python (Python 3.10, envs/baselines 참고)
export BL_PYTHON=/mnt/data1/$USER/miniconda3/envs/bomis_baselines/bin/python

# categorical_baselines 프로젝트 코드 원본(sync_baselines.sh가 $BOMIS_RUNTIME/baselines_src로 복사)
export BL_CATEGORICAL_SRC=

# 과거 파이프라인 코드(TabDDPM, ForestDiffusion)
export BL_V4_SRC=
export BL_TABDDPM_SOURCE=
export BL_FF_SOURCE=
export BL_FF_PATCH=

# 원래 서버에서 baseline env 위에 PYTHONPATH로 덧대던 패키지 폴더. env에 모두 설치했다면 빈 값으로 둔다.
export BL_GEN_SITE=
export BL_TABDDPM_SITE=
export BL_EXTRAS_TAR=
