# 실험 환경 설치 (brl 서버, Slurm)

원래 서버(calvinkim1027)에서 10/27–28 실험을 돌린 환경을 패키지 버전까지 같게 다시 만든다. 이 서버는 로그인 노드에서 Python 실행이 금지되고, 계산 노드는 인터넷이 안 된다. 그래서 인터넷이 필요한 단계(conda env 만들기, wheel 받기)는 로그인 노드에서, 설치는 계산 노드에서 한다.

## 환경 구성

| 이름 | 용도 | Python | 정의 파일 |
|---|---|---|---|
| `bomis` env | DGP, 채점, BOMiS(원저자 코드), A/B/C, ROS | 3.11.17 | `bomis/requirements.txt` (39개) |
| `bomis_baselines` env | 외부 baseline 실행(SMOTE 계열, ARF, CTGAN, TabSyn, TabDiff, TabDDPM, ForestDiffusion) | 3.10.20 | `baselines/base.txt` (129개, torch 2.5.1+cu121) |
| 덧댄 폴더 3개 | baseline env 위에 PYTHONPATH로 얹는 패키지 | – | `baselines/generation.txt`(ctgan, imbalanced-learn, rdt 등), `tabddpm_extras.txt`(rtdl, setuptools 79), `extras.txt`(timm 등) |

baseline 쪽을 env 하나로 합치지 않고 폴더를 덧대는 이유: 원래 환경이 이 구조였고, 실행 경로마다 PYTHONPATH 순서가 다르다(`baselines/external.py`). 같은 결과를 내려면 구조까지 같게 둔다.

## 설치

**0. 준비 (한 번)**

- miniconda를 `/mnt/data1/$USER/miniconda3`에 설치한다(로그인 노드):
  ```bash
  curl -fsSLO https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash Miniconda3-latest-Linux-x86_64.sh -b -p /mnt/data1/$USER/miniconda3
  ```
- `jq`, `curl`이 있어야 한다(서버의 `/conda/anaconda3/bin`에 있음).

**1. 저장소와 경로 설정**

```bash
git clone git@github.com:Calvin1027/BOMiS.git
cd BOMiS/synthetic
cp config/paths.example.sh config/paths.sh   # 경로를 바꾸려면 이 파일을 고친다
```

**2. 설치 (로그인 노드에서 한 번)**

```bash
bash envs/setup_all.sh
```

- conda env 두 개를 만들고, wheel 약 6GB를 `/mnt/data1/$USER/bomis_wheels`에 받은 뒤, 설치 job 3개를 sbatch로 제출한다.
- 기본 QOS가 꽉 찼으면 `SBATCH_OPTS="--qos=normal" bash envs/setup_all.sh`
- 로그(`/mnt/data1/$USER/bomis_setup_logs`) 끝줄이 `INSTALL OK`(두 개), `LAYERS OK`면 끝이다. `INSTALL OK`는 설치된 버전이 정의 파일과 모두 같다는 뜻이다.

**3. 실행 폴더 준비와 확인**

```bash
bash experiments/sync_runtime.sh            # 코드를 $BOMIS_RUNTIME으로 복사
cd $BOMIS_RUNTIME
srun --ntasks=1 --cpus-per-task=2 --mem=8G bash -c 'source config/paths.sh && $BOMIS_PY -I scripts/build_pops.py --start 0 --n 3 --out results/pops'
sbatch experiments/pytest.sbatch            # 테스트
```

실험 실행은 `synthetic/README.md`와 `experiments/` 참고. sbatch는 항상 `$BOMIS_RUNTIME`에서 제출한다(각 sbatch가 제출 폴더의 `config/paths.sh`를 읽는다).

## 원래 결과와 같은지

같은 모집단 seed와 표본 seed면 DGP와 표본은 비트 단위로 같다. 방법 결과는 다음을 알아 둔다.

- CPU 방법(A/B/C, ROS, SMOTE 계열, ARF)은 같은 env에서 같은 숫자가 나온다.
- BOMiS(CPU, Transformer 학습)는 CPU 종류에 따라 recovery가 최대 약 0.008 달랐다(Intel brl0·2·3·4 대 AMD hpc-stat1·brl5). 같은 CPU 종류에서는 같았다.
- GPU 방법(CTGAN, TabSyn, TabDiff, TabDDPM)은 GPU 연산 특성상 실행마다 조금 다를 수 있다.
