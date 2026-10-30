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
- 공유 디스크가 바쁘면 설치가 몇 시간 걸린다(10/29 시험에서 env 하나에 2시간 넘게 걸림). job이 중간에 끊겨도(TIMEOUT 등) `bash envs/setup_all.sh`를 다시 실행하면 된다. 있는 env와 받은 wheel은 건너뛰고, pip도 이미 설치된 패키지는 건너뛴다.

**3. 실행 폴더 준비와 확인**

```bash
bash experiments/sync_runtime.sh            # 코드를 $BOMIS_RUNTIME으로 복사
cd $BOMIS_RUNTIME
srun --ntasks=1 --cpus-per-task=2 --mem=8G bash -c 'source config/paths.sh && $BOMIS_PY -I scripts/build_pops.py --start 0 --n 3 --out results/pops'
sbatch experiments/pytest.sbatch            # 테스트
```

실험 실행은 `synthetic/README.md`와 `experiments/` 참고. sbatch는 항상 `$BOMIS_RUNTIME`에서 제출한다(각 sbatch가 제출 폴더의 `config/paths.sh`를 읽는다).

## 원래 결과와 같은지

10/29 시험 설치에서 새 env를 원래 env와 파일 단위(md5)로 비교했다. bomis env는 21,663개 파일이 모두 같았다. baseline env는 실험 코드가 쓰지 않는 부분만 달랐다: 설치 도구(pip, wheel), setuptools 설치 형식, tabpfn_extensions(원래는 GitHub 커밋 `9ffa814`에서 설치), tabpfn 실행 중 받은 체크포인트.

이어서 새 env로 frequent__fav 한 칸(모집단 0, n1=400, 표본 seed 0)을 돌려 원래 결과 csv와 열별로 비교했다(brl4, RTX A5000).

| 결과 | 방법 |
|---|---|
| 모든 값이 같음 | A, B, C, ROS, SMOTE, SMOTE-N, SMOTE-ENC, ARF, CTGAN, TabDDPM, TabDiff |
| 조금 다름 | TabSyn (recovery 0.5988 → 0.5978, leak 0.1555 → 0.1512). 원래 실행은 brl5(RTX 4090)였다 |
| 아직 확인 전 | BOMiS, ForestDiffusion, `pytest` |

TabSyn 차이의 원인이 GPU 종류인지는 아직 확인하지 않았다. 칸 하나만 비교한 결과이므로 다른 칸도 같다고 보장하지는 않는다.

같은 모집단 seed와 표본 seed면 DGP와 표본은 비트 단위로 같다. 방법 결과는 다음을 알아 둔다.

- CPU 방법(A/B/C, ROS, SMOTE 계열, ARF)은 같은 env에서 같은 숫자가 나온다.
- BOMiS(CPU, Transformer 학습)는 CPU 종류에 따라 recovery가 최대 약 0.008 달랐다(Intel brl0·2·3·4 대 AMD hpc-stat1·brl5). 같은 CPU 종류에서는 같았다.
- GPU 방법은 GPU 종류나 실행에 따라 조금 다를 수 있다.
- ForestDiffusion은 칸당 시간이 길고 brl4에서 joblib 오류(`BrokenProcessPool`)로 실패한 적이 있다. 원래 실험처럼 brl4를 빼고 CPU 8개로 돌린다.
