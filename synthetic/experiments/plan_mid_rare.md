# mid·rare STZ 환경 실행 계획 (10/28 02:30)

## 범위 (사용자 승인 10/28)
- 환경: mid·rare × fav·avg·unf = 6환경, 모집단 3, n1 400·800, 표본 seed 0–9, n0 8000 → 방법마다 360칸
- 방법: BOMiS, A/B/C, ROS, SMOTE-N, SMOTE-ENC, smote(SMOTE-NC 대체), ARF, CTGAN, TabDDPM
- 제외: TabDiff, TabSyn(시간), ForestDiffusion
- 데이터는 새로 만들지 않음. 모집단 파일(`results/pops`)에 mid·rare 환경이 이미 들어 있음
- 결과 폴더: `results/mid_{fav,avg,unf}`, `results/rare_{fav,avg,unf}` (런타임 폴더 기준)

## 코드 변경 (우리 실행 코드만)
- `scripts/run_easy.py`: `env_list`, `out_dir` 추가, `--levels` 추가(`--all-envs`와 함께 씀). frequent 폴더 이름은 그대로
- `scripts/run_dl.py`: 같은 함수를 써서 `--levels` 추가
- 연결 확인 srun 23252: mid__fav 첫 칸과 rare__unf 마지막 칸에서 ROS·smote 성공, TabDDPM 작업 분할 120개 확인

## 서버 현황 (02:29)
- 기본 QOS: 사용자당 GPU 4, CPU 32, 메모리 128G. 내 다른 job 없음
- 빈 GPU: brl3 4, brl4 3, brl0 1 (brl2는 10/27 GPU 장애 이력으로 GPU job에서 제외, brl1 down)

## 자원 산정 (frequent 실행 sacct 기준)
| 묶음 | 실측 | 요청 |
|---|---|---|
| CPU 묶음(칸 하나에 9방법) | BOMiS 묶음 평균 10분·최장 15분, ARF+smote 평균 3분·최장 10.5분, MaxRSS 최대 1.7G | CPU 4, 10G, 1시간 (`easy_cpu.sbatch` 그대로) |
| CTGAN | 칸당 평균 2분·최장 9분, MaxRSS 1.7G | GPU 1, CPU 2, 6G, 45분 (`easy_gpu.sbatch` 그대로), brl2 제외 |
| TabDDPM | 모집단 3칸 작업 하나에 3–6분 | GPU 1, CPU 6, 24G, 3시간 (`easy_dl.sbatch` 그대로) |

## 배치
| job | 배열 | 동시 실행 | 순서·의존 |
|---|---|---|---|
| mr_cpu 23253 | 0–359 (환경 순서 mid fav→avg→unf → rare fav→avg→unf) | 6 (CTGAN이 CPU 8개를 쓰도록). CTGAN 끝나면 8로 올림(23264가 자동으로) | 바로 시작 |
| mr_ctgan 23254 | 0–359 | GPU 4 | 바로 시작(CPU가 비는 대로) |
| mr_tabddpm 23255 | 0–119 (seed→n1→환경, 작업 하나 = 모집단 3칸) | GPU 4 | `afterany:23253` (CPU 묶음 끝난 뒤) |
| mr_throttle 23264 | – | – | `afterany:23254`, mr_cpu 동시 실행을 8로 올림 |

- CPU 묶음과 TabDDPM을 같이 돌리면 TabDDPM(CPU 6 × 4 = 24)이 CPU 한도를 잡아 BOMiS가 2개씩만 돌아서, 사용자 지시대로 CPU 묶음 뒤에 붙임
- 처음 제출 때 mr_cpu가 CPU 32개를 다 잡아 CTGAN이 QOSMaxCpuPerUserLimit으로 막혀서 02:31에 동시 실행 6으로 줄임(돌던 8개는 그대로 끝남)

## 예상 (보장 아님)
- CTGAN: 360칸 × 2분 / GPU 4 ≈ 3시간 → 05:30 안팎
- CPU 묶음: 칸당 약 13분. 처음 3시간은 6개 동시(약 80칸), 뒤로 8개 동시 → mid 약 08시, rare 약 13시
- TabDDPM: 120작업 × 약 5분 / GPU 4 ≈ 2.5–4시간 → 15:30–17:00
- 다른 사용자의 GPU·CPU 사용에 따라 몇 시간씩 달라질 수 있음
- `normal` QOS(사용자당 한도 없음)를 쓰면 CPU 묶음이 훨씬 빨라지지만 공용 자원이라 승인 필요, 이번엔 쓰지 않음

## 상태 보기
`sacct -j 23253,23254,23255,23264 -X --format=JobID,JobName,State,Elapsed,NodeList -P`
