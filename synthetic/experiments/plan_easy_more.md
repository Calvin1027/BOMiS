# 제출 계획: 쉬운 환경 나머지 baseline (2026-10-27 13:05 조회)

## 범위 (사용자 지시 10/27 13:0x: 같은 세팅으로 나머지 baseline 전부, ForestDiffusion은 시간 측정까지)
- 세팅 그대로: frequent × (fav, avg, unf), 모집단 3, $n_1$ {400, 800}, 표본 seed 0–9 = 180칸.
- ARF, smote(SMOTE-NC 대체. 범주형만 있는 표라 SMOTE-NC는 정의되지 않아 baseline 프로젝트의 one-hot SMOTE): CPU, 180칸 전부.
- TabDiff: 본 설정(baseline 프로젝트 `models.reference.json`, 8000 epoch) 3칸으로 시간 측정 후 전체 여부 결정.
- ForestDiffusion(Forest-Flow): 과거 파이프라인 어댑터(Baseline_exp_v4 `ForestFlowAdapter`, 수정 없음)를 `baselines/forestflow_cli.py`로 부름. 과거 최종 설정(Baseline_exp_v2 `models.climb_final.v4.json`: n_t 50, duplicate_K 100, 트리 100개, 깊이 7, n_jobs 2). 1칸으로 시간 측정. 과거 파이프라인에서 시간 때문에 제외된 모델.

## 현황
- DL 배열 20883 실행 중(GPU 4장, CPU 24). QOS 사용자당 GPU 4, CPU 32, 메모리 128G.
- TabDiff 측정에 GPU 1장을 주려고 20883을 ArrayTaskThrottle 3으로 낮춤(남은 작업은 3장으로 계속).

## 연결 확인 (srun 21183, 짧은 설정, fav 모집단 0, $n_1$=400, seed 0)
- ARF 198초 통과(recovery 0.55, leak 0.12), smote 15초 통과(새 셀 0개: one-hot 보간 뒤 범주로 되돌리면 기존 셀로 돌아감. ROS와 같은 결과).
- ForestDiffusion은 짧은 설정에서도 남은 6분 안에 끝나지 않아 srun 시간 제한으로 중단. sbatch 스모크로 다시 확인.

## 제출
| job | 내용 | 자원 | 의존 |
|---|---|---|---|
| 21198 ff_smoke | ForestDiffusion 짧은 설정 1칸(`results/cpu_smoke`) | CPU 8, 16G, 1시간 | – |
| 21199 tabdiff_smoke | TabDiff 짧은 설정 1칸(`results/dl_smoke`) | GPU 1, CPU 2, 6G, 30분, brl2 제외 | – |
| 21200 tabdiff_time | TabDiff 본 설정, fav seed 0 $n_1$=400 모집단 3칸 동시 | GPU 1, CPU 6, 24G, 8시간 | afterok 21199 |
| 21201 ff_time | ForestDiffusion 본 설정, fav 모집단 0 $n_1$=400 seed 0 1칸 | CPU 8, 24G, 3.5시간(래퍼 timeout 3시간) | afterok 21198 |
| 21202 easy_arf | ARF·smote 180작업(`--all-envs`, 작업 번호 = 환경 × 60 + 칸) | CPU 4, 10G, 1시간, %1 | – |

- 측정 작업의 결과는 본 설정이라 그대로 본 결과 폴더에 쌓인다.
- easy_arf는 DL 배열이 CPU를 쓰는 동안 %1로 두고, 20883이 끝나면 올린다.

## GPU/CPU 선택
- ARF(트리), smote(sklearn), ForestDiffusion(XGBoost, 과거 최종 설정이 CPU)는 CPU. TabDiff는 torch라 GPU.
