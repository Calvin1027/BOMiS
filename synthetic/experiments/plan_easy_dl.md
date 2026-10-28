# 제출 계획: 쉬운 환경 DL baseline (TabDDPM, TabSyn) (2026-10-27 10:21 조회)

## 현황
- brl1 down, brl2는 GPU 고장 이력(10/27 "No devices were found")이라 제외. 빈 GPU brl5 8, brl3 5, brl0·brl4 일부. 빈 CPU brl5 96, brl3 26.
- default_qos 사용자당 GPU 4, CPU 32, 메모리 128G. 내 다른 job 없음.

## 작업
- 모델 2개 × 3환경(frequent__fav/avg/unf) × 모집단 3 × $n_1$ {400, 800} × 표본 seed 0–9 = 모델당 180칸, 합계 360칸.
- 데이터: 환경 × 모집단 = 9개 모집단 설정, 각 설정에서 표본 20개(n1 2 × seed 10). 칸마다 모델을 새로 학습한다.
- 배열 작업 120개(0–119) = (seed, $n_1$, 환경) 묶음 60개 × 모델 2개. 작업 하나가 모집단 3칸을 GPU 1장에서 동시에 돈다(slot 0–2).
- 순서: seed → $n_1$ → 환경, 같은 묶음 안에서 TabDDPM 다음 TabSyn. 일찍 끝난 결과가 9개 설정과 $n_1$ 둘을 고루 덮는다. 0번은 TabDDPM(짧은 작업).
- 결과는 기존 폴더 `results/easy`, `easy_avg`, `easy_unf`에 방법 이름 `tabddpm`, `tabsyn`으로 쌓인다. 이미 csv가 있는 칸은 건너뛴다.

## 설정 (실험 조건)
- TabSyn: `categorical_baselines` `models.reference.json` 그대로(epochs 10001, vae_epochs 4000, batch 4096, sampling_steps 50). 실패 시 RETRY(max_raw_multiplier 2000).
- TabDDPM: 과거 파이프라인 최종 설정(`Baseline_exp_v4/configs/models.climb_final.v8.full_access.no_forest.json`) 그대로. 업데이트 10,000회, diffusion 1,000 step, batch 1024, MLP (256, 256), lr 2e-4 → 1e-4 → 5e-5 재시도, 기울기 clip 1.0.
  - 과거 어댑터는 숫자 열이 있는 표를 전제로 해서 범주형 8열 표에 그대로 못 쓴다. `baselines/tabddpm_cli.py`가 과거 코드의 설정 생성(`_config`), 안정화 패치, 성공 기준을 import해 쓰고 X_num 파일만 만들지 않는다. 과거 코드 수정 없음.

## 산정 근거
- 연결 확인(srun 20876, 스모크 설정, fav 모집단 0, $n_1$=400, seed 0): TabDDPM 104초(대부분 프로세스·소스 복사 부담), TabSyn 13초. 둘 다 생성·채점 성공.
- 본 설정 시간은 이 서버 기록 없음. 과거 실데이터 표(Table 23, A5000) TabSyn 평균 1,436초. TabDDPM은 설정상 수 분으로 추정.
- 작업당 GPU 1, CPU 6(slot 3 × 스레드 2), 메모리 24G(slot당 8G, CTGAN 실측 2.0G에 TabSyn 여유), 시간 3시간(TabSyn 3칸 동시, 측정 없음이라 넉넉히).
- 4개 동시 실행이면 CPU 24/32, 메모리 96/128G.

## 전략
| 전략 | 동시 실행 칸 | 비고 |
|---|---|---|
| A. 작업당 1칸, GPU 4장 | 4 | 작은 모델이라 GPU가 많이 논다 |
| B. 작업당 3칸 동시(선택) | 12 | GPU 공유로 칸당 시간은 늘지만 처리량이 커질 것으로 예상(측정 없음) |
| C. 모델별 배열 2개 | - | 먼저 낸 배열이 GPU를 다 써서 다른 모델 결과가 늦게 나온다 |

B를 고른다. 두 모델을 한 배열에 섞어 GPU 4장을 계속 채우고, 랩미팅(14:00) 전에 끝나지 않은 칸은 앞 seed 결과로 미리 본다.

## GPU/CPU 선택
- 둘 다 torch 모델이고 과거 실행도 GPU였다. GPU job.
