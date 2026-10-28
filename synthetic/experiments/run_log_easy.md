# 쉬운 환경 본실행 기록 (frequent__fav, 모집단 3개 × n1 {400, 800} × 표본 seed 0–9 = 60칸)

- 제출: CPU 배열 20207 (`easy_cpu.sbatch`, 4코어·10G·60분, 동시 6개로 제한), GPU 배열 20208 (`easy_gpu.sbatch`, GPU 1·2코어·6G·45분, ctgan)
- 결과: `/mnt/data1/calvinkim1027/BOMIS_new_수만이형_runtime/results/easy/<방법>/<pop>_<n1>_<seed>.csv`, 실패는 `results/easy/_failed/`
- 로그: `runtime/logs/easy_{cpu,gpu}_<jobid>_<idx>.log`
- 상태 보기: `sacct -j 20207,20208 -X --format=JobID,State,Elapsed,NodeList -P`

## 제출 전 확인 (10/27 03:0x)

- 빠른 테스트 41개 통과, 느린 진단 테스트 1개 통과, 1단계 점검 `stage1_pass: true`
- 짧은 설정 end-to-end(모집단 0, n1=400, seed 0, BOMiS 250 step, baseline 스모크 설정): 7개 방법 모두 성공, 요약 표 생성 확인 (`reports/easy_quick.md`)
- sbatch 수정: 끝날 때 `/tmp/bl_deps_<jobid>` 지우는 trap 추가, 시간 CPU 40→60분·GPU 30→45분

## 30분 점검

| 시각 | CPU 완료/실행/대기/실패 | GPU 완료/실행/대기/실패 | 조치 |
|---|---|---|---|
| 03:20 | 0/7/53/0 | 0/0/60/0 | 제출 직후. CPU 동시 6개 제한(ArrayTaskThrottle=6)으로 GPU 몫 CPU 8 남김 |
| 03:40 | 7/6/47/0 (작업당 14–15분) | 13/4/…/16 | GPU 실패 16개는 모두 brl2(`nvidia-smi: No devices were found`, exit 6). 대기분 brl2 제외(scontrol), 실패 13–18·21–30 재제출 → 20284(--exclude=brl2). ctgan은 첫 시도 성공, 1.5–8분 |

## 추가 제출 (사용자 지시 10/27 03:3x: fav 끝나면 같은 frequent의 avg·unf도)

- `run_easy.py`에 `--env` 인자 추가(기본 frequent__fav 그대로), `summarize_easy.py` 제목을 데이터의 env로
- 연결 확인 job 20287 (`scripts/plumb_env.py`, 로그 `logs/plumb_20287.log`, 끝줄 `PLUMB OK`면 통과)
- avg: CPU 20288 (`--array=0-59%3`, afterany:20207), GPU 20289 (afterany:20208:20284, brl2 제외), 결과 `results/easy_avg`
- unf: CPU 20290 (`%3`), GPU 20291, 결과 `results/easy_unf`
- 상태 보기: `sacct -j 20207,20208,20284,20287,20288,20289,20290,20291 -X --format=JobID,State,Elapsed,NodeList -P`
- 예상: fav CPU ~06:00 끝, avg·unf CPU 120칸/동시 6/15분 → ~11:00 끝 (대기에 따라 달라짐)

| 시각 | 상태 요약 | 조치 |
|---|---|---|
| 03:49 | fav CPU 27/60 완료·6 실행, 실패 0. fav ctgan 59/60(20284 마지막 1개 실행 중). 연결 확인 20287 `PLUMB OK`(avg·unf에서 env별 M0 0.633/0.598로 바뀜, C 전부 인증, BOMiS leak 0) | 조치 없음. avg·unf는 의존성 대기 |
| 04:19 | fav CPU 45/60 완료·6 실행, 실패 0. fav ctgan 60/60 끝. avg ctgan(20289) 39/60, 로그·csv에 `env frequent__avg` 확인 | 조치 없음 |
| 04:49 | **fav 60칸 × 8개 방법 전부 완료, 실패 0.** avg CPU 20288·unf CPU 20290 각 3개 실행, avg ctgan 60/60 끝, unf ctgan 19/60 | fav 요약은 CPU 한도가 차서 srun 대신 sbatch 20565로 제출(`reports/easy_preview.md`) |
| 05:19 | avg CPU 12/60, unf CPU 14/60, 실패 0. ctgan avg·unf 모두 60/60 끝 → GPU 배열 전부 종료. fav 요약 완료(`reports/easy_preview.md`) | GPU 몫 CPU 8이 비어 **avg 20288·unf 20290 ArrayTaskThrottle 3→4 (합계 8개=32코어)**. 이후 점검에서 4로 유지. 예상 끝 ~08:15 |

### fav 요약 첫 검토 (05:19)

- BOMiS: recovery 0.424 / 0.435 (n1 400 / 800), leak 0, control 1.0. SMOTEN·SMOTE-ENC 대비 recovery +0.39, leak도 낮음 (짝 차이 MCSE 0.013–0.016)
- CTGAN: recovery 0.426 / 0.365로 BOMiS와 비슷하지만 leak 0.25, control 0.68 → BOMiS가 control +0.31~0.33
- A/B/C: C는 30칸 모두 4개 규칙 인증, 거짓 인증 0. A도 STZ 억제 1.0 (fav는 금지 쌍이 D1에 나올 수 없어 D1만으로도 막힘), 대신 SZ 회수 0.981/0.996로 C(0.997/0.999)보다 약간 낮음. B는 STZ 억제 0.75
- BOMiS 진단: R1≈1(support 손실 거의 없음), 손실 대부분이 개수(count) 0.37 — 새 셀 L'≈1150–1470개로 SZ 셀 ~28000개 중 질량 상위만 덮음. 선택 손실 0.19. M_hat ≈ M0 (오차 ≤0.004). mask 없는 Transformer는 STZ에 0.30 질량을 두고 mask가 0으로 만듦
| 05:59 | avg CPU 26/60·unf CPU 31/60 완료, 각 4개 실행, 실패 0 | 조치 없음 |
| 06:29 | avg CPU 38/60·unf CPU 43/60 완료, 각 4개 실행, 실패 0 | 조치 없음 |
| 06:59 | avg CPU 49/60·unf CPU 57/60(3개 실행, 대기 없음), 실패 0 | unf 대기분이 없어 avg 20288 ArrayTaskThrottle 4→8 |
| 07:29 | **avg·unf 60칸 × 8개 방법 전부 완료, 실패 0** (전체 3환경 × 60칸 × 8방법 = 1440행) | srun으로 avg·unf 요약 → `reports/easy_preview_avg.md`, `_unf.md`. 보고서 3개를 /data1 프로젝트 `reports/`에 복사. 점검 cron 삭제 |

### 3환경 비교 (07:29, n1 = 400 / 800, 칸 30개씩, ± MCSE)

| 환경 | BOMiS recovery | BOMiS leak | CTGAN recovery / leak | SMOTEN recovery | BOMiS−CTGAN control | A SZ 회수(개수) | B STZ 억제 | C 인증·거짓 |
|---|---|---|---|---|---|---|---|---|
| fav (γ=0) | 0.424 / 0.435 | 0 | 0.43/0.25, 0.37/0.24 | 0.032 / 0.038 | +0.31 / +0.33 | 0.865 / 0.962 | 0.756 / 0.753 | 1.0 · 0 |
| avg (γ=0.5) | 0.434 / 0.453 | 0 | 0.42/0.25, 0.36/0.24 | 0.036 / 0.039 | +0.30 / +0.32 | 0.854 / 0.951 | 0.691 / 0.689 | 1.0 · 0 |
| unf (γ=1) | 0.458 / 0.485 | 0 | 0.43/0.24, 0.39/0.23 | 0.040 / 0.040 | +0.30 / +0.31 | 0.700 / 0.867 | 0.696 / 0.689 | 1.0 · 0 |

- 세 환경 모두 BOMiS가 SMOTEN·SMOTE-ENC보다 recovery +0.39~0.45, CTGAN과는 recovery 비슷(n1=400에서 0.00~0.03, n1=800에서 +0.07~0.10)하고 leak 0 대 0.23~0.25
- 겹침 축(γ)의 효과는 BOMiS 생성보다 support 단계에서 보임: A(D1만)의 SZ 개수 회수가 fav→unf에서 0.865→0.700 (n1=400)으로 떨어지고, C는 0.96~0.99 유지. B의 STZ 억제는 fav 0.75 → avg·unf 0.69
- BOMiS recovery는 오히려 unf에서 약간 높음(0.458 vs 0.424). 손실은 세 환경 모두 개수 단계가 대부분(loss_count 0.34~0.38)
- 해석 주의: A의 STZ 억제가 늘 1.0인 것은 설계상 local support가 성립(pairwise feasible = S1)해서 금지 쌍이 D1에 절대 안 나오기 때문

## DL baseline 추가 (사용자 지시 10/27 10:1x: TabSyn·TabDDPM, 3환경 모두)

- 계획: `experiments/plan_easy_dl.md`. 모델당 180칸(3환경 × 모집단 3 × n1 2 × seed 10), 합계 360칸
- TabDDPM 연결: `baselines/tabddpm_cli.py`(과거 Baseline_exp_v4 코드의 설정·패치·성공 기준을 import, 숫자 열 0개라 X_num 파일만 생략), `external.py`에 tabddpm 분기 추가
- 연결 확인 srun 20876(스모크 설정, `results/dl_smoke`): TabDDPM 104초, TabSyn 13초, 둘 다 채점까지 성공
- 본 제출: 배열 20883 (`experiments/easy_dl.sbatch`, 0–119, GPU 1·CPU 6·24G·3시간, brl2 제외). 작업 하나 = 한 모델의 모집단 3칸 동시. 순서 seed → n1 → 환경, 모델 교대
- slot 로그: `logs/easy_dl_20883_<idx>_s<slot>.log`
- 상태 보기: `sacct -j 20883 -X --format=JobID,State,Elapsed,NodeList -P`

| 시각 | 상태 요약 | 조치 |
|---|---|---|
| 10:29 | TabDDPM 첫 작업 0·2 완료(3칸 동시 3분 33–44초, 칸당 190–214초), 6칸 실패 0. TabSyn 1·3 실행 4분째 | 조치 없음. 3분 점검 cron 시작 |
| 10:35 | TabSyn 첫 작업 3 완료(3칸 동시 9분, 칸당 457–517초, 첫 시도 성공), 실패 0. 완료 4·실행 4 | 예상: TabDDPM 60×3.5분 + TabSyn 60×9분 ≈ 750 GPU-분 / 4 → 13:30–13:40 끝 |
| 10:46 | 완료 10·실행 4, TabDDPM 18칸·TabSyn 12칸, 실패 0. 노드별 차이: TabSyn brl5 9분, brl3 14.7분 / TabDDPM brl5 3–3.5분, brl4 6.2분 | 예상 수정: 작업 평균 ~7.5분 × 120 / 4 ≈ 225분 → 14:10 안팎. 14:00엔 seed 0–8 정도 |
| 11:22 | 완료 30·실행 4, TabDDPM 48칸·TabSyn 42칸(합 90/360), 실패 0. 속도 분당 약 1.67칸 | 예상 끝 14:05 안팎 유지 |
| 11:49 | 완료 42·실행 4, TabDDPM 66칸·TabSyn 60칸(합 126/360), 실패 0. 속도가 분당 1.33칸으로 느려짐: 최근 작업이 brl3·brl4(A5000)에 배정, TabSyn 13–16분(brl5 4090은 7.7분). brl5는 GPU 8장 모두 다른 사용자 할당이라 노드 고정은 오히려 대기만 늘어 하지 않음 | 예상 끝 14:30–14:50. 14:00엔 약 300칸 = seed 0–7 완료, seed 8 일부. 13:30쯤 그 시점 결과로 그림·표 갱신 예정 |
| 11:52 | 완료 42·실행 4, TabDDPM 66칸·TabSyn 62칸(128/360), 실패 0. 최근 30분 속도 분당 1.27칸으로 느려짐: brl5 GPU가 다른 사용자로 다 차서 작업이 brl3·brl4(A5000, 부하 있음)로 감. TabSyn brl5 7.8분 vs brl3·4 12.6–15.9분 | brl5에 빈 GPU가 없어 노드 지정 이득 없음, 조치 없음. 예상 끝 14:30–15:00, 13:50에 약 280/360칸(seed 0–7) |
| 12:19 | 완료 53·실행 4, TabDDPM 84칸·TabSyn 77칸(합 161/360), 실패 0. 최근 30분 분당 1.17칸 | 예상 끝 15:00 안팎. 14:00엔 약 280칸 = seed 0–6 완료, seed 7 일부 |
| 12:22 | 완료 54·실행 4(모두 brl3·brl4), TabDDPM 85칸·TabSyn 78칸(163/360), 실패 0. 최근 30분 분당 1.17칸 | brl5는 다른 사용자가 GPU 8장 모두 사용 중, 조치 없음. 예상 끝 15:00–15:15, 13:50에 약 265칸(seed 0–6, 일부 7) |
| 12:40 | 완료 61·실행 4, TabDDPM 93칸·TabSyn 90칸(183/360), 실패 0 | 사용자 요청으로 3분 점검 cron 중단(job 20883은 계속 실행) |
| 12:49 | 완료 64·실행 4, TabDDPM 102칸·TabSyn 93칸(합 195/360), 실패 0. 최근 30분 분당 1.13칸 | 예상 끝 15:10 안팎, 14:00엔 약 275칸(seed 0–6) |
| 13:13 | 완료 72·실행 3, TabDDPM 111칸·TabSyn 106칸(합 217/360), 실패 0. 20883에 동시 3개 제한(%3)이 걸림(이 세션에서 건 것 아님), CPU job 21183(env, 4코어) 실행 중 | 제한은 그대로 둠. 남은 143칸, GPU 3장이면 끝 15:30–15:45 예상 |

## 나머지 baseline (사용자 지시 10/27 13:0x, 랩미팅 하루 연기)

- 계획: `experiments/plan_easy_more.md`. ARF·smote 전체, TabDiff·ForestDiffusion은 시간 측정 먼저
- 코드: `external.py`에 arf·smote·forest_flow 추가(과거 파이프라인 분기 `OLD_PIPELINE`로 일반화, TabDDPM 동작 그대로), `baselines/forestflow_cli.py` 새로, `run_easy.py --methods --all-envs`, `run_dl.py --models`
- 20883 ArrayTaskThrottle 4→3 (TabDiff 측정에 GPU 1장)
- 제출: ff_smoke 21198, tabdiff_smoke 21199, tabdiff_time 21200, ff_time 21201, easy_arf 21202(0–179%1)
- 상태 보기: `sacct -j 20883,21198,21199,21200,21201,21202 -X --format=JobID,JobName,State,Elapsed,NodeList -P`
| 13:34 | TabDDPM 120칸·TabSyn 114칸, 실패 0. 그림·표 갱신 job 21213(CPU 한도로 대기 → DL 대기분 잠깐 hold 후 실행, 1분 15초) 완료, hold 해제 | `reports/easy_results_1027.md` 7절·요약에 DL 결과 반영 |
| 13:2x | ff_smoke 21198 통과(39초, recovery 0.45, leak 0.28), tabdiff_smoke 21199 통과(35초). ff_time 21201(brl4) 원래 설정에서 86초 만에 실패: joblib `BrokenProcessPool: A task has failed to un-serialize`. traceback 남기게 고쳐 다시 돌린 21210(brl4)도 478초에 같은 오류 | 원인 확인용 ff_debug 21226(brl2, CPU 4)을 forestflow_cli 직접 호출로 돌림 |
| 13:47 | ff_debug는 brl2에서 오류 없이 진행: 10분에 XGBoost 학습 9개(n_t 50 × 2클래스 = 100개 중) → 칸당 약 1–2시간(CPU 4). brl4에서만 실패해서 brl4 메모리 부족(FREE_MEM 수 GB) 때 joblib 메모리맵 문제로 추정. 그리고 ff_debug의 24G 때문에 20883이 QOSMaxMemoryPerUser로 2개만 돌고 있었음 | ff_debug 취소(측정 목적은 칸당 시간 규모 확인으로 충분), DL 3개 실행 복구. ForestDiffusion 원래 설정 1칸 측정 ff_time 21252를 brl4 제외, CPU 8, afterany:20883으로 다시 냄 |
| 14:04 | TabDiff 측정 첫 칸 완료: 2074초(약 35분, 3칸 동시, brl4), recovery 0.60, leak 0.10, control 0.85, 첫 시도 성공. TabDDPM 131·TabSyn 124·ARF 15·smote 15, 새 실패 0 | TabDiff 전체 추정: 작업당 약 37분 × 59작업 / GPU 4 ≈ 9시간 → 20883 끝(~16시) 뒤 시작하면 내일 01–02시. 사용자에게 전체 제출 여부 질문 |
| 14:07 | 사용자 승인으로 TabDiff 전체 배열 easy_tabdiff 21306 제출(`easy_dl.sbatch`, `--models tabdiff`, 1–59, 0번은 측정 21200이 맡음, 시간 2시간). 처음 상태 Pending(QOSMaxCpuPerUserLimit, 정상 대기) | 21200이나 20883 작업이 끝나 GPU가 비는 대로 시작 |
| 14:12 | tabdiff_time 21200 완료(45분 42초, 3칸 2074–2719초, recovery 0.54–0.60, leak 0.08–0.10). easy_tabdiff 21306 작업 1 시작 | TabDiff 끝 예상: 59작업 × ~46분 / GPU 4 ≈ 11시간 → 내일 03–04시. 점검 방식 변경(사용자 지시): 10분 cron 대신 사건 감시 스크립트(gpu-monitoring 스킬 watch_slurm.sh) + 1시간 대비 점검(매시 17분) |
| 14:19 | 완료 89·실행 3, TabDDPM 138칸·TabSyn 130칸(합 268/360), 실패 0. 13:34 이후 %3 제한과 메모리 한도(다른 job과 공유)로 분당 0.76칸 | 예상 끝 16:20 안팎 |
| 15:38 | TabDDPM 168·TabSyn 162(330/360), ARF 38·smote 37, TabDiff 6칸, 새 실패 0, 감시 스크립트 정상 | CPU 28/32·메모리 106/128G라 easy_arf 21202 ArrayTaskThrottle 1→2 (DL 작업이 끝나면 TabDiff 작업이 들어갈 CPU 6은 남음) |
| 16:22 | 감시 스크립트 사건: 20883 120작업 모두 COMPLETED(sacct 확인). TabDDPM·TabSyn 180칸씩 완료, 실패 0. 같은 시각 사용자 쪽 job easy_final 21580(요약, CPU 2)이 CPU 한도로 대기 | easy_arf 21202 Throttle 2→1로 잠시 낮춰 21580 시작시킨 뒤 다시 2. 다른 세션이 reports/에 fig4·fig5·easy_compare4_table.md를 만들고 있어 보고서 갱신은 보류하고 사용자에게 확인. 감시 스크립트를 21202·21252·21306으로 다시 띄움 |
| 16:22 | **20883 전체 120작업 완료, TabDDPM·TabSyn 각 180칸, 실패 0** | 3분 점검 cron 삭제. 최종 요약 job 21580(summarize_easy 3환경 + make_easy_report) 완료 → `reports/easy_preview*.md`, `easy_report_tables.md`, `figures/easy_fig*.png`, `easy_results_1027.md` 7절 최종 숫자로 갱신 |

## Chao1 상한 ablation 시간 측정 (10/27 22:4x)
- job 22582 (`experiments/ablation_time.sbatch`, `scripts/run_ablation.py`, CPU 4·10G, brl0): COMPLETED 12분 4초, MaxRSS 0.49G
- fav pop0 n1=400 seed0: fit 587초, sample 조건당 2.5초. base recovery 0.5090 = 기존 결과와 같음
- nocap: recovery 0.509 → 0.808, leak 0, L' 1,310 → 5,194, 개수 회수 0.049 → 0.193, 선택 손실 0.184 → 0.119
- 결과: `results/ablation_cap_time/frequent__fav/0000_400_0.csv`
| 01:55 | 감시 사건: TabDiff 21306 59작업 모두 COMPLETED(sacct 확인). TabDiff·ARF·smote 각 180칸, 실패 0 | 그림·표 재생성(sbatch 23218), easy_results_1027.md 8절·요약·그림 1 설명 갱신. 1시간 대비 점검 cron 삭제, 감시 스크립트 종료. 이 세팅의 baseline 실행 끝(ForestDiffusion 제외) |

## Chao1 상한 ablation 180칸 (job array 22752, 10/27 23:02 – 10/28 01:5x)
- `experiments/ablation_cap.sbatch` (brl, qos normal, CPU 4·2G·40분, %12), 180/180 COMPLETED, 실패·재제출 0, 작업당 6–13분
- 집계 `scripts/summarize_ablation.py` → `reports/ablation_cap_table.md`, `reports/ablation_cap_cells.csv`
- nocap − base: SZ 질량 회수 +0.23–0.36, 개수 회수 +0.08–0.14, leak 0 유지, 선택 손실 −0.04–0.07, uni_tv +0.004–0.015, pair_tv +0.004–0.018
- 검산: base recovery가 기존 결과와 174칸 일치, 6칸 최대 7.2e-4 차이(노드 수치 차이로 추정, 미확인)

## mid·rare 환경 (10/28 02:30–)
- 계획: `experiments/plan_mid_rare.md`. CPU 묶음(BOMiS, A/B/C, ROS, SMOTE-N, SMOTE-ENC, smote, ARF) + CTGAN, CPU 묶음 뒤 TabDDPM. TabDiff·TabSyn 제외
- 코드: `run_easy.py`·`run_dl.py`에 `--levels`, `env_list`, `out_dir` 추가. 연결 확인 srun 23252 통과
- 제출: mr_cpu 23253(0–359, 동시 6), mr_ctgan 23254(0–359, brl2 제외), mr_tabddpm 23255(0–119, afterany:23253), mr_throttle 23264(afterany:23254, mr_cpu 동시 8로)
- 상태 보기: `sacct -j 23253,23254,23255,23264 -X --format=JobID,JobName,State,Elapsed,NodeList -P`
- 02:40 gpu-monitoring 시작: 감시 스크립트(23253·23254·23255, 실패 glob mid_*·rare_*, 상태 ~/.cache/gpu-monitoring/bomis_midrare.txt) + 1시간 대비 점검 cron d3fda1b1(매시 17분)
- 02:57 사용자 요청으로 gpu-monitoring 중단(감시 스크립트·cron d3fda1b1 종료). job은 계속 실행
- 03:00 gpu-monitoring 재개: sacct 확인 결과 중단 사이 실패 0(mr_cpu 14 완료·6 실행, mr_ctgan 30 완료·4 실행). 감시 스크립트 재기동 + cron 27ce14f4(매시 17분)
- 06:22 대비 점검: mr_ctgan 23254 05:50 완료(360칸, 실패 0), mr_throttle 23264 완료 → mr_cpu 동시 실행 8. mr_cpu 136/360. mid_fav·mid_avg BOMiS 묶음 60칸씩 완료
- 06:22 감시 스크립트 버그 발견·수정: 상태 파일이 비어 있으면 awk `NR==FNR`가 stdin 줄에도 참이 되어 모든 사건을 "본 것"으로 처리(CTGAN array_done을 놓침). `FILENAME==ARGV[1]`로 고치고 빈 상태 파일로 23254 array_done 감지 확인. 감시 재기동(23253, 23255)
- 07:22 대비 점검: mr_cpu 182/360(동시 8), 실패 0. mid 3환경 CPU 묶음 완료. 중간 확인: mid $n_1=400$은 C 인증 0/4(90칸 모두) → BOMiS leak 0.053–0.058·control 0.915–0.917, $n_1=800$은 인증 4/4 → leak 0. 설계 3.4절 인증 경계 예측(mid $s$ = 0.71 / 1.41)과 같음
- 10:45 자원 재배치(사용자 요청 "효율적으로"): CTGAN 끝난 뒤 GPU 4장이 놀고 TabDDPM은 afterany:23253로 대기 중이었음. 20883 실측 TabDDPM은 CPU 평균 2.05개·MaxRSS 4.1G인데 6 CPU·24G 요청 → 23255 대기분을 CPU 4·8G로 줄이고(scontrol), 의존성 제거, mr_cpu 동시 8→4. CPU 16 + GPU job 4×4 = 32로 GPU 4장 모두 사용. 23255_0 brl3에서 5분 11초 완료·3칸 성공 확인
- 예상: mr_cpu 남은 ~27개 × 10분 / 4 → 12시 안팎, TabDDPM 119작업 × ~5분 / GPU 4 → 13:15 안팎(재배치 전 예상 14–15시)
- 10:48 사용자 승인으로 normal QOS: 23253·23255 대기분 QOS=normal, 상한 mr_cpu 12·TabDDPM 8. 직후 CPU 12개·TabDDPM 5개 실행
- 10:50 gpu-monitoring 스킬 확장: watch_slurm.sh에 진행 단계 사건(배열별 20·40·60·80%, `--milestones`), SKILL.md에 "3-1. 자원 재배치" 절. 감시 재기동(새 상태 파일, 이미 넘은 단계는 본 것으로)
- 10:59 사용자 출근으로 gpu-monitoring 중단(감시 스크립트·cron 5f98eff7 종료). job은 계속 실행
- 13:02 확인: mr_cpu 23253 360/360·mr_tabddpm 23255 120/120 COMPLETED(12:50 전후), 6환경 × 11방법 각 60칸, 실패 0. 감시 재기동 불필요
- 13:03–13:22 요약 job 24478(make_easy_report --level mid/rare, compare_counts --level mid/rare) → reports/{mid,rare}_report_tables.md·_counts.md·figures/{mid,rare}_fig*.png, 결과 문서 reports/midrare_results_1028.md. 핵심: 인증 경계 예측대로 mid n1=400·rare 인증 0 → BOMiS leak mid400 0.053–0.058, rare 0.013–0.020. rare에서 TabDDPM leak 0.022–0.024로 BOMiS와 비슷, recovery는 0.035–0.13 높음

## Chao1 상한 ablation mid·rare (10/28 13:53–)
- 계획 `experiments/plan_ablation_midrare.md`. 배열 24560 (`ablation_cap_midrare.sbatch`, 0–359%10, normal QOS, CPU 4·2G·40분). 제출 직후 10개 실행(brl0·3·4·5), 오류 없음
- 15:24 서버 장애: brl0·brl5(15:04:41), brl3(15:16:21), brl2(15:18:01), brl4(15:19:41) 차례로 Not responding → 계산 노드 전부 down(brl1은 10/24부터). 로그인 노드 hpc·slurmctld·NFS(/mnt/data1)는 정상, 노드는 ping 응답 없음. ablation 24560: 42칸 완료·저장, NODE_FAIL 7개(34·40·44–48) 자동 재대기. E3 24529_0–4도 NODE_FAIL 후 재대기. 원인은 로그인 노드에서 확인 불가(관리자 필요)
- 15:32 brl 전체 다운 대응(사용자 요청): hpc-stat1(partition1, 128 CPU·515G, 살아 있음)에서 bomis env 연결 확인(srun 24679 import 60초 초과로 시간 초과, 24688은 정상 — 첫 import 50초). 24560 대기분 Partition=brl,partition1로 바꿈 → 49–58번 10개 hpc-stat1에서 실행, BOMiS 학습 단계 진입 확인. 주의: hpc-stat1은 예전에 NFS D-state로 멈춘 적 있음
- 15:34 gpu-faster 재배치: 24560 대기분 partition1 전용·동시 24·시간 60분. hpc-stat1에서 24개 실행 확인(CPUAlloc 106/128)

### 2026-10-28 19시대 mid·rare Chao1 ablation 완료·집계
- 24560: 360/360 COMPLETED(hpc-stat1 318, brl 42). 집계 `scripts/summarize_ablation.py --level mid|rare`(옵션 추가, frequent 출력 이름은 그대로) → `reports/ablation_cap_{mid,rare}_{table.md,cells.csv}`. 집계 srun은 E3가 default_qos CPU 32를 다 써서 normal QOS(2 CPU, BOMiS ablation 승인 범위)로 돌림. rare는 10분 제한에 걸려 40분으로 다시 돌림.
- 검산: base recovery가 본실행 값과 다른 칸 mid 121, rare 132(최대 8.3e-3). 다른 칸은 전부 hpc-stat1(AMD EPYC 9354)·brl5에서 돈 칸이고, brl0·2·3·4(Intel Xeon Gold 6526Y)에서 돈 42칸 중 brl5 제외분은 모두 일치 → CPU 종류에 따른 수치 차이로 추정. base−nocap 짝 차이는 같은 실행 안의 비교라 영향 없음.
- 결과: nocap은 recovery를 +0.23–0.35 올림(mid·rare 모두 0.70–0.82, TabDDPM·ARF보다 높음). 대신 인증 실패 구간에서 leak 증가(mid n1=400: 0.053–0.058 → 0.073–0.077, rare: 0.013–0.020 → 0.024–0.047). 인증 성공 구간(mid n1=800)은 nocap도 leak 0. pair_tv는 +0.006–0.018 나빠짐.
