# Chao1 상한 ablation mid·rare 확장 계획 (10/28 13:50)

- 범위(사용자 승인 10/28 13:50): frequent ablation(22752)과 같은 base/nocap 비교를 mid·rare 6환경 × 모집단 3 × n1 2 × seed 10 = 360칸에. normal QOS, 동시 10개
- 코드: `scripts/run_ablation.py`의 ENV_OUT에 mid·rare 결과 폴더만 추가(검산용 기존 BOMiS recovery 읽기). 원저자 코드 수정 없음
- 칸 목록 `experiments/ablation_cells_midrare.txt`: 0번 = 기존 BOMiS 시간 중간 칸(rare__unf 1 400 3), 나머지는 기존 BOMiS 시간 긴 순
- 자원: CPU 4·2G·40분 (frequent ablation 실측 평균 7.8분·최장 12.8분·MaxRSS 0.58G, mid·rare BOMiS 최장 칸 1001초). GPU 안 씀(BOMiS CPU 코드)
- 서버(13:50): 빈 CPU 약 80개, GPU 거의 다 참. 개인연구 E3가 default_qos 한도(CPU 32) 사용 중 → normal QOS로 병행
- 결과: `results/ablation_cap/{mid,rare}__{fav,avg,unf}/<pop>_<n1>_<seed>.csv`
- 예상(보장 아님): 360칸 × 약 8분 / 10 ≈ 5시간 → 19시 안팎
- 상태 보기: `sacct -j 24560 -X --format=JobID,State,Elapsed,NodeList -P`

## 15:34 재배치: brl 전체 다운 → hpc-stat1(partition1) 전용 (사용자 승인)
- 서버(15:33): brl0–5 모두 down(Not responding). hpc-stat1 128 논리 CPU(64코어 × 2)·515G, 다른 사용자 CPU 8·GPU 4 사용, FreeMem 78G. partition1 대기: 다른 사용자 7개(2 CPU·18G, Dependency로 대기)
- 실측: frequent ablation은 CPU 4개 중 평균 3.66개 사용(효율 높음 → 작업당 CPU 그대로), MaxRSS 0.47–0.58G. mid·rare 완료 42칸 평균 16.6분·최장 33.6분
- 전략 비교 (남은 318개, 작업당 15–25분 가정)

| 안 | 동시 실행 | CPU 사용(우리) | 예상 완료 | 비고 |
|---|---|---|---|---|
| 그대로 | 10 | 40 | 약 8시간 | 노드 여유 많이 남김 |
| **선택** | **24** | **96** | **약 3.5–5.5시간** | 다른 사용자 대기분(CPU 14·126G) 자리 남김 |
| 최대 | 30 | 120 | 약 3–4.5시간 | 노드 거의 독점, 하이퍼스레드 경합으로 작업당 시간 늘 위험 |

- 적용: 대기분 Partition=partition1, ArrayTaskThrottle=24, TimeLimit 40분→60분(mid·rare 최장 33.6분이라 여유 부족). 실행 중이던 49–58번은 40분 그대로(실행 중 작업은 늘릴 권한 없음)
- 상태 보기: `sacct -j 24560 -X --format=JobID,State,Elapsed,NodeList -P`
