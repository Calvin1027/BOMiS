# 제출 계획: 쉬운 환경 미리보기 (2026-10-27 02:56 조회)

## 현황
- brl1 down. 빈 CPU brl0 32, brl5 56, brl4 12, brl2 8, brl3 0. 빈 GPU brl2 5, brl5 4, brl0 1, brl3 6(CPU 없음). 노드별 빈 메모리 약 28–48G.
- default_qos 사용자당 GPU 4, CPU 32, 메모리 128G. 내 다른 job 없음.

## 작업
- 60칸 = 모집단 3개(seed 0, 1, 2) × $n_1$(400, 800) × 표본 seed 0–9, 환경 `frequent__fav`. 칸 순서는 $n_1$=400 먼저.
- CPU 그룹(array 0–59): A/B/C support, BOMiS 전체(기본 설정, CPU), ROS, SMOTE-N, SMOTE-ENC.
- GPU 그룹(array 0–59): CTGAN.

## 산정 근거
- BOMiS 기본 fit 약 585초(CPU 4, 10/26 측정). SMOTE-N 218초, SMOTE-ENC 61초(10/27 스모크 20180, 실제 크기). 칸당 약 16분 → 시간 40분(×2.5, 측정 불확실).
- CTGAN 548초, MaxRSS 2.0G, TotalCPU 2:17(평균 0.25코어)(스모크 20181, A5000) → GPU 1, CPU 2, 메모리 6G, 시간 30분.
- CPU 그룹 메모리: SMOTE 스모크 MaxRSS 0.33G, BOMiS 미측정 → 10G.

## 전략
| 전략 | 동시 실행 | 예상 완료 |
|---|---|---|
| A. CPU job 6개(4코어, 10G) + GPU job 4개(2코어, 6G) | CPU 32/32, 메모리 84G | CPU 60칸/6 × 16분 ≈ 2.7시간, GPU 60칸/4 × 11분 ≈ 2.8시간 |
| B. CPU job 8개 + GPU job 0개 먼저 | CPU 32 | CPU 2시간, 그 뒤 GPU 2.8시간 → 4.8시간 |

A를 고른다. 두 그룹이 같이 끝난다. QOS가 동시 실행을 막으므로 array 상한은 두지 않는다.

## GPU/CPU 선택
- BOMiS는 박사형 프로토콜이 CPU(README "The experiments run on CPU")라 장치를 바꾸지 않는다. SMOTE 계열은 sklearn이라 CPU.
- CTGAN은 torch라 GPU. 기존 baseline 프로젝트도 GPU로 돌렸다.
