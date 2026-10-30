# BOMiS 합성 데이터 실험 (`synthetic/`)

박사형 초안 "Certifying Structural Zeros for Categorical Minority Oversampling"의 방법(BOMiS)을, 정답 support를 아는 합성 모집단에서 단계별로 평가하는 실험 코드다. 방법 코드는 원저자 저장소 `github.com/suman-cha/bomis` 커밋 `fa088a9`를 수정 없이 `vendor/`에 고정해 쓴다.

## 실험 개요

- **데이터**: 8열 × 4수준(셀 65,536개). 부모 4개·잎 4개 트리에서 majority 분포 $p_0$를 만들고, 금지 쌍을 심은 뒤 질량 이동과 섞기로 minority 분포 $p_1$을 만든다. floor 가정($\pi_1 \ge \pi_0/2$)과 local support 가정이 구성으로 성립한다.
- **환경 축**: STZ 축(금지 쌍 $\pi_0$ = frequent 0.06 / mid 0.03 / rare 0.005) × 겹침 축(Z 질량 fav 0 / avg 0.1 / unf 0.2) = 9환경
- **표본**: $n_0 = 8000$, $n_1 \in \{400, 800\}$ (설계는 50–800), 개발용 모집단 3개 × 표본 seed 10개
- **방법**: support 단계 A(minority만) / B(합침, 인증 없음) / C(인증, BOMiS 기본), BOMiS 전체, baseline(ROS, SMOTE-N, SMOTE-ENC, smote, ARF, CTGAN, TabDDPM, TabSyn, TabDiff)
- **지표**: 박사형 `evaluate`(recovery, leak, control, uni_tv, pair_tv, auroc)와 진단 지표(SZ 회수·STZ 억제, 인증 검정력·FWER, BOMiS 회수율 분해 $R_1 \ge R_2 \ge R_4$)

## 먼저 읽을 문서 (경로는 `synthetic/` 기준)

1. 설계: `docs/superpowers/specs/2026-10-26-synthetic-study-ademp-design.md`
2. 처음 보는 사람용 통합 정리: `reports/easy_synthetic_integrated_1027.md`
3. 결과: `reports/easy_results_1027.md`(frequent), `reports/midrare_results_1028.md`(mid·rare), `reports/ablation_cap_*table.md`(Chao1 상한 ablation)
4. 발표 자료: `발표/BOMiS_단계별_실험.pdf`

## 폴더

| 폴더 | 내용 |
|---|---|
| `dgp/` | 셀 공간, 분포 생성, 표본, 정답, 채점, 박사형 코드 연결(`harness.py`), BOMiS 단계별 진단 |
| `vendor/bomis_fa088a9/` | 원저자 코드 고정 사본(MIT), `SHA256SUMS`로 무결성 확인 |
| `baselines/` | ROS와 외부 baseline CLI 래퍼 |
| `scripts/` | 모집단 생성, 칸 실행, 집계·그림 |
| `experiments/` | sbatch 파일, 실행 계획, 실행 기록(`run_log_easy.md`) |
| `tests/` | pytest |
| `reports/` | 결과 문서, 표, 그림 |
| `docs/` | 설계·계획·명세. `docs/outdated/`는 폐기된 v1 |

## 실행 환경

- Python 3.11, numpy 2.4.6, scipy 1.17.1, pandas 2.3.3, scikit-learn 1.9.1, torch 2.5.1 (원저자 `requirements.txt`와 같음), 테스트용 pytest
- 원래 서버는 Slurm 전용이라 모든 실행이 sbatch/srun이다. 코드를 런타임 폴더로 복사(`experiments/sync_runtime.sh`)한 뒤 거기서 실행한다.

기본 흐름 (런타임 폴더 기준):

```bash
python scripts/build_pops.py --start 0 --n 10 --out results/pops   # 모집단 생성
python -m pytest                                                    # 테스트
python scripts/run_easy.py --group cpu --task-id 0                  # 칸 하나 (A/B/C, BOMiS, ROS, SMOTE)
```

## 환경 설치와 경로

- 설치 방법: [`envs/README.md`](envs/README.md). env 두 개와 덧댄 폴더 3개를 패키지 버전까지 원래와 같게 만들고, 원래 결과와 비교한 내용도 거기에 적혀 있다.
- 경로: `config/paths.sh`에서 읽는다. `config/paths.example.sh`를 복사해서 만들고, 이 파일은 git에 올리지 않는다.
- 외부 baseline 코드: `third_party/` (출처와 라이선스는 `third_party/SOURCES.md`).
- `experiments/*.sbatch`의 Slurm 파티션은 brl 서버 기준이다.
