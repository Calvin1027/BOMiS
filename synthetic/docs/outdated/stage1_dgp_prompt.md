# 작업: BOMiS 1단계 합성 DGP와 정답 채점기 구현

## 목표

정답(SZ/STZ)을 모두 아는 8열×4수준 합성 모집단을 만든다. 환경은 18개다(STZ 축 3 × 겹침 축 3 × 가정 축 2). 그 위에 표본 추출, 정답 분류, 채점기 (a)–(d), 1단계 점검 실행기를 구현한다.

이 코드는 다음 단계에서 BOMiS의 support 추정(인증과 허용 규칙)을 시험하는 데 쓴다.

## 먼저 읽을 문서 (이 순서로, 끝까지)

1. **설계**: `/data1/calvinkim1027/BOMIS_new_수만이형/docs/superpowers/specs/2026-10-22-synthetic-dgp-design.md`
   - 무엇을 왜 만드는지 설명한다.
   - 값과 정의가 서로 어긋나면 이 문서가 기준이다.
2. **구현 계획**: `/data1/calvinkim1027/BOMIS_new_수만이형/docs/superpowers/plans/2026-10-22-synthetic-dgp.md`
   - 작업 8개, 파일 구성, 함수 이름과 시그니처, 테스트 42개가 들어 있다.
   - 이 순서대로 구현한다.
3. **구성 요소 명세서**: `/data1/calvinkim1027/BOMIS_new_수만이형/docs/algo_specs/stage1_dgp_spec.md`
   - C1–C20, D, Q, V 목록이 들어 있다.
   - 대응표를 채우는 기준이고, 자체 검토의 점검표다.

배경 논문(필요할 때만 참고): `/data1/calvinkim1027/BOMIS_new_수만이형/refs/draft_certifying_structural_zeros.pdf`

## 경로

| 용도 | 경로 |
|---|---|
| 코드 위치 | `/data1/calvinkim1027/BOMIS_new_수만이형/` (`dgp/`, `scripts/`, `tests/`, `experiments/`, `pytest.ini`) |
| 실행용 복사본 | `/mnt/data1/calvinkim1027/BOMIS_new_수만이형_runtime/` (`experiments/sync_runtime.sh`로 복사) |
| conda env | `/mnt/data1/calvinkim1027/miniconda3/envs/tabular_foundation` (Python 3.10, numpy 2.2.6, scipy 1.15.3, pandas 2.3.3, pytest 9.1.1) |

## 서버 제약

- **Python 실행**
  - srun/sbatch로만 돌린다. 로그인 노드에서 `python`을 직접 부르면 막힌다.
  - 계산 노드에서는 `/mnt/data1`만 보이고 `/data1`은 안 보인다. 테스트 전에 항상 `bash experiments/sync_runtime.sh`를 실행한다.
- **테스트 실행 명령**:
  `srun --partition=brl --ntasks=1 --qos=normal --exclude=brl1,brl2 --mem=8G --time=00:30:00 --chdir=/mnt/data1/calvinkim1027/BOMIS_new_수만이형_runtime /mnt/data1/calvinkim1027/miniconda3/envs/tabular_foundation/bin/python -I -m pytest <대상> -v`
- **`-I` 모드**: 스크립트 폴더가 `sys.path`에 들어가지 않는다. 진입 스크립트와 pytest 처리는 계획 문서의 Global Constraints를 따른다.
- **설치와 인터넷**: 계산 노드는 인터넷이 안 된다. 패키지 설치가 필요하면 멈추고 보고한다.
- **기타**
  - git 저장소가 아니므로 commit하지 않는다.
  - GPU를 쓰지 않는다.
- **실험 본실행 금지**: 예비 실행(`dgp_pilot.sbatch`)과 점검 실행(`dgp_check.sbatch`)은 돌리지 않는다. 테스트(`pytest`, `--quick` 포함)만 돌린다.

## 규칙

1. 계획 문서의 작업 1–8을 순서대로 한다. 작업마다 테스트를 먼저 쓰고, 실패를 확인하고, 구현하고, 통과를 확인한다.
2. 명세서의 모든 C와 D를 구현한다. 함수마다 담당 ID를 주석으로 단다(예: `# C7, D7.1: 설계 3.3절`).
3. 명세서와 계획에 없는 기능, 최적화, 기본값을 넣지 않는다. 필요해 보이면 구현하지 말고 "제안"으로 보고한다.
4. 모호점 Q1–Q5는 적힌 결정대로 구현한다. 문서끼리 어긋나거나 새 모호점을 만나면 멈추고 보고한다.
5. 라이브러리 기본값에 기대는 인자는 명시적으로 적는다. `np.random.default_rng(np.random.SeedSequence(...))`, `searchsorted(side="right")`, `binom.sf` 인자, `beta.ppf` 인자가 해당한다.
6. 진입점(`scripts/run_check.py`, `scripts/run_pilot.py`)에서 모든 C가 실제로 호출되는지 확인한다.
7. 계획의 테스트 42개를 계획에 적힌 대로 만든다. 기대값을 코드 출력에 맞춰 바꾸지 않는다. 테스트가 실패하면 실패한 채로 보고하고, 테스트가 틀렸다고 보면 그 이유를 적는다.
8. 성능 목표: `move_mass` 한 번 1초 이내. 못 맞추면 측정값을 보고한다.
9. 지정된 폴더 밖의 파일은 고치지 않는다. 대상은 `dgp/`, `scripts/`, `tests/`, `experiments/`, `pytest.ini`, `reports/stage1_codex_review.md`이다.

## 자체 검토 (2회, 각도를 다르게)

**1회차: 누락과 추가**

- 명세서의 C1–C20과 모든 D를 하나씩 코드에서 찾는다.
- 진입점부터 호출을 따라가서 모든 C가 실행 경로에 있는지 확인한다.
- 거꾸로, 코드에서 하는 처리 중 명세서에 없는 것을 찾는다. 임의의 클리핑, 정규화, 하드코딩된 상수가 해당한다.
- 계획 문서 끝의 "1차: spec 대조" 항목을 포함한다.

**2회차: 세부**

- 각 D를 해당 줄과 대조한다. 합 범위, 분모, 갱신 순서, 축, 인덱스, 라이브러리 기본값, 난수 위치를 본다.
- 작은 입력으로 손계산과 코드 결과를 비교한다(명세서 V 참고).
- 명세서 6절의 결과 민감 항목(D7.1, D4.2, D5.2, D13.2, D14.1, D18.2)을 반드시 본다.
- 계획 문서 끝의 "2차: 수치 경계" 항목을 포함한다.

검토마다 찾은 문제와 고친 내용을 `reports/stage1_codex_review.md`에 적는다.

## 산출물

1. 코드와 테스트(계획의 파일 구성대로)
2. 명세서 4절 형식으로 채운 대응표: 모든 C와 D → `파일:함수:줄`. `reports/stage1_codex_review.md`에 넣는다
3. 대응 못 한 항목, 또는 명세서와 다르게 구현한 항목과 각각의 이유
4. 전체 테스트 결과(srun 출력 그대로)
5. 자체 검토 2회에서 찾고 고친 것
6. 문서가 모호하거나 틀렸다고 보는 곳
7. `move_mass` 실행 시간 측정값
