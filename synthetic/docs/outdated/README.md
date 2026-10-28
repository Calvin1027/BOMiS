# 지난 문서 (쓰지 않음)

1단계 합성 실험의 v1 문서들이다. 2026-10-26에 `docs/superpowers/specs/2026-10-26-synthetic-study-ademp-design.md`(v2)로 대체되어 여기로 옮겼다. 기록용으로만 둔다.

| 파일 | 원래 위치 | 내용 |
|---|---|---|
| `2026-10-22-synthetic-dgp-design.md` | `docs/superpowers/specs/` | v1 설계 (금지 규칙을 무작위 표에서 찾는 방식) |
| `2026-10-22-synthetic-dgp.md` | `docs/superpowers/plans/` | v1 구현 계획 (작업 8개, 테스트 42개) |
| `stage1_dgp_spec.md` | `docs/algo_specs/` | v1 구성 요소 명세서 |
| `stage1_dgp_prompt.md` | `docs/codex_prompts/` | v1 Codex 구현 프롬프트 |
| `stage1_codex_review.md` | `reports/` | v1 기준 Codex 부분 구현 보고 (Task 1–2 통과, Task 3에서 멈춤) |

v2 승인 뒤 계획, 명세서, 프롬프트는 v2 기준으로 새로 쓴다.

**10/27 추가**

| 파일 | 내용 |
|---|---|
| `2026-10-26-synthetic-study-ademp-design_18env.md` | v2 설계의 18환경 판(10/27 사본). local support 가정 위반 환경(3변수 금지 규칙, $e^T \le 0.2$, $q \le 0.36$)이 들어 있다. 10/27에 가정 축을 빼고 9환경으로 줄였다. 리뷰에서 위반 환경을 요청하면 이 판의 4.3·4.5·4.6절을 되살린다. 측정 스크립트는 `/mnt/data1/calvinkim1027/scratch_bomis/accept_variants.py`(V2)다. |
