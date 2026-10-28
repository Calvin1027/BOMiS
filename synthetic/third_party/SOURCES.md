# 외부 baseline 코드 출처

합성 실험(2026-10-27–28)에서 실제로 쓴 코드를 그대로 복사했다. `.git` 폴더와 캐시만 뺐다. 이 폴더의 코드는 고치지 않는다. 고쳐야 하면 출처와 이유를 이 문서에 적는다.

| 폴더 | 출처 | 커밋·버전 | 라이선스 | 비고 |
|---|---|---|---|---|
| `categorical_baselines/` (`src`, `configs`, `pyproject.toml`, `README.md`) | 연구실 코드 `과거 BOMIS/TDG_baseline_categorical` | 10/28 작업 폴더 그대로 | 연구실 내부 | SMOTE-N, SMOTE-ENC, smote, ARF, CTGAN, TabSyn, TabDiff 실행기(CLI) |
| `categorical_baselines/third_party/arfpy` | https://github.com/bips-hb/arfpy | `8b63c1b` | MIT | 수정 없음 |
| `categorical_baselines/third_party/tabdiff` | https://github.com/MinkaiXu/TabDiff | `5ecdb33` | MIT | 수정 없음 |
| `categorical_baselines/third_party/tabsyn` | https://github.com/amazon-science/tabsyn | `cb5ac0f` | Apache-2.0 | 수정 없음 |
| `baseline_exp_v4/src` | 연구실 코드 `TDG/Baseline_exp_v4/src` (런타임 사본 `baseline_exp_runtime_v4_32g_20260910b`) | 9/10 런타임 사본 그대로 | 연구실 내부 | TabDDPM·ForestDiffusion 어댑터(`baselines/tabddpm_cli.py`, `forestflow_cli.py`가 import) |
| `tab-ddpm` | https://github.com/yandex-research/tab-ddpm | `b476257` + **연구실 로컬 수정 4파일** | MIT (`LICENSE.md`) | 수정 내용: `patches/tab-ddpm_local_changes.diff` (`lib/data.py`, `scripts/pipeline.py`, `sample.py`, `train.py`). 실행 때 `baseline_exp_v4`가 기울기 자르기 패치를 추가로 적용한다 |
| `ForestDiffusion` | https://github.com/SamsungSAILMontreal/ForestDiffusion | `818ac3b` | 복사본에 라이선스 파일 없음(원 저장소 확인 필요) | 실행 때 `patches/forest_flow_target_label.patch`를 적용한다. 이번 실험에서는 시간 때문에 돌리지 않았다 |

`categorical_baselines`의 원래 `third_party`에 있던 STaSy, TabNAT은 이 실험에서 쓰지 않고 라이선스 파일도 없어서 넣지 않았다.

## 위치 관계

- `categorical_baselines`의 어댑터는 `src/categorical_baselines/adapters/vendor.py`에서 `<categorical_baselines>/third_party`를 상대 경로로 찾는다. 그래서 `src`와 `third_party`를 같은 폴더 아래에 둔다.
- 경로는 `config/paths.sh`의 `BL_SRC`, `BL_V4_SRC`, `BL_TABDDPM_SOURCE`, `BL_FF_SOURCE`, `BL_FF_PATCH`로 넘긴다(`config/paths.example.sh` 참고).
