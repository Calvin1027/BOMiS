"""BOMiS와 DL 계열(CTGAN, TabDDPM, TabSyn, TabDiff)의 생성 지표를 STZ 수준별 표로 만든다. 값은 칸 30개(모집단 3 × 표본 seed 10)의 평균 ± SD.
사용: table_dl_vs_bomis.py (결과: reports/dl_vs_bomis_sd.md)
"""
from pathlib import Path

import pandas as pd

LEVELS = {"frequent": {"fav": "results/easy", "avg": "results/easy_avg", "unf": "results/easy_unf"},
          "mid": {ov: f"results/mid_{ov}" for ov in ("fav", "avg", "unf")},
          "rare": {ov: f"results/rare_{ov}" for ov in ("fav", "avg", "unf")}}
METHODS = {"bomis": "BOMiS", "tabsyn": "TabSyn", "tabdiff": "TabDiff", "tabddpm": "TabDDPM", "ctgan": "CTGAN"}
COLS = [("rec_count", "SZ 개수 회수"), ("rec_mass", "확률 질량 회수"), ("recovery", "recovery"), ("leak", "leak"), ("control", "control")]


def main():
    out = ["# BOMiS vs DL 계열 생성 지표 (scripts/table_dl_vs_bomis.py)", "",
           "값은 (환경, $n_1$)마다 칸 30개(모집단 3 × 표본 seed 10)의 평균 ± SD(ddof=1).", ""]
    gap = 0.0
    for lv, outs in LEVELS.items():
        frames = [pd.read_csv(f) for ov, d in outs.items() for m in METHODS for f in sorted(Path(d, m).glob("*.csv"))]
        df = pd.concat(frames, ignore_index=True)
        df["ov"] = df["env"].str.split("__").str[1]
        gap = max(gap, (df["recovery"] - df["rec_mass"]).abs().max())
        out += [f"## {lv}", "", "| 환경 | $n_1$ | 방법 | 칸 | " + " | ".join(n for _, n in COLS) + " |", "|" + "---|" * (4 + len(COLS))]
        for ov in outs:
            for n1 in (400, 800):
                for m, name in METHODS.items():
                    g = df[(df.ov == ov) & (df.n1 == n1) & (df.method == m)]
                    if not len(g):
                        continue
                    vals = [f"{g[c].mean():.3f} ± {g[c].std(ddof=1):.3f}" for c, _ in COLS]
                    out.append(f"| {ov} | {n1} | {name} | {len(g)} | " + " | ".join(vals) + " |")
        out.append("")
    out.append(f"recovery와 rec_mass의 칸별 최대 차이: {gap:.2e}")
    Path("reports/dl_vs_bomis_sd.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
