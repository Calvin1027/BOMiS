"""C28 요약: 칸별 csv를 모아 (방법, n1)별 평균과 MCSE(모집단 군집), BOMiS 짝 차이, 회수율 분해 표를 만든다."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

GEN = ["bomis", "ros", "smoten", "smote_enc", "ctgan", "tabsyn", "tabddpm"]
SUP = ["A", "B", "C"]
GEN_COLS = ["recovery", "leak", "control", "novel", "uni_tv", "pair_tv", "auroc", "rec_count"]
SUP_COLS = ["sz_rec_mass", "sz_rec_count", "stz_sup_count", "stz_sup_maj", "all_cert", "n_false", "S_size"]
DIAG_COLS = ["diag_R1", "diag_R2", "diag_R4", "diag_loss_support", "diag_loss_count", "diag_loss_select",
             "diag_M_hat", "diag_M0", "diag_M_err", "diag_f0", "diag_L_prime", "diag_n_sz",
             "diag_Q_sz", "diag_Q_stz", "diag_Qu_stz", "diag_Q_tv", "diag_novel_tv", "diag_shell_tv",
             "diag_ell_err", "diag_steps"]


def mean_mcse(df, cols):
    # D28.3: 칸 평균, MCSE = 모집단별 평균의 표준편차 / sqrt(모집단 수).
    rows = []
    for (m, n1), g in df.groupby(["method", "n1"], sort=False):
        r = {"method": m, "n1": n1, "cells": len(g)}
        by_pop = g.groupby("pop")[cols].mean()
        for c in cols:
            r[c] = g[c].mean()
            r[c + "_mcse"] = by_pop[c].std(ddof=1) / np.sqrt(len(by_pop)) if len(by_pop) > 1 else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def fmt(table, cols):
    head = "| 방법 | $n_1$ | 칸 | " + " | ".join(cols) + " |"
    sep = "|" + "---|" * (3 + len(cols))
    lines = [head, sep]
    for _, r in table.iterrows():
        vals = [f"{r[c]:.3f} ± {r[c + '_mcse']:.3f}" if pd.notna(r[c + "_mcse"]) else f"{r[c]:.3f}" for c in cols]
        lines.append(f"| {r['method']} | {int(r['n1'])} | {int(r['cells'])} | " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="results/easy")
    ap.add_argument("--out", default="reports/easy_preview.md")
    args = ap.parse_args()
    root = Path(args.inp)
    frames = [pd.read_csv(p) for p in root.glob("*/*.csv") if p.parent.name != "_failed"]
    df = pd.concat(frames, ignore_index=True)
    fails = sorted(p.parent.name + "/" + p.name for p in (root / "_failed").glob("*/*.csv")) if (root / "_failed").exists() else []
    order = {m: i for i, m in enumerate(SUP + GEN)}
    df = df.sort_values(["method", "n1"], key=lambda s: s.map(order) if s.name == "method" else s)
    gen = df[df.method.isin(GEN)]
    sup = df[df.method.isin(SUP)]
    bom = df[df.method == "bomis"]
    out = [f"# 쉬운 환경 미리보기 ({', '.join(sorted(df.env.unique()))})", "",
           f"칸 수: 방법별 {df.groupby('method').size().to_dict()}. 실패: {len(fails)}개 {fails[:10]}", "",
           "값은 칸 평균 ± MCSE(모집단별 평균의 표준편차 / √모집단 수).", ""]
    if len(gen):
        out += ["## 생성 행 지표 (박사형 evaluate + 개수 회수율)", "", fmt(mean_mcse(gen, GEN_COLS), GEN_COLS), ""]
        key = ["pop", "n1", "seed"]
        b = bom.set_index(key)[["recovery", "leak", "control"]]
        diffs = []
        for m in GEN[1:]:
            o = df[df.method == m].set_index(key)[["recovery", "leak", "control"]]
            d = (b - o).dropna().reset_index()
            if len(d):
                d["method"] = f"bomis − {m}"
                diffs.append(d)
        if diffs:
            dd = pd.concat(diffs)
            out += ["## BOMiS − baseline 짝 차이", "", fmt(mean_mcse(dd, ["recovery", "leak", "control"]), ["recovery", "leak", "control"]), ""]
    if len(sup):
        out += ["## support 단계 A/B/C", "", fmt(mean_mcse(sup, SUP_COLS), SUP_COLS), ""]
    if len(bom):
        cols = [c for c in DIAG_COLS if c in bom]
        out += ["## BOMiS 단계별 진단 (7.2절 (e))", "", fmt(mean_mcse(bom, cols), cols), ""]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(out))
    df.to_csv(Path(args.out).with_suffix(".csv"), index=False)
    print("\n".join(out))


if __name__ == "__main__":
    main()
