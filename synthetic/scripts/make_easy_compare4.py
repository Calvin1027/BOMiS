"""쉬운 환경 3개 × n1 2개에서 생성 방법 4개(BOMiS, CTGAN, SMOTE-N, SMOTE-ENC)를 지표 4개로 비교한다.

지표: SZ 개수 회수(rec_count), SZ 질량 회수(recovery), STZ leak(leak), control.
값은 칸 평균, ± 는 MCSE = 모집단별 평균의 표준편차 / √모집단 수 (make_easy_report.py와 같은 정의).
출력: 표(md) 하나, 그림 둘.
  fig4 = 점 그림. 지표마다 칸 하나, 세로축 = (환경, n1), 방법마다 점 + MCSE 막대.
  fig5 = 질량 회수 대 leak. (환경, n1)마다 칸 하나, 왼쪽 위가 좋은 쪽.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ENVS = {"frequent__fav": ("fav", "results/easy"), "frequent__avg": ("avg", "results/easy_avg"),
        "frequent__unf": ("unf", "results/easy_unf")}
# BOMiS와 CTGAN을 진한 색으로, SMOTE 두 개는 서로 비슷한 옅은 색으로 둔다
METHODS = [("bomis", "BOMiS", "#1f6fd1", "o"), ("ctgan", "CTGAN", "#e8622c", "s"),
           ("smoten", "SMOTE-N", "#9a8fc9", "v"), ("smote_enc", "SMOTE-ENC", "#5fa88a", "D")]
METRICS = [("rec_count", "SZ count recovery", "↑"), ("recovery", "SZ mass recovery", "↑"),
           ("leak", "STZ leak", "↓"), ("control", "control", "↑")]
N1S = (400, 800)
INK, SUB, MUTED, GRID, AXIS, BG, BAND = "#16161a", "#4a4a52", "#8a8a92", "#e6e6ea", "#c9c9d0", "#ffffff", "#f4f5f8"


def load(root: Path) -> pd.DataFrame:
    frames = []
    for _, sub in ENVS.values():
        for m, *_ in METHODS:
            frames += [pd.read_csv(p) for p in (root / sub / m).glob("*.csv")]
    return pd.concat(frames, ignore_index=True)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (env, m, n1), g in df.groupby(["env", "method", "n1"]):
        r = {"env": env, "method": m, "n1": n1, "cells": len(g)}
        by_pop = g.groupby("pop")[[c for c, *_ in METRICS]].mean()
        for c, *_ in METRICS:
            r[c] = g[c].mean()
            r[c + "_mcse"] = by_pop[c].std(ddof=1) / np.sqrt(len(by_pop))
        rows.append(r)
    return pd.DataFrame(rows).set_index(["env", "method", "n1"])


def md(t: pd.DataFrame) -> str:
    head = "| 환경 | $n_1$ | 방법 | 칸 | " + " | ".join(n for _, n, _ in METRICS) + " |"
    lines = [head, "|" + "---|" * (4 + len(METRICS))]
    for env, (short, _) in ENVS.items():
        for n1 in N1S:
            for m, label, *_ in METHODS:
                r = t.loc[(env, m, n1)]
                vals = [f"{r[c]:.3f} ± {r[c + '_mcse']:.3f}" for c, *_ in METRICS]
                lines.append(f"| {short} | {n1} | {label} | {int(r['cells'])} | " + " | ".join(vals) + " |")
    return "\n".join(lines)


def base(ax):
    ax.set_facecolor(BG)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=SUB, labelsize=9, length=0)


def mstyle(m, color, size):
    """SMOTE-N은 속 빈 표식으로 그려 SMOTE-ENC와 겹쳐도 둘 다 보이게 한다."""
    if m == "smoten":
        return {"markersize": size, "markerfacecolor": "none", "markeredgecolor": color, "markeredgewidth": 1.8}
    return {"markersize": size, "markerfacecolor": color, "markeredgecolor": BG, "markeredgewidth": 1.0}


def legend(fig, y):
    h = [Line2D([], [], color=c, marker=mk, linestyle="", **mstyle(m, c, 8)) for m, _, c, mk in METHODS]
    fig.legend(h, [lab for _, lab, *_ in METHODS], loc="upper center", ncol=len(METHODS), frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, y), labelcolor=SUB, handletextpad=0.3, columnspacing=1.6)


def fig_dots(t: pd.DataFrame, out: Path):
    combos = [(env, n1) for env in ENVS for n1 in N1S]
    ys = np.array([i + (i // 2) * 0.6 for i in range(len(combos))])  # 환경 사이에 틈
    offs = np.linspace(-0.27, 0.27, len(METHODS))
    fig, axes = plt.subplots(1, len(METRICS), figsize=(14, 6.4), sharey=True)
    fig.patch.set_facecolor(BG)
    for ax, (col, name, arrow) in zip(axes, METRICS):
        base(ax)
        for k in range(len(ENVS)):  # 환경마다 옅은 띠
            if k % 2 == 0:
                ax.axhspan(ys[2 * k] - 0.5, ys[2 * k + 1] + 0.5, color=BAND, zorder=0, linewidth=0)
        top = max(t[col] + t[col + "_mcse"].fillna(0))
        for (m, _, color, mk), off in zip(METHODS, offs):
            v = np.array([t.loc[(e, m, n1), col] for e, n1 in combos])
            e = np.array([t.loc[(e, m, n1), col + "_mcse"] for e, n1 in combos])
            ax.errorbar(v, ys + off, xerr=e, fmt=mk, color=color, elinewidth=1.4, capsize=0, zorder=3,
                        **mstyle(m, color, 7.5 if m == "bomis" else 6.5))
            if m in ("bomis", "ctgan"):  # 주인공 두 방법만 숫자를 단다. 오차 막대 오른쪽 끝 뒤에 둔다
                lo = 0.6 if col == "control" else 0.0
                for x, err, y in zip(v, e, ys + off):
                    ax.text(x + err + (top - lo) * 0.03, y, f"{x:.3f}", va="center", fontsize=7.8, color=color,
                            zorder=4)
        if col == "control":  # 값이 0.67–1.0에 모여 있어 0.6부터 그린다
            ax.set_xlim(0.6, 1.1)
        else:
            ax.set_xlim(-top * 0.03, top * 1.25)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_title(f"{name}  {arrow}", fontsize=11.5, color=INK, loc="left", pad=10, fontweight="bold")
    axes[0].set_yticks(ys, [f"{ENVS[e][0]}   n1={n1}" for e, n1 in combos], fontsize=10)
    axes[0].invert_yaxis()
    fig.suptitle("Easy environments (frequent STZ): generated minority rows, mean of 30 cells ± MCSE",
                 x=0.01, ha="left", fontsize=12.5, color=INK, y=1.05)
    legend(fig, 1.01)
    fig.tight_layout(rect=(0, 0, 1, 0.97), w_pad=2.0)
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def fig_tradeoff(t: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(len(N1S), len(ENVS), figsize=(12.5, 7.6), sharex=True, sharey=True)
    fig.patch.set_facecolor(BG)
    xmax = max(t["leak"] + t["leak_mcse"]) * 1.15
    ymax = max(t["recovery"] + t["recovery_mcse"]) * 1.18
    for r, n1 in enumerate(N1S):
        for c, env in enumerate(ENVS):
            ax = axes[r, c]
            base(ax)
            ax.spines["left"].set_visible(True)
            ax.spines["left"].set_color(AXIS)
            ax.grid(color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)
            ax.axvspan(-1, xmax * 0.06, color="#eaf2fc", zorder=0, linewidth=0)  # leak ≈ 0 띠
            for m, label, color, mk in METHODS:
                row = t.loc[(env, m, n1)]
                ax.errorbar(row["leak"], row["recovery"], xerr=row["leak_mcse"], yerr=row["recovery_mcse"],
                            fmt=mk, color=color, elinewidth=1.4, capsize=0,
                            zorder=5 if m == "smoten" else 3, **mstyle(m, color, 11 if m == "bomis" else 9))
                if m in ("bomis", "ctgan"):
                    ax.annotate(f"{row['recovery']:.3f}", (row["leak"], row["recovery"]), xytext=(9, 6),
                                textcoords="offset points", fontsize=8.5, color=color, fontweight="bold")
            ax.set_xlim(-xmax * 0.03, xmax)
            ax.set_ylim(0, ymax)
            ax.set_title(f"{ENVS[env][0]},  n1 = {n1}", fontsize=11, color=INK, loc="left", fontweight="bold")
            if r == len(N1S) - 1:
                ax.set_xlabel("STZ leak  (← lower better)", fontsize=9.5, color=SUB)
            if c == 0:
                ax.set_ylabel("SZ mass recovery  (higher better ↑)", fontsize=9.5, color=SUB)
    fig.suptitle("Recovery vs leak: the upper-left corner is better  (mean of 30 cells ± MCSE)",
                 x=0.01, ha="left", fontsize=12.5, color=INK, y=1.04)
    legend(fig, 1.005)
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=2.2, w_pad=1.6)
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--fig-dir", default="reports/figures")
    ap.add_argument("--table", default="reports/easy_compare4_table.md")
    args = ap.parse_args()
    t = summarize(load(Path(args.root)))
    figs = Path(args.fig_dir)
    figs.mkdir(parents=True, exist_ok=True)
    fig_dots(t, figs / "easy_fig4_compare4.png")
    fig_tradeoff(t, figs / "easy_fig5_tradeoff.png")
    Path(args.table).write_text("# 생성 방법 4개 × 지표 4개 (scripts/make_easy_compare4.py)\n\n" + md(t) + "\n")
    print("ok")


if __name__ == "__main__":
    main()
