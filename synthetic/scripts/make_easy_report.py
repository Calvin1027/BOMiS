"""STZ 수준 하나의 3환경(기본 frequent__fav/avg/unf, --level mid·rare) 결과를 모아 그림 3개와 표(md)를 만든다. 다시 돌리면 그 시점 결과로 갱신된다.

pandas·matplotlib만 쓴다(bomis env에는 matplotlib이 없어 tabular_foundation env로 돈다).
값은 칸 평균, ± 는 MCSE = 모집단별 평균의 표준편차 / √모집단 수.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

def envs_for(level):
    """STZ 수준 하나의 3환경: 환경 키 → (겹침 이름, γ, 결과 폴더). frequent는 처음 쓴 폴더 이름을 그대로 쓴다."""
    subs = ("results/easy", "results/easy_avg", "results/easy_unf") if level == "frequent" else \
        tuple(f"results/{level}_{ov}" for ov in ("fav", "avg", "unf"))
    return {f"{level}__{ov}": (ov, g, s) for (ov, g), s in zip((("fav", 0.0), ("avg", 0.5), ("unf", 1.0)), subs)}


ENVS = envs_for("frequent")  # main에서 --level로 바꾼다
# 범주 색은 고정 순서(dataviz 기본 팔레트 8색), 방법마다 표식도 달리 둔다. 그림에는 8개까지만 넣는다.
GEN = [("bomis", "BOMiS", "#2a78d6", "o"), ("ctgan", "CTGAN", "#eb6834", "s"), ("tabddpm", "TabDDPM", "#1baf7a", "D"),
       ("tabsyn", "TabSyn", "#eda100", "^"), ("smoten", "SMOTE-N", "#e87ba4", "v"),
       ("smote_enc", "SMOTE-ENC", "#008300", "P"), ("tabdiff", "TabDiff", "#6250d6", "X"), ("arf", "ARF", "#e34948", "h")]
# ROS와 smote(SMOTE-NC 대체)는 새 셀을 만들지 않아 recovery·leak이 0이라 그림에서 빼고 표에만 둔다
GEN_ALL = GEN + [("ros", "ROS", None, None), ("smote", "SMOTE (SMOTE-NC 대체)", None, None)]
SUP = [("A", "A (D1 only)", "#2a78d6", "o"), ("B", "B (D0+D1)", "#eb6834", "s"), ("C", "C (certified)", "#1baf7a", "D")]
INK, MUTED, GRID, AXIS = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
N1S = (400, 800)


def load(root: Path) -> pd.DataFrame:
    frames = []
    for env, (_, _, sub) in ENVS.items():
        for p in (root / sub).glob("*/*.csv"):
            if p.parent.name != "_failed":
                frames.append(pd.read_csv(p))
    return pd.concat(frames, ignore_index=True)


def mean_mcse(df: pd.DataFrame, keys, cols) -> pd.DataFrame:
    rows = []
    for k, g in df.groupby(keys, sort=False):
        r = dict(zip(keys, k if isinstance(k, tuple) else (k,)))
        r["cells"] = len(g)
        by_pop = g.groupby("pop")[cols].mean()
        for c in cols:
            r[c] = g[c].mean()
            r[c + "_mcse"] = by_pop[c].std(ddof=1) / np.sqrt(len(by_pop)) if len(by_pop) > 1 else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def style(ax):
    ax.set_facecolor("#fcfcfb")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor="#52514e", labelsize=9)


def env_lines(ax, table, methods, col, n1, total):
    """x = 환경(γ 순서), 방법마다 점 + MCSE 막대 + 2px 선. 겹치지 않게 방법별로 조금 비켜 둔다."""
    xs = np.arange(len(ENVS))
    k = len(methods)
    for i, (m, label, color, marker) in enumerate(methods):
        t = table[(table.method == m) & (table.n1 == n1)].set_index("env").reindex(list(ENVS))
        if t[col].notna().sum() == 0:
            continue
        off = (i - (k - 1) / 2) * 0.06
        partial = (t["cells"].fillna(0) < total).any()
        ax.errorbar(xs + off, t[col], yerr=t[col + "_mcse"], color=color, marker=marker, markersize=6,
                    linewidth=2, linestyle="--" if partial else "-", capsize=0, elinewidth=1.2,
                    markeredgecolor="#fcfcfb", markeredgewidth=1.5, label=label)
    ax.set_xticks(xs, [f"{v[0]}\n(γ={v[1]:g})" for v in ENVS.values()])
    ax.set_xlim(-0.5, len(ENVS) - 0.5)
    style(ax)


def fig_grid(table, methods, cols, titles, ylims, out, total, suptitle):
    fig, axes = plt.subplots(len(N1S), len(cols), figsize=(3.4 * len(cols), 5.6), sharex=True)
    fig.patch.set_facecolor("#fcfcfb")
    for r, n1 in enumerate(N1S):
        for c, col in enumerate(cols):
            ax = axes[r, c]
            env_lines(ax, table, methods, col, n1, total)
            ax.set_ylim(*ylims[c])
            if r == 0:
                ax.set_title(titles[c], fontsize=10, color=INK, loc="left")
            if c == 0:
                ax.set_ylabel(f"n1 = {n1}", fontsize=10, color=INK)
    shown = {l for ax in axes.flat for l in ax.get_legend_handles_labels()[1]}  # 결과가 있는 방법만
    handles = [Line2D([], [], color=c, marker=mk, markersize=6, linewidth=2, markeredgecolor="#fcfcfb")
               for _, lab, c, mk in methods if lab in shown]
    labels = [lab for _, lab, _, _ in methods if lab in shown]
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.965), labelcolor="#52514e")
    fig.suptitle(suptitle, y=1.0, fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def fig_loss(bom, out):
    """BOMiS 회수율 분해: 1 = 회수(R4) + 선택 손실 + 개수 손실 + support 손실. 막대 하나 = (환경, n1)."""
    parts = [("diag_R4", "recovered", "#2a78d6"), ("diag_loss_select", "lost at selection", "#eb6834"),
             ("diag_loss_count", "lost at count (L')", "#1baf7a"), ("diag_loss_support", "lost at support", "#eda100")]
    rows = [(f"{ENVS[e][0]}  n1={n1}", bom[(bom.env == e) & (bom.n1 == n1)]) for e in ENVS for n1 in N1S]
    rows = [(lab, g) for lab, g in rows if len(g)]
    fig, ax = plt.subplots(figsize=(8.2, 0.48 * len(rows) + 1.4))
    fig.patch.set_facecolor("#fcfcfb")
    for y, (lab, g) in enumerate(rows):
        left = 0.0
        for col, name, color in parts:
            v = g[col].mean()
            ax.barh(y, v, left=left, color=color, height=0.62, edgecolor="#fcfcfb", linewidth=2,
                    label=name if y == 0 else None)
            if v >= 0.06:
                ax.text(left + v / 2, y, f"{v:.2f}", ha="center", va="center", fontsize=8.5, color="#fcfcfb")
            left += v
    ax.set_yticks(range(len(rows)), [lab for lab, _ in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of missing SZ mass $M_0$", fontsize=9, color="#52514e")
    style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=4, frameon=False, fontsize=9, labelcolor="#52514e")
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def md_table(table, methods, cols, names):
    lines = ["| 환경 | 방법 | $n_1$ | 칸 | " + " | ".join(names) + " |", "|" + "---|" * (4 + len(cols))]
    for e in ENVS:
        for m, label, _, _ in methods:
            for n1 in N1S:
                t = table[(table.env == e) & (table.method == m) & (table.n1 == n1)]
                if not len(t):
                    continue
                r = t.iloc[0]
                vals = [("–" if pd.isna(r[c]) else f"{r[c]:.3f}" + ("" if pd.isna(r[c + "_mcse"]) else f" ± {r[c + '_mcse']:.3f}"))
                        for c in cols]
                lines.append(f"| {ENVS[e][0]} | {label} | {n1} | {int(r['cells'])} | " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--fig-dir", default="reports/figures")
    ap.add_argument("--tables", default="reports/easy_report_tables.md")
    ap.add_argument("--level", default="frequent", choices=["frequent", "mid", "rare"])
    ap.add_argument("--prefix", default="easy", help="그림 파일 이름 앞부분")
    args = ap.parse_args()
    global ENVS
    ENVS = envs_for(args.level)
    pre = args.prefix
    root, figs = Path(args.root), Path(args.fig_dir)
    figs.mkdir(parents=True, exist_ok=True)
    df = load(root)
    total = 30  # (환경, 방법, n1)마다 모집단 3 × seed 10
    keys = ["env", "method", "n1"]
    gcols = ["recovery", "leak", "control", "novel", "rec_count", "uni_tv", "pair_tv", "auroc"]
    gen = mean_mcse(df[df.method.isin([m for m, *_ in GEN_ALL])], keys, gcols)
    scols = ["sz_rec_mass", "sz_rec_count", "stz_sup_count", "all_cert", "n_false", "S_size"]
    sup = mean_mcse(df[df.method.isin([m for m, *_ in SUP])], keys, scols)
    bom = df[df.method == "bomis"]
    dcols = ["diag_R1", "diag_loss_count", "diag_loss_select", "diag_R4", "diag_M_hat", "diag_M0", "diag_L_prime",
             "diag_n_sz", "diag_Qu_stz", "diag_Q_stz"]
    diag = mean_mcse(bom, keys, dcols)

    # BOMiS − baseline 짝 차이 (같은 환경·모집단·n1·seed 칸끼리)
    k = ["env", "pop", "n1", "seed"]
    b = bom.set_index(k)[["recovery", "leak", "control"]]
    diffs = []
    for m, label, *_ in GEN_ALL[1:]:
        o = df[df.method == m].set_index(k)[["recovery", "leak", "control"]]
        d = (b - o).dropna(how="all").reset_index()
        d = d[d.set_index(k).index.isin(o.index)]
        if len(d):
            d["method"] = m
            diffs.append(d)
    dd = mean_mcse(pd.concat(diffs), keys, ["recovery", "leak", "control"]) if diffs else pd.DataFrame()

    fig_grid(gen, GEN, ["recovery", "leak", "control"],
             ["recovery (higher better)", "leak (lower better)", "control (higher better)"],
             [(0, 0.7), (-0.01, 0.3), (0.5, 1.02)], figs / f"{pre}_fig1_generation.png", total,
             "Generated minority rows: mean over cells ± MCSE (dashed = still running)")
    fig_grid(sup, SUP, ["sz_rec_count", "stz_sup_count"],
             ["SZ count recovery (higher better)", "STZ suppression (higher better)"],
             [(0.5, 1.02), (0.5, 1.02)], figs / f"{pre}_fig2_support.png", total,
             "Support stage A / B / C: mean over cells ± MCSE")
    fig_loss(bom, figs / f"{pre}_fig3_bomis_loss.png")

    diff_methods = [g for g in GEN_ALL[1:] if g[0] in set(dd.get("method", []))]
    out = ["# 자동 생성 표 (scripts/make_easy_report.py)", "",
           f"칸 수: {df.groupby(['method']).size().to_dict()}", "",
           "## 생성 행 지표", "", md_table(gen, GEN_ALL, ["recovery", "leak", "control", "novel", "rec_count", "pair_tv", "auroc"],
                                        ["recovery", "leak", "control", "novel", "개수 회수", "pair_tv", "auroc"]), "",
           "## BOMiS − baseline 짝 차이", "",
           md_table(dd, [(m, f"BOMiS − {lab}", c, mk) for m, lab, c, mk in diff_methods], ["recovery", "leak", "control"],
                    ["recovery", "leak", "control"]) if len(dd) else "(없음)", "",
           "## support 단계", "", md_table(sup, SUP, scols, ["SZ 질량 회수", "SZ 개수 회수", "STZ 억제", "전부 인증", "거짓 인증 수", "|S|"]), "",
           "## BOMiS 진단", "", md_table(diag, GEN[:1], dcols, ["R1", "개수 손실", "선택 손실", "R4", "M_hat", "M0", "L'", "SZ 셀 수", "mask 전 STZ 질량", "mask 후 STZ 질량"]), ""]
    Path(args.tables).write_text("\n".join(out))
    print("wrote", args.tables, "and figures in", figs)


if __name__ == "__main__":
    main()
