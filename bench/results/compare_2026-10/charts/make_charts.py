#!/usr/bin/env python3
"""Charts for the compare_2026-10 benchmark (QEngine v1.0 / v0.1 vs vLLM 0.28).

Reads only the summary CSVs one directory up and writes PNGs + a standalone
index.html (PNGs embedded as base64) into this directory. No GPU work.

    python3 bench/results/compare_2026-10/charts/make_charts.py
"""
import base64
import csv
import html
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)

# ---------------------------------------------------------------- style ----
# Categorical slots 1/3/2 of the dataviz reference palette (validated all-pairs,
# light mode: CVD dE >= 9.2, normal-vision dE >= 24.0).
C_V10 = "#2a78d6"   # QEngine v1.0 (brand, strongest)
C_V01 = "#1baf7a"   # QEngine v0.1
C_VLLM = "#eb6834"  # vLLM 0.28 (best config)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

ENGINES = [  # (key, display name, color)
    ("qengine-now", "QEngine v1.0", C_V10),
    ("qengine-sunday", "QEngine v0.1", C_V01),
    ("vllm-best", "vLLM 0.28 (best config)", C_VLLM),
]
NAME = {k: n for k, n, _ in ENGINES}
COLOR = {k: c for k, _, c in ENGINES}
VARIANT = {"vllm-mtp": "MTP", "vllm-nospec": "no-spec"}

FOOTER = "Qwen3.8-27B INT4 · RTX 3090 · vllm bench serve"
HW_NOTE = ("QEngine: 3× RTX 3090 (TP=3) · vLLM: 2× RTX 3090 (TP=2; this model "
           "cannot be sharded 3 ways in vLLM)")

for fam in ("Adwaita Sans", "Noto Sans", "DejaVu Sans"):
    try:
        font_manager.findfont(fam, fallback_to_default=False)
        FONT = fam
        break
    except Exception:
        continue
plt.rcParams.update({
    "font.family": FONT,
    "font.size": 11,
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK2,
    "axes.titlecolor": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "grid.linestyle": "-",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "text.color": INK,
})

DPI = 200
LANDSCAPE = (1920 / DPI, 1080 / DPI)
SQUARE = (1200 / DPI, 1200 / DPI)


# ----------------------------------------------------------------- data ----
def read_csv(rel):
    with open(os.path.join(SRC, rel), newline="") as f:
        return list(csv.DictReader(f))


def fnum(s):
    return float(s) if s not in ("", None) else None


classic = {}   # (engine, scenario, c) -> row
for r in read_csv("classic/summary.csv"):
    classic[(r["engine"], r["scenario"], int(r["concurrency"]))] = r
goodput = {}   # (engine, scenario, rate) -> row
for r in read_csv("goodput/summary.csv"):
    goodput[(r["engine"], r["scenario"], float(r["request_rate"]))] = r
prefill = {}   # (engine, target_len) -> row
for r in read_csv("prefill/summary.csv"):
    prefill[(r["engine"], int(r["target_len"]))] = r
drift = read_csv("drift/summary.csv")

best_variant = {}  # (block, scenario, point) -> vllm-mtp | vllm-nospec
for r in read_csv("best_vllm.csv"):
    block, scen, point = r["block"], r["scenario"], r["point"]
    key_v = float(point.split("=")[1])
    best_variant[(block, scen, key_v)] = r["best_variant"]
    v = r["best_variant"]
    if block == "classic":
        classic[("vllm-best", scen, int(key_v))] = classic[(v, scen, int(key_v))]
    elif block == "goodput":
        goodput[("vllm-best", scen, key_v)] = goodput[(v, scen, key_v)]
    elif block == "prefill":
        prefill[("vllm-best", int(key_v))] = prefill[(v, int(key_v))]

SCEN_C = {"short": [1, 8, 16, 32], "agent": [1, 8, 16, 32],
          "sharegpt": [1, 8, 16, 32], "rag16k": [1, 8]}
SCEN_LABEL = {
    "short": "Short chat · 1K → 512",
    "agent": "Agent · 500 → 2K",
    "sharegpt": "ShareGPT (real chats)",
    "rag16k": "RAG · 15K → 128",
}
LAT_LABEL = {"short": "Short chat", "agent": "Agent", "sharegpt": "ShareGPT",
             "rag16k": "RAG 15K"}
GP_SCEN = ["goodput_chat", "goodput_long16k", "goodput_agent"]
GP_LABEL = {
    "goodput_chat": "Chat · 2K in / 256 out",
    "goodput_long16k": "Long context · 16K in / 512 out",
    "goodput_agent": "Agent · 6K in / 1K out",
}
PF_LENS = [1024, 8192, 32768, 100000, 200000]
PF_LABEL = {1024: "1K", 8192: "8K", 32768: "32K", 100000: "100K", 200000: "200K"}


def cval(e, s, c, col):
    return fnum(classic[(e, s, c)][col])


def speedup(s, c):
    return cval("qengine-now", s, c, "output_tok_s") / cval("vllm-best", s, c, "output_tok_s")


def fmt_int(v):
    return f"{v:,.0f}"


def fmt_x(v):
    return f"{v:.2f}".rstrip("0").rstrip(".") + "×" if v < 10 else f"{v:.0f}×"


def fmt_x1(v):
    return f"{v:.1f}×"


# --------------------------------------------------------------- helpers ----
def header(fig, title, subtitle, top=0.955):
    fig.text(0.04, top, title, fontsize=19, fontweight="bold", color=INK, va="top")
    fig.text(0.04, top - 0.062, subtitle, fontsize=11.5, color=INK2, va="top")


def footer(fig, note=None, y=0.018):
    fig.text(0.04, y, FOOTER, fontsize=8.6, color=MUTED, va="bottom")
    if note:
        fig.text(0.04, y + 0.036, note, fontsize=8.6, color=INK2, va="bottom")


def legend_row(fig, keys, y, x=0.04, names=None):
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR[k]) for k in keys]
    labels = [names[k] if names else NAME[k] for k in keys]
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(x - 0.008, y),
               ncol=len(keys), fontsize=11, handlelength=1.1, handleheight=1.1,
               columnspacing=1.8, labelcolor=INK)


def style_ax(ax, ygrid=True):
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", visible=ygrid)
    ax.tick_params(length=0, labelsize=10)
    ax.spines["left"].set_visible(False)


def kfmt(v, _pos=None):
    if v >= 1000:
        return f"{v/1000:g}K"
    return f"{v:g}"


def save(fig, name):
    path = os.path.join(HERE, name)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


# ---------------------------------------------------------------- charts ----
def chart_throughput():
    fig = plt.figure(figsize=LANDSCAPE)
    header(fig, "Output throughput vs concurrent requests",
           "Output tokens/s, higher is better. Label above each group = QEngine v1.0 ÷ vLLM best.")
    legend_row(fig, ["qengine-now", "qengine-sunday", "vllm-best"], y=0.835)
    order = ["short", "agent", "sharegpt", "rag16k"]
    widths = [4, 4, 4, 2.3]
    gs = fig.add_gridspec(1, 4, left=0.055, right=0.975, top=0.70, bottom=0.215,
                          wspace=0.28, width_ratios=widths)
    keys = ["qengine-now", "qengine-sunday", "vllm-best"]
    bw = 0.26
    for i, s in enumerate(order):
        ax = fig.add_subplot(gs[0, i])
        cs = SCEN_C[s]
        ymax = max(cval(k, s, c, "output_tok_s") for k in keys for c in cs)
        for j, k in enumerate(keys):
            xs = [x + (j - 1) * bw for x in range(len(cs))]
            ys = [cval(k, s, c, "output_tok_s") for c in cs]
            ax.bar(xs, ys, width=bw, color=COLOR[k], edgecolor=SURFACE, linewidth=1.2,
                   zorder=2)
            if k == "qengine-now":
                for x, y in zip(xs, ys):
                    ax.text(x, y + ymax * 0.012, fmt_int(y), ha="center", va="bottom",
                            fontsize=7.6, color=INK2, rotation=0)
        for x, c in enumerate(cs):
            top = max(cval(k, s, c, "output_tok_s") for k in keys)
            ax.text(x, top + ymax * 0.085, fmt_x1(speedup(s, c)), ha="center",
                    va="bottom", fontsize=11, fontweight="bold", color=INK)
        ax.set_xticks(range(len(cs)))
        ax.set_xticklabels([f"c={c}" for c in cs], fontsize=10)
        ax.set_ylim(0, ymax * 1.2)
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=6, steps=[1, 2, 4, 5, 10]))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(kfmt))
        ax.set_title(SCEN_LABEL[s], fontsize=11, loc="left", color=INK, pad=8,
                     fontweight="semibold")
        style_ax(ax)
    ns = defaultdict(list)
    for (b, s, c), v in best_variant.items():
        if b == "classic" and v == "vllm-nospec":
            ns[s].append(int(c))
    won = [f"{LAT_LABEL[s].lower() if s == 'short' else LAT_LABEL[s]} c="
           + "/".join(str(c) for c in sorted(cs)) for s, cs in ns.items()]
    fig.text(0.04, 0.098,
             "vLLM best = per point the faster of MTP (4 draft tokens) and no-spec. No-spec won at "
             + ", ".join(won) + "; MTP everywhere else.", fontsize=8.6, color=INK2)
    footer(fig, HW_NOTE)
    return save(fig, "throughput.png")


def chart_prefill():
    fig = plt.figure(figsize=LANDSCAPE)
    t_now = fnum(prefill[("qengine-now", 200000)]["ttft_ms_median"]) / 1000
    t_vl = fnum(prefill[("vllm-best", 200000)]["ttft_ms_median"]) / 1000
    header(fig, f"200K-token prompt: first token in {t_now:.1f} s instead of {t_vl:.1f} s",
           "Prefill, one request at a time, unique prompts (no prefix-cache hits). "
           "Median of 3 runs. Lower TTFT is better.")
    legend_row(fig, ["qengine-now", "qengine-sunday", "vllm-best"], y=0.835)
    gs = fig.add_gridspec(1, 2, left=0.06, right=0.985, top=0.70, bottom=0.215,
                          wspace=0.2, width_ratios=[1.15, 1])
    keys = ["qengine-now", "qengine-sunday", "vllm-best"]

    # left: TTFT vs prompt length (linear axes, lines)
    ax = fig.add_subplot(gs[0, 0])
    xs = [fnum(prefill[("qengine-now", L)]["prompt_tokens"]) / 1000 for L in PF_LENS]
    for k in keys[::-1]:
        ys = [fnum(prefill[(k, L)]["ttft_ms_median"]) / 1000 for L in PF_LENS]
        ax.plot(xs, ys, color=COLOR[k], lw=2.2, solid_capstyle="round", zorder=3)
        ax.scatter(xs, ys, s=42, color=COLOR[k], edgecolor=SURFACE, linewidth=1.6, zorder=4)
    for k, dy in (("qengine-now", -2), ("vllm-best", 0), ("qengine-sunday", 0)):
        y = fnum(prefill[(k, 200000)]["ttft_ms_median"]) / 1000
        ax.annotate(f"{y:.1f} s", (xs[-1], y), xytext=(8, dy), textcoords="offset points",
                    va="center", fontsize=10.5, color=INK, fontweight="semibold")
    ax.text(xs[-1] + 9, 104, f"{t_vl / t_now:.1f}×\nfaster", ha="left", va="center",
            fontsize=13, fontweight="bold", color=INK, linespacing=1.1)
    ax.annotate("", xy=(xs[-1] - 4, t_now + 3), xytext=(xs[-1] - 4, t_vl - 3),
                arrowprops=dict(arrowstyle="<->", color=INK2, lw=1))
    ax.set_xlim(0, 250)
    ax.set_ylim(0, 145)
    ax.set_xticks([0, 50, 100, 150, 200])
    ax.set_xticklabels(["0", "50K", "100K", "150K", "200K"])
    ax.set_xlabel("Prompt length (tokens)", fontsize=10)
    ax.set_title("Time to first token (s)", fontsize=11, loc="left", fontweight="semibold",
                 pad=8)
    ax.grid(axis="x", visible=False)
    ax.tick_params(length=0, labelsize=10)
    ax.spines["left"].set_visible(False)

    # right: prompt tokens/s grouped bars
    ax = fig.add_subplot(gs[0, 1])
    bw = 0.26
    ymax = max(fnum(prefill[(k, L)]["prompt_tok_s_median"]) for k in keys for L in PF_LENS)
    for j, k in enumerate(keys):
        bx = [x + (j - 1) * bw for x in range(len(PF_LENS))]
        ys = [fnum(prefill[(k, L)]["prompt_tok_s_median"]) for L in PF_LENS]
        ax.bar(bx, ys, width=bw, color=COLOR[k], edgecolor=SURFACE, linewidth=1.2, zorder=2)
    for x, L in enumerate(PF_LENS):
        a = fnum(prefill[("vllm-best", L)]["ttft_ms_median"])
        b = fnum(prefill[("qengine-now", L)]["ttft_ms_median"])
        top = max(fnum(prefill[(k, L)]["prompt_tok_s_median"]) for k in keys)
        ax.text(x, top + ymax * 0.03, fmt_x1(a / b), ha="center", va="bottom", fontsize=11,
                fontweight="bold", color=INK)
    ax.set_xticks(range(len(PF_LENS)))
    ax.set_xticklabels([PF_LABEL[L] for L in PF_LENS])
    ax.set_xlabel("Prompt length (tokens)", fontsize=10)
    ax.set_ylim(0, ymax * 1.18)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(kfmt))
    ax.set_title("Prompt tokens/s · label = speed-up vs vLLM", fontsize=11,
                 loc="left", fontweight="semibold", pad=8)
    style_ax(ax)
    fig.text(0.04, 0.098, "vLLM best = no-spec at every length (MTP was 3–6% slower). "
             "TTFT is measured client-side and includes HTTP, tokenization and the first sample.",
             fontsize=8.6, color=INK2)
    footer(fig, HW_NOTE)
    return save(fig, "prefill.png")


def chart_goodput():
    fig = plt.figure(figsize=LANDSCAPE)
    header(fig, "Requests served within the latency SLA under live traffic",
           "Poisson arrivals at a fixed request rate. SLA per request: time to first token "
           "≤ 3 s and time per output token ≤ 50 ms.")
    legend_row(fig, ["qengine-now", "qengine-sunday", "vllm-best"], y=0.835)
    gs = fig.add_gridspec(1, 3, left=0.06, right=0.985, top=0.70, bottom=0.215, wspace=0.16)
    keys = ["qengine-now", "qengine-sunday", "vllm-best"]
    for i, s in enumerate(GP_SCEN):
        ax = fig.add_subplot(gs[0, i])
        rates = sorted({r for (e, sc, r) in goodput if sc == s and e == "qengine-now"})
        for k in keys[::-1]:
            ys = [fnum(goodput[(k, s, r)]["goodput_frac"]) * 100 for r in rates]
            ax.plot(rates, ys, color=COLOR[k], lw=2.2, zorder=3, solid_capstyle="round")
            ax.scatter(rates, ys, s=40, color=COLOR[k], edgecolor=SURFACE, linewidth=1.6,
                       zorder=4)
        # label: highest rate where v1.0 still >= 95% and vLLM's value at that rate
        ok = [r for r in rates if fnum(goodput[("qengine-now", s, r)]["goodput_frac"]) >= 0.95]
        if ok:
            r = ok[-1]
            q = fnum(goodput[("qengine-now", s, r)]["goodput_frac"]) * 100
            v = fnum(goodput[("vllm-best", s, r)]["goodput_frac"]) * 100
            both = abs(q - v) < 3
            ax.annotate(f"{q:.0f}%" + (" (both)" if both else ""), (r, q), xytext=(0, 9),
                        textcoords="offset points",
                        ha="center", fontsize=10.5, fontweight="semibold", color=INK)
            if abs(q - v) < 3:
                pass  # same value: the single label above covers both
            elif r == rates[0]:
                ax.annotate(f"{v:.0f}%", (r, v), xytext=(0, -11), textcoords="offset points",
                            ha="center", va="top", fontsize=10.5, fontweight="semibold",
                            color=INK)
            else:
                ax.annotate(f"{v:.0f}%", (r, v), xytext=(10, 8), textcoords="offset points",
                            ha="left", va="center", fontsize=10.5, fontweight="semibold",
                            color=INK)
        ax.set_ylim(-4, 112)
        ax.set_yticks([0, 25, 50, 75, 100])
        if i == 0:
            ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
        else:
            ax.set_yticklabels([])
        ax.set_xticks(rates)
        ax.set_xticklabels([f"{r:g}" for r in rates])
        ax.set_xlabel("Request rate (requests/s)", fontsize=10)
        ax.set_title(GP_LABEL[s], fontsize=11, loc="left", fontweight="semibold", pad=8)
        ax.grid(axis="x", visible=False)
        ax.tick_params(length=0, labelsize=10)
        ax.spines["left"].set_visible(False)
    mtp = [f"{GP_LABEL[s].split(' ·')[0].lower()} {r:g}/s" for (b, s, r), v in
           sorted(best_variant.items()) if b == "goodput" and v == "vllm-mtp"]
    fig.text(0.04, 0.098, "% in SLA = goodput ÷ request throughput. vLLM best = no-spec except "
             + ", ".join(mtp) + " (MTP). Requests share a prompt prefix; lengths vary ±50%.",
             fontsize=8.6, color=INK2)
    footer(fig, HW_NOTE)
    return save(fig, "goodput.png")


def chart_latency():
    fig = plt.figure(figsize=LANDSCAPE)
    header(fig, "Single user: how fast tokens stream",
           "One request at a time (c=1). Medians, lower is better. Streaming speed = 1000 ÷ TPOT.")
    legend_row(fig, ["qengine-now", "qengine-sunday", "vllm-best"], y=0.835)
    gs = fig.add_gridspec(1, 2, left=0.15, right=0.975, top=0.70, bottom=0.19, wspace=0.3)
    keys = ["qengine-now", "qengine-sunday", "vllm-best"]
    order = ["short", "agent", "sharegpt", "rag16k"]
    for i, (col, title, unit) in enumerate([
            ("tpot_p50_ms", "Time per output token, median (ms)", "ms"),
            ("ttft_p50_ms", "Time to first token, median (ms)", "ms")]):
        ax = fig.add_subplot(gs[0, i])
        bh = 0.26
        xmax = max(cval(k, s, 1, col) for k in keys for s in order)
        for j, k in enumerate(keys):
            ys = [len(order) - 1 - y + (1 - j) * bh for y in range(len(order))]
            vals = [cval(k, s, 1, col) for s in order]
            ax.barh(ys, vals, height=bh, color=COLOR[k], edgecolor=SURFACE, linewidth=1.2,
                    zorder=2)
            for y, v in zip(ys, vals):
                if col == "tpot_p50_ms":
                    lab = f"{v:.1f} ms · {1000 / v:.0f} tok/s"
                else:
                    lab = f"{v / 1000:.1f} s" if v >= 1000 else f"{v:.0f} ms"
                ax.text(v + xmax * 0.012, y, lab, va="center", fontsize=8.6, color=INK2)
        ax.set_yticks(range(len(order)))
        if i == 0:
            ax.set_yticklabels([LAT_LABEL[s] for s in order[::-1]], fontsize=10.5, color=INK)
        else:
            ax.set_yticklabels([])
        ax.set_xlim(0, xmax * (1.5 if col == "tpot_p50_ms" else 1.25))
        ax.set_title(title, fontsize=11, loc="left", fontweight="semibold", pad=8)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
        ax.tick_params(length=0, labelsize=10)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(kfmt))
    fig.text(0.04, 0.098, "vLLM best at c=1 = MTP (speculative decoding, 4 draft tokens) in "
             "every scenario. QEngine uses its built-in MTP head (adaptive, up to 7 drafts).",
             fontsize=8.6, color=INK2)
    footer(fig, HW_NOTE)
    return save(fig, "latency.png")


def chart_vllm_wins():
    """Honest chart: the points where vLLM is ahead, with the trade-off shown."""
    fig = plt.figure(figsize=LANDSCAPE)
    header(fig, "Where vLLM is ahead",
           "RAG with 8 concurrent 15K-token prompts: vLLM's first token comes sooner, "
           f"then each token takes {cval('vllm-mtp', 'rag16k', 8, 'tpot_p50_ms') / cval('qengine-now', 'rag16k', 8, 'tpot_p50_ms'):.0f}× longer.")
    names = {"qengine-now": "QEngine v1.0", "vllm-best": "vLLM 0.28 (variant named below)"}
    legend_row(fig, ["qengine-now", "vllm-best"], y=0.835, names=names)
    gs = fig.add_gridspec(1, 3, left=0.05, right=0.985, top=0.665, bottom=0.23, wspace=0.42,
                          width_ratios=[1, 1, 1.75])

    def bars(ax, items, unit_fmt, title, sub):
        # items: list of (label, value, engine_key)
        n = len(items)
        xs = list(range(n))
        vals = [v for _, v, _ in items]
        ax.bar(xs, vals, width=0.5, color=[COLOR[k] for _, _, k in items],
               edgecolor=SURFACE, linewidth=1.2, zorder=2)
        top = max(vals)
        for x, v in zip(xs, vals):
            ax.text(x, v + top * 0.02, unit_fmt(v), ha="center", va="bottom", fontsize=11,
                    fontweight="semibold", color=INK)
        ax.set_xticks(xs)
        ax.set_xticklabels([lab for lab, _, _ in items], fontsize=9.5)
        ax.set_ylim(0, top * 1.18)
        ax.set_title(title, fontsize=11, loc="left", fontweight="semibold", pad=22)
        ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=9, color=INK2, va="bottom")
        style_ax(ax)

    q_ttft = cval("qengine-now", "rag16k", 8, "ttft_p50_ms") / 1000
    m_ttft = cval("vllm-mtp", "rag16k", 8, "ttft_p50_ms") / 1000
    q_tpot = cval("qengine-now", "rag16k", 8, "tpot_p50_ms")
    m_tpot = cval("vllm-mtp", "rag16k", 8, "tpot_p50_ms")
    n_ttft = cval("vllm-nospec", "rag16k", 8, "ttft_p50_ms") / 1000
    n_tpot = cval("vllm-nospec", "rag16k", 8, "tpot_p50_ms")
    ax = fig.add_subplot(gs[0, 0])
    bars(ax, [("QEngine v1.0", q_ttft, "qengine-now"), ("vLLM MTP", m_ttft, "vllm-best")],
         lambda v: f"{v:.1f} s", "RAG c=8 · first token (s)", "median TTFT · vLLM wins")
    ax = fig.add_subplot(gs[0, 1])
    bars(ax, [("QEngine v1.0", q_tpot, "qengine-now"), ("vLLM MTP", m_tpot, "vllm-best")],
         lambda v: f"{v:.1f} ms", "RAG c=8 · per token (ms)", "median TPOT · QEngine wins")
    ax = fig.add_subplot(gs[0, 2])
    pts = [("short", 32, "tpot_p50_ms", "vllm-nospec", "Short c=32\nper token"),
           ("sharegpt", 8, "ttft_p50_ms", "vllm-nospec", "ShareGPT c=8\nfirst token"),
           ("sharegpt", 32, "ttft_p50_ms", "vllm-nospec", "ShareGPT c=32\nfirst token")]
    bw = 0.34
    top = 0
    for x, (s, c, col, v, lab) in enumerate(pts):
        q = cval("qengine-now", s, c, col)
        w = cval(v, s, c, col)
        top = max(top, q, w)
        for dx, val, k in ((-bw / 2, q, "qengine-now"), (bw / 2, w, "vllm-best")):
            ax.bar(x + dx, val, width=bw, color=COLOR[k], edgecolor=SURFACE, linewidth=1.2,
                   zorder=2)
    for x, (s, c, col, v, lab) in enumerate(pts):
        for dx, val in ((-bw / 2, cval("qengine-now", s, c, col)), (bw / 2, cval(v, s, c, col))):
            ax.text(x + dx, val + top * 0.02, f"{val:.0f}" if val >= 100 else f"{val:.1f}",
                    ha="center", va="bottom", fontsize=9.5, color=INK, fontweight="semibold")
    ax.set_xticks(range(len(pts)))
    ax.set_xticklabels([p[4] for p in pts], fontsize=9)
    ax.set_ylim(0, top * 1.18)
    ax.set_title("Smaller margins (ms, lower is better)", fontsize=11, loc="left",
                 fontweight="semibold", pad=22)
    ax.text(0, 1.02, "vLLM no-spec variant vs QEngine v1.0", transform=ax.transAxes,
            fontsize=9, color=INK2, va="bottom")
    style_ax(ax)
    fig.text(0.04, 0.098,
             f"RAG c=8, vLLM no-spec: TTFT {n_ttft:.1f} s, TPOT {n_tpot:.0f} ms. Output throughput "
             f"at RAG c=8 is still {speedup('rag16k', 8):.1f}× higher on QEngine v1.0 "
             f"({cval('qengine-now', 'rag16k', 8, 'output_tok_s'):.0f} vs "
             f"{cval('vllm-best', 'rag16k', 8, 'output_tok_s'):.0f} tok/s).",
             fontsize=8.6, color=INK2)
    footer(fig, HW_NOTE)
    return save(fig, "vllm_wins.png")


def hero_numbers():
    t_now = fnum(prefill[("qengine-now", 200000)]["ttft_ms_median"]) / 1000
    t_vl = fnum(prefill[("vllm-best", 200000)]["ttft_ms_median"]) / 1000
    c1 = [speedup(s, 1) for s in ("short", "agent", "sharegpt")]
    cN = [speedup(s, c) for s in ("short", "agent", "sharegpt") for c in (8, 16, 32)]
    gp_q = fnum(goodput[("qengine-now", "goodput_chat", 2.0)]["goodput_frac"]) * 100
    gp_v = fnum(goodput[("vllm-best", "goodput_chat", 2.0)]["goodput_frac"]) * 100
    return dict(t_now=t_now, t_vl=t_vl, pf_x=t_vl / t_now, c1_lo=min(c1), c1_hi=max(c1),
                cN_lo=min(cN), cN_hi=max(cN), gp_q=gp_q, gp_v=gp_v)


def chart_hero():
    h = hero_numbers()
    fig = plt.figure(figsize=SQUARE)
    fig.text(0.07, 0.935, "QEngine v1.0 vs vLLM 0.28", fontsize=21, fontweight="bold",
             color=INK, va="top")
    fig.text(0.07, 0.868, "Same Qwen3.8-27B INT4 model, same benchmark client,\n"
             "same prompts. vLLM in its best community config.",
             fontsize=10.5, color=INK2, va="top", linespacing=1.4)
    tiles = [
        (f"{h['pf_x']:.1f}×", "faster 200K-token prefill",
         f"{h['t_now']:.1f} s vs {h['t_vl']:.1f} s to first token"),
        (f"{h['c1_lo']:.1f}–{h['c1_hi']:.1f}×", "single-user output speed",
         "short chat, agent and ShareGPT at c=1"),
        (f"{h['cN_lo']:.1f}–{h['cN_hi']:.1f}×", "throughput at 8–32 users",
         "output tokens/s, same three scenarios"),
        (f"{h['gp_q']:.0f}%", "chat requests within SLA",
         f"at 2 req/s; vLLM: {h['gp_v']:.0f}%\nSLA: TTFT ≤ 3 s, TPOT ≤ 50 ms"),
    ]
    x0s = [0.07, 0.525]
    w, hgt = 0.405, 0.30
    k = 0
    for y0 in (0.475, 0.155):
        for x0 in x0s:
            big, lab, sub = tiles[k]
            k += 1
            box = FancyBboxPatch((x0, y0), w, hgt, boxstyle="round,pad=0,rounding_size=0.02",
                                 transform=fig.transFigure, facecolor="#f3f2ee",
                                 edgecolor="none", zorder=0)
            fig.patches.append(box)
            fig.patches.append(plt.Rectangle((x0 + 0.03, y0 + hgt - 0.05), 0.05, 0.008,
                                             transform=fig.transFigure, color=C_V10))
            fig.text(x0 + 0.03, y0 + hgt - 0.07, big, fontsize=30, fontweight="bold",
                     color=INK, va="top")
            fig.text(x0 + 0.03, y0 + 0.105, lab, fontsize=11.5, fontweight="semibold",
                     color=INK, va="top")
            fig.text(x0 + 0.03, y0 + 0.058, sub, fontsize=8.6, color=INK2, va="top",
                     linespacing=1.35)
    fig.text(0.07, 0.128, "QEngine: 3× RTX 3090 (TP=3) · vLLM 0.28.0 + club-3090 dual-fast: "
             "2× RTX 3090 (TP=2),\nthe most this model can be sharded in vLLM",
             fontsize=8, color=INK2, va="top", linespacing=1.35)
    fig.text(0.07, 0.03, FOOTER, fontsize=8, color=MUTED, va="bottom")
    return save(fig, "hero.png")


# ------------------------------------------------------------------ html ----
def b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def table(head, rows, num_from=1):
    out = ['<div class="tw"><table><thead><tr>']
    out += [f"<th{' class=n' if i >= num_from else ''}>{html.escape(h)}</th>"
            for i, h in enumerate(head)]
    out.append("</tr></thead><tbody>")
    for r in rows:
        out.append("<tr>" + "".join(
            f"<td{' class=n' if i >= num_from else ''}>{html.escape(str(c))}</td>"
            for i, c in enumerate(r)) + "</tr>")
    out.append("</tbody></table></div>")
    return "".join(out)


def tables():
    t = {}
    rows = []
    for s in ["short", "agent", "sharegpt", "rag16k"]:
        for c in SCEN_C[s]:
            v = best_variant[("classic", s, float(c))]
            rows.append([s, c, fmt_int(cval("qengine-now", s, c, "output_tok_s")),
                         fmt_int(cval("qengine-sunday", s, c, "output_tok_s")),
                         f"{fmt_int(cval('vllm-best', s, c, 'output_tok_s'))} ({VARIANT[v]})",
                         f"{speedup(s, c):.2f}×"])
    t["throughput"] = table(["Scenario", "c", "QEngine v1.0", "QEngine v0.1",
                             "vLLM best", "v1.0 ÷ vLLM"], rows, 1)
    rows = []
    for L in PF_LENS:
        def cell(k):
            r = prefill[(k, L)]
            return f"{fnum(r['ttft_ms_median'])/1000:.2f} s · {fmt_int(fnum(r['prompt_tok_s_median']))}"
        a = fnum(prefill[("vllm-best", L)]["ttft_ms_median"])
        b = fnum(prefill[("qengine-now", L)]["ttft_ms_median"])
        rows.append([fmt_int(fnum(prefill[("qengine-now", L)]["prompt_tokens"])),
                     cell("qengine-now"), cell("qengine-sunday"), cell("vllm-best"),
                     f"{a / b:.2f}×"])
    t["prefill"] = table(["Prompt tokens", "QEngine v1.0", "QEngine v0.1",
                          "vLLM best (no-spec)", "Speed-up"], rows, 1)
    rows = []
    for s in GP_SCEN:
        rates = sorted({r for (e, sc, r) in goodput if sc == s and e == "qengine-now"})
        for r in rates:
            v = best_variant[("goodput", s, r)]

            def g(k):
                row = goodput[(k, s, r)]
                return f"{fnum(row['goodput_frac'])*100:.0f}% · {fnum(row['goodput_req_s']):.2f}"
            rows.append([GP_LABEL[s].split(" ·")[0], f"{r:g}", g("qengine-now"),
                         g("qengine-sunday"), f"{g('vllm-best')} ({VARIANT[v]})"])
    t["goodput"] = table(["Scenario", "Rate (req/s)", "QEngine v1.0", "QEngine v0.1",
                          "vLLM best"], rows, 1)
    rows = []
    for s in ["short", "agent", "sharegpt", "rag16k"]:
        def lt(k):
            return (f"{cval(k, s, 1, 'tpot_p50_ms'):.1f} ms / "
                    f"{cval(k, s, 1, 'ttft_p50_ms'):,.0f} ms")
        rows.append([s, lt("qengine-now"), lt("qengine-sunday"), lt("vllm-best") + " (MTP)"])
    t["latency"] = table(["Scenario (c=1)", "QEngine v1.0", "QEngine v0.1", "vLLM best"],
                         rows, 1)
    rows = []
    for s, c in [("rag16k", 8), ("short", 32), ("sharegpt", 8), ("sharegpt", 32)]:
        rows.append([f"{s} c={c}",
                     f"{cval('qengine-now', s, c, 'ttft_p50_ms'):,.0f} / {cval('qengine-now', s, c, 'tpot_p50_ms'):.1f}",
                     f"{cval('vllm-mtp', s, c, 'ttft_p50_ms'):,.0f} / {cval('vllm-mtp', s, c, 'tpot_p50_ms'):.1f}",
                     f"{cval('vllm-nospec', s, c, 'ttft_p50_ms'):,.0f} / {cval('vllm-nospec', s, c, 'tpot_p50_ms'):.1f}"])
    t["wins"] = table(["Point", "QEngine v1.0", "vLLM MTP", "vLLM no-spec"], rows, 1)
    rows = []
    dn = {"qengine-now": "QEngine v1.0", "qengine-sunday": "QEngine v0.1",
          "vllm-mtp": "vLLM MTP", "vllm-nospec": "vLLM no-spec"}
    for r in drift:
        rows.append([dn[r["engine"]], r["concurrency"], fmt_int(fnum(r["classic_output_tok_s"])),
                     fmt_int(fnum(r["drift_output_tok_s"])), f"{fnum(r['output_delta_pct']):+.1f}%"])
    t["drift"] = table(["Engine", "c", "First run tok/s", "Repeat tok/s", "Δ"], rows, 1)
    return t


CSS = """
:root{color-scheme:light;--bg:#f9f9f7;--card:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
--line:#e1e0d9;--accent:#2a78d6;--chip:#f3f2ee}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--bg:#0d0d0d;
--card:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--line:#2c2c2a;--accent:#3987e5;--chip:#232321}}
:root[data-theme="dark"]{color-scheme:dark;--bg:#0d0d0d;--card:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;
--muted:#898781;--line:#2c2c2a;--accent:#3987e5;--chip:#232321}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1080px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:clamp(26px,5vw,38px);line-height:1.15;margin:0 0 8px;letter-spacing:-.01em}
h2{font-size:22px;margin:0 0 6px}
h3{font-size:16px;margin:20px 0 6px}
p{margin:0 0 10px;color:var(--ink2)}
.lede{font-size:18px;max-width:760px}
.hw{display:flex;flex-wrap:wrap;gap:8px;margin:16px 0 8px}
.hw span{background:var(--chip);border:1px solid var(--line);border-radius:999px;padding:4px 12px;font-size:13px;color:var(--ink2)}
.key{display:flex;flex-wrap:wrap;gap:18px;margin:8px 0 24px;font-size:14px;color:var(--ink2)}
.key i{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:6px;vertical-align:-1px}
section.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:20px;margin:22px 0}
figure{margin:12px 0 10px}
figure img{width:100%;height:auto;display:block;border-radius:8px;border:1px solid var(--line);background:#fcfcfb}
.hero img{max-width:560px;margin:0 auto}
.dl{display:inline-block;font-size:14px;color:var(--accent);text-decoration:none;border:1px solid var(--line);
border-radius:8px;padding:5px 12px;margin:4px 0 6px}
.dl:hover{background:var(--chip)}
details{margin-top:8px}
summary{cursor:pointer;color:var(--ink2);font-size:14px}
.tw{overflow-x:auto;margin-top:8px}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{padding:6px 10px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}
th{color:var(--ink2);font-weight:600}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
ul{color:var(--ink2);padding-left:20px}
li{margin:4px 0}
code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:13px;background:var(--chip);
padding:1px 5px;border-radius:4px}
pre{background:var(--chip);border:1px solid var(--line);border-radius:8px;padding:12px;overflow-x:auto;font-size:12.5px;line-height:1.5}
footer{color:var(--muted);font-size:13px;margin-top:32px}
"""


def build_html(pngs):
    h = hero_numbers()
    t = tables()

    def fig_block(key, title, caption, tab_key, alt):
        name = os.path.basename(pngs[key])
        data = b64(pngs[key])
        return f"""
<section class="card{' hero' if key == 'hero' else ''}">
  <h2>{title}</h2>
  <p>{caption}</p>
  <figure><img alt="{html.escape(alt)}" src="data:image/png;base64,{data}"></figure>
  <a class="dl" download="{name}" href="data:image/png;base64,{data}">Download {name}</a>
  {f'<details><summary>Data table</summary>{t[tab_key]}</details>' if tab_key else ''}
</section>"""

    blocks = [
        fig_block("hero", "Summary card",
                  "Lead image: the four headline numbers of QEngine v1.0 over the best vLLM "
                  "configuration we could run.", None,
                  f"Summary: {h['pf_x']:.1f}x faster 200K prefill, {h['c1_lo']:.1f}-{h['c1_hi']:.1f}x "
                  f"single-user speed, {h['gp_q']:.0f}% vs {h['gp_v']:.0f}% of chat requests within SLA at 2 req/s"),
        fig_block("throughput", "Throughput vs concurrency",
                  "Output tokens per second at 1, 8, 16 and 32 concurrent requests. The advantage "
                  "is largest for a single user (2.2–2.8×) and narrows as batches fill up "
                  f"(1.2–1.4× at 32). The gap is smallest on agent c=32 ({speedup('agent', 32):.2f}×).",
                  "throughput", "Grouped bar charts of output tokens per second"),
        fig_block("prefill", "Long-prompt prefill",
                  f"Time to first token for a single new prompt of 1K to 200K tokens. At 200K "
                  f"tokens QEngine v1.0 answers in {h['t_now']:.1f} s, vLLM in {h['t_vl']:.1f} s "
                  f"({h['pf_x']:.1f}×). Cells: median TTFT · prompt tokens/s.",
                  "prefill", "TTFT versus prompt length and prompt tokens per second"),
        fig_block("goodput", "Goodput under a latency SLA",
                  "Share of requests that meet TTFT ≤ 3 s and TPOT ≤ 50 ms when requests arrive "
                  "randomly (Poisson) at a fixed rate. Chat: QEngine v1.0 keeps 100% at 2 req/s, "
                  f"vLLM {h['gp_v']:.0f}%. Agent: both hold 100% at 0.5 req/s. Cells: % in SLA · "
                  "goodput req/s.", "goodput", "Percentage of requests within SLA vs request rate"),
        fig_block("latency", "Single-user latency",
                  "One request at a time. Median time per output token and time to first token. "
                  "Cells: TPOT / TTFT (median).", "latency",
                  "Horizontal bars of TPOT and TTFT at concurrency 1"),
        fig_block("vllm_wins", "Where vLLM is ahead",
                  "Every point where a vLLM variant beat QEngine v1.0 on a median latency metric. "
                  "At RAG c=8 vLLM's chunked prefill interleaves the 15K-token prefills with "
                  "decoding: the first token arrives sooner (12.3 s vs 22.3 s), but each following "
                  "token takes 115–165 ms instead of 9.6 ms. The other three are 4–19% margins. "
                  "Cells: TTFT p50 (ms) / TPOT p50 (ms).", "wins",
                  "Bars of the points where vLLM leads"),
    ]

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>QEngine vs vLLM</title>
<style>{CSS}</style></head>
<body><main>
<h1>QEngine v1.0 vs vLLM 0.28 on RTX 3090s</h1>
<p class="lede">Serving benchmark of a custom C++/CUDA inference engine (QEngine) against vLLM, on the
same Qwen3.8-27B INT4 checkpoint, the same OpenAI-compatible chat API and the same benchmark
client (<code>vllm bench serve</code>), with identical prompts and seeds for every engine.
Measured 21–22 October 2026.</p>
<div class="hw"><span>Model: Qwen3.8-27B INT4 (same checkpoint)</span>
<span>QEngine: 3× RTX 3090, tensor parallel 3</span>
<span>vLLM: 2× RTX 3090, tensor parallel 2</span><span>Client: vllm bench serve 0.28.0</span></div>
<div class="key"><span><i style="background:{C_V10}"></i>QEngine v1.0 (1be10df, 2026-10-21)</span>
<span><i style="background:{C_V01}"></i>QEngine v0.1 (1c3f603, 2026-10-18)</span>
<span><i style="background:{C_VLLM}"></i>vLLM 0.28 (best config)</span></div>
{''.join(blocks)}
<section class="card">
<h2>Methodology</h2>
<h3>Engines</h3>
<ul>
<li><b>QEngine v1.0</b>: master <code>1be10df</code> (2026-10-21). <b>QEngine v0.1</b>: master
<code>1c3f603</code> (2026-10-18). Both on 3× RTX 3090 with tensor parallel 3, 32 sequences,
262,144 context, built-in MTP head (adaptive, up to 7 draft tokens).</li>
<li><b>vLLM 0.28 (best config)</b>: vLLM 0.28.0 with the community club-3090 “qwen3.8-27b
dual-fast” recipe, ported flag for flag, plus the two club-3090 runtime patches that fix MTP
crashes under concurrency (FlashInfer decode-plan pinned-buffer race; GDN + MTP async
spec-decode ordering). W4A8 Marlin, fp8 KV cache, prefix caching, chunked prefill, 32 sequences,
262,144 context. Measured twice: with MTP speculative decoding (4 draft tokens) and without;
every chart uses, per point, the better of the two (named in each chart's footnote and table).
vLLM 0.28.0 is the newest release that builds with this host's toolchain.</li>
<li><b>GPU count.</b> vLLM runs on 2 GPUs, QEngine on 3. Tensor parallel 3 is not possible in
vLLM for this model: its 4 KV heads and 16 DeltaNet key heads are not divisible by 3. TP=2 is the
strongest vLLM layout available on this machine and the one the community recipe uses. All
numbers are whole-system throughput and latency, not per GPU.</li>
</ul>
<h3>Client and workloads</h3>
<ul>
<li><code>vllm bench serve --backend openai-chat --endpoint /v1/chat/completions</code>, sampling
sent explicitly (<code>--temperature 0.7 --top-p 0.8 --top-k 20</code>), thinking off.
<code>--ignore-eos</code> on synthetic datasets (fixed output length); ShareGPT uses its real
lengths. Each point has its own seed, identical across engines, so no point reuses another's
prompts. A short unsaved warm-up precedes every point. Engines ran one at a time, each on a
freshly started server.</li>
<li><b>Classic</b>: <code>--max-concurrency C --request-rate inf</code>, C ∈ {{1, 8, 16, 32}}.
short = 1024 in / 512 out; agent = 500 / 2000; rag16k = 15000 / 128 (c=1, 8);
ShareGPT V3 first user turns.</li>
<li><b>Goodput</b>: Poisson arrivals (<code>--request-rate R --burstiness 1.0</code>), no client
concurrency cap. <b>SLA: TTFT ≤ 3000 ms and TPOT ≤ 50 ms per request</b>
(<code>--goodput ttft:3000 tpot:50</code>). chat = 1024 shared + 1024 unique / 256 out, 180 s of
arrivals per rate; long16k = 12288 + 4096 / 512, 240 s; agent = 4096 + 2048 / 1024, 240 s;
lengths vary ±50%.</li>
<li><b>Prefill</b>: one unique prompt, <code>--random-output-len 1 --max-concurrency 1</code>,
lengths 1K–200K, 1 warm-up + 3 measured runs with different seeds (no prefix-cache hits);
median TTFT reported. TTFT is measured client-side.</li>
<li>TPOT follows the vLLM definition: (end-to-end latency − TTFT) / (output tokens − 1), per
request; medians shown.</li>
</ul>
<h3>Stability check</h3>
<p>At the end of the session, short c=1 and c=32 were repeated on a freshly started server for
every engine with the same prompts. All engines reproduced within ±5% at c=1 and ±2.2% at c=32
(QEngine v1.0: −2.6% / +0.3%).</p>
{t['drift']}
<h3>Caveats</h3>
<ul>
<li>One failed request in the whole run: vLLM MTP, goodput chat at 1 req/s (179 of 180 completed,
client-side disconnect, server stayed up). The no-spec variant is the vLLM best at that point.</li>
<li>QEngine v1.0 is slightly slower than v0.1 on short chat at c=1 (208 vs 218 tok/s,
reproduced in the stability check) and faster at every other point.</li>
<li>vLLM best in goodput is chosen by goodput req/s, so at agent 0.25 req/s it is the MTP run
(0.230 req/s, 97% in SLA) although no-spec had 100% in SLA at 0.228 req/s.</li>
<li>Different GPU counts (3 vs 2), see above.</li>
</ul>
</section>
<footer>{html.escape(FOOTER)} · Raw <code>vllm bench serve</code> JSON and logs for every run are
kept with the benchmark. Charts generated by <code>charts/make_charts.py</code> from the summary
CSVs.</footer>
</main></body></html>
"""


def main():
    pngs = {
        "hero": chart_hero(),
        "throughput": chart_throughput(),
        "prefill": chart_prefill(),
        "goodput": chart_goodput(),
        "latency": chart_latency(),
        "vllm_wins": chart_vllm_wins(),
    }
    with open(os.path.join(HERE, "index.html"), "w") as f:
        f.write(build_html(pngs))
    for p in pngs.values():
        print(p)
    print(os.path.join(HERE, "index.html"))


if __name__ == "__main__":
    main()
