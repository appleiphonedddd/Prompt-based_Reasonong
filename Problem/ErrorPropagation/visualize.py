"""
Three-figure visualization for the Error Propagation experiment.

Figure 1 — Bar chart:
    Final accuracy by injection position (clean / early / mid / late).
    Proves that even a single injected error kills the final answer.

Figure 2 — Heatmap:
    Per-problem, per-step arithmetic correctness from natural CoT traces.
    Shows the "once wrong, stays wrong" cascade pattern visually.

Figure 3 — Line chart (Propagation Rate):
    P(final answer wrong | error injected at relative step position X).
    The curve is expected to stay high across all positions,
    with the earliest injections causing the most damage.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from tracer import StepTrace
from injector import InjectionResult


# ── Theme ──────────────────────────────────────────────────────────────────────

_BG   = "#0f1117"
_CARD = "#1a1d27"
_GRID = "#2a2d3a"

_C = {
    "correct":   "#2ecc71",
    "wrong":     "#e74c3c",
    "unknown":   "#4a5568",
    "clean":     "#3498db",
    "early":     "#e74c3c",
    "mid":       "#e67e22",
    "late":      "#9b59b6",
    "highlight": "#f39c12",
    "text":      "#ecf0f1",
    "subtext":   "#95a5a6",
}

_POS_ORDER  = ["clean", "early", "mid", "late"]
_POS_LABELS = ["Clean\n(no injection)", "Early\n(~25% through)", "Mid\n(~50%)", "Late\n(~75%)"]


def _style(ax: plt.Axes, title: str = "") -> None:
    ax.set_facecolor(_CARD)
    for spine in ax.spines.values():
        spine.set_edgecolor(_GRID)
    ax.tick_params(colors=_C["text"], labelsize=9)
    ax.xaxis.label.set_color(_C["text"])
    ax.yaxis.label.set_color(_C["text"])
    if title:
        ax.set_title(title, color=_C["text"], fontsize=11, fontweight="bold", pad=10)


# ── Figure 1: Accuracy by injection position ───────────────────────────────────

def _fig1_accuracy_bar(
    ax: plt.Axes,
    results: List[InjectionResult],
) -> None:
    _style(ax, "Fig 1 — Final Accuracy by Injection Position")

    acc: dict[str, list[bool]] = {p: [] for p in _POS_ORDER}
    for r in results:
        if r.position in acc:
            acc[r.position].append(r.final_correct)

    accs  = [100 * sum(v) / len(v) if v else 0.0 for v in acc.values()]
    counts = [len(v) for v in acc.values()]
    colors = [_C[p] for p in _POS_ORDER]
    xs = range(len(_POS_ORDER))

    bars = ax.bar(xs, accs, color=colors, width=0.55, edgecolor="#ffffff18", linewidth=0.8)

    for bar, acc_val, n in zip(bars, accs, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{acc_val:.0f}%\n(n={n})",
            ha="center", va="bottom", color=_C["text"], fontsize=10, fontweight="bold",
        )

    # Annotate the drop from clean to early
    if accs[0] > 0 and accs[1] < accs[0]:
        drop = accs[0] - accs[1]
        ax.annotate(
            f"↓ {drop:.0f}pp drop",
            xy=(0.5, (accs[0] + accs[1]) / 2),
            color=_C["highlight"], fontsize=9, fontweight="bold", ha="center",
        )

    ax.set_xticks(list(xs))
    ax.set_xticklabels(_POS_LABELS, color=_C["text"], fontsize=9)
    ax.set_ylabel("Final Answer Accuracy (%)", color=_C["text"])
    ax.set_ylim(0, 120)
    ax.axhline(accs[0], color=_C["clean"], linestyle="--", alpha=0.4, lw=1.2)
    ax.text(3.4, accs[0] + 2, "Clean baseline", color=_C["clean"], fontsize=8, alpha=0.7)


# ── Figure 2: Step correctness heatmap ────────────────────────────────────────

def _fig2_heatmap(
    ax: plt.Axes,
    traces: List[StepTrace],
    max_steps: int = 8,
) -> None:
    _style(ax, "Fig 2 — Step-Level Correctness Heatmap")

    n_problems = len(traces)
    # Build matrix: rows=problems, cols=steps
    matrix = np.full((n_problems, max_steps), np.nan)  # nan = step doesn't exist

    for row, trace in enumerate(traces):
        for step in trace.steps[:max_steps]:
            col = step.index - 1   # 0-based
            if col < max_steps:
                if step.is_correct is True:
                    matrix[row, col] = 1.0
                elif step.is_correct is False:
                    matrix[row, col] = 0.0
                else:
                    matrix[row, col] = 0.5   # unknown / no arithmetic

    # Custom colormap: red=0, grey=0.5, green=1
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "ep",
        [_C["wrong"], _C["unknown"], _C["correct"]],
        N=3,
    )

    masked = np.ma.masked_where(np.isnan(matrix), matrix)
    im = ax.imshow(masked, cmap=cmap, aspect="auto", vmin=0, vmax=1,
                   interpolation="nearest")

    # Mark final answer correct/wrong with a symbol in the rightmost +1 column
    for row, trace in enumerate(traces):
        symbol = "★" if trace.final_correct else "✗"
        color  = _C["correct"] if trace.final_correct else _C["wrong"]
        ax.text(
            max_steps - 0.35, row, symbol,
            color=color, fontsize=9, ha="right", va="center", fontweight="bold",
        )

    ax.set_xticks(range(max_steps))
    ax.set_xticklabels([f"Step {i+1}" for i in range(max_steps)],
                       color=_C["text"], fontsize=8)
    ax.set_yticks(range(n_problems))
    ax.set_yticklabels(
        [f"Q{t.problem_index:02d} {'✓' if t.final_correct else '✗'}" for t in traces],
        color=_C["text"], fontsize=7.5,
    )
    ax.set_xlabel("Reasoning Step", color=_C["text"])
    ax.set_ylabel("Problem", color=_C["text"])

    # Legend patches
    legend = [
        mpatches.Patch(color=_C["correct"], label="Arithmetic correct"),
        mpatches.Patch(color=_C["wrong"],   label="Arithmetic wrong"),
        mpatches.Patch(color=_C["unknown"], label="No arithmetic to verify"),
    ]
    ax.legend(handles=legend, loc="lower right", facecolor=_CARD,
              edgecolor=_GRID, labelcolor=_C["text"], fontsize=7)


# ── Figure 3: Propagation rate line chart ─────────────────────────────────────

def _fig3_propagation(
    ax: plt.Axes,
    results: List[InjectionResult],
) -> None:
    _style(ax, "Fig 3 — Error Propagation Rate")

    # Group by position and compute P(wrong | injected at position X)
    wrong_rate: dict[str, float] = {}
    sample_n: dict[str, int] = {}

    for pos in _POS_ORDER:
        subset = [r for r in results if r.position == pos]
        if not subset:
            continue
        wrong_rate[pos] = sum(1 for r in subset if not r.final_correct) / len(subset)
        sample_n[pos] = len(subset)

    # Map positions to x-axis values (relative chain position)
    x_map = {"clean": 0.0, "early": 0.25, "mid": 0.50, "late": 0.75}
    xs = [x_map[p] for p in _POS_ORDER if p in wrong_rate]
    ys = [wrong_rate[p] for p in _POS_ORDER if p in wrong_rate]
    ns = [sample_n[p] for p in _POS_ORDER if p in wrong_rate]

    if not xs:
        return

    # Shaded area + line
    ax.fill_between(xs, ys, alpha=0.15, color=_C["wrong"])
    ax.plot(xs, ys, "o-", color=_C["wrong"], lw=2.5, ms=9, zorder=3)

    for x, y, pos, n in zip(xs, ys, [p for p in _POS_ORDER if p in wrong_rate], ns):
        ax.scatter(x, y, s=110, color=_C[pos], zorder=5, edgecolors="white", lw=1.2)
        ax.annotate(
            f"{y * 100:.0f}%\n(n={n})",
            xy=(x, y), xytext=(0, 14), textcoords="offset points",
            ha="center", color=_C["text"], fontsize=9, fontweight="bold",
        )

    # Clean baseline (should be near 0)
    if "clean" in wrong_rate:
        ax.axhline(
            wrong_rate["clean"], color=_C["correct"], linestyle="--",
            alpha=0.5, lw=1.2, label=f"Clean ({wrong_rate['clean']*100:.0f}% wrong)",
        )

    ax.set_xticks([0.0, 0.25, 0.50, 0.75])
    ax.set_xticklabels(["Clean", "Early\n(~25%)", "Mid\n(~50%)", "Late\n(~75%)"],
                       color=_C["text"], fontsize=9)
    ax.set_ylabel("P(final answer wrong)", color=_C["text"])
    ax.set_ylim(-0.05, 1.15)
    ax.legend(facecolor=_CARD, edgecolor=_GRID, labelcolor=_C["text"], fontsize=8)


# ── Main entry ─────────────────────────────────────────────────────────────────

def plot_all(
    traces: List[StepTrace],
    results: List[InjectionResult],
    out_dir: Path,
    model_name: str = "Qwen",
    task_name: str = "BBH Geometric Shapes",
    n_problems: int = 0,
) -> None:
    """Render all three figures as a single PNG and individual PNGs.

    Args:
        traces:     Step traces from the tracer (for Fig 2).
        results:    Injection results (for Fig 1 and Fig 3).
        out_dir:    Directory to write PNG files.
        model_name: Shown in the figure title.
        task_name:  Dataset/task label shown in the figure title.
        n_problems: Total problems run (for subtitle).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    n_correct   = sum(1 for t in traces if t.final_correct)
    n_total     = len(traces)
    n_injected  = sum(1 for r in results if r.position != "clean")
    subtitle = (
        f"{task_name}  —  {model_name}  —  "
        f"{n_total} problems ({n_correct} correct)  —  "
        f"{n_injected} injections"
    )

    fig, axes = plt.subplots(
        1, 3,
        figsize=(21, 7),
        facecolor=_BG,
        gridspec_kw={"wspace": 0.35},
    )
    fig.subplots_adjust(left=0.06, right=0.97, top=0.85, bottom=0.14)

    _fig1_accuracy_bar(axes[0], results)
    _fig2_heatmap(axes[1], traces)
    _fig3_propagation(axes[2], results)

    fig.suptitle(
        "Error Propagation in Chain-of-Thought Reasoning\n" + subtitle,
        color=_C["text"], fontsize=13, fontweight="bold", y=0.97,
    )

    combined_path = out_dir / "error_propagation.png"
    fig.savefig(combined_path, dpi=150, bbox_inches="tight", facecolor=_BG)
    plt.close(fig)
    print(f"[viz] Combined figure → {combined_path}")

    # Also save each figure individually for easy reference
    for idx, (render_fn, args, name) in enumerate([
        (_fig1_accuracy_bar, (results,),           "fig1_accuracy_bar"),
        (_fig2_heatmap,      (traces,),             "fig2_heatmap"),
        (_fig3_propagation,  (results,),            "fig3_propagation_rate"),
    ]):
        fig_single, ax_single = plt.subplots(figsize=(8, 6), facecolor=_BG)
        render_fn(ax_single, *args)
        fig_single.tight_layout()
        path = out_dir / f"{name}.png"
        fig_single.savefig(path, dpi=150, bbox_inches="tight", facecolor=_BG)
        plt.close(fig_single)
        print(f"[viz] {name} → {path}")
