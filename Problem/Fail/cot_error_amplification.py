"""
CoT/GoT Error Amplification Experiment: Code Execution Mental Simulation

Background
----------
Research shows CoT reasoning amplifies errors through three failure modes:
  1. Subtask decomposition errors -> every branch is wrong from the start (Generate)
  2. Subtask solving errors       -> correct branches get "diluted" in Aggregate
  3. Final synthesis errors       -> Refine corrupts an already-correct answer

Task: CRUXEval (predict Python function return values)
  - LLMs are inherently bad at mentally simulating code execution
  - Reasoning chains give the model more opportunities to turn correct into wrong

Logic
-----
- Run GoT(k=3 branches, keep=2, refine=1) on CRUXEval problems
- After each phase (Generate / Aggregate / Refine) evaluate every thought
  node with the CRUXEval scorer
- Stop once MIN_AMPLIFICATION "had a correct Generate but still wrong final"
  cases have been collected
- Save a PNG visualisation

Usage
-----
  cd /home/infor/Code/Prompt-based-Reasoning
  python Problem/Fail/cot_error_amplification.py
"""

from __future__ import annotations

import ast
import os
import sys
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is on sys.path
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import math

import matplotlib
matplotlib.use("Agg")           # headless - save to PNG at the end
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np

from benchmark.CRUXEval.cruxeval import (
    CRUXEval,
    _extract_python_literal,
    _eval_literal,
)
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from baseline.GoT.got import GoT
from models.base import BaseLLM, LLMResponse

# ── Config ────────────────────────────────────────────────────────────────────
MAX_QUESTIONS      = 20   # maximum number of problems to run
MIN_AMPLIFICATION  = 3    # stop after collecting this many amplification cases
GOT_BRANCHES       = 5    # number of Generate branches (fan-out)
GOT_KEEP           = 3    # KeepBest N (kept paths fed into Aggregate)
GOT_REFINE         = 2    # Refine rounds (creates 2 sequential Refine nodes)
MODEL              = "Qwen/Qwen2.5-3B-Instruct"
MODEL_PATH         = (
    "/home/infor/.cache/huggingface/hub/"
    "models--Qwen--Qwen2.5-3B-Instruct/snapshots/"
    "aa8e72537993ba99e69dfaafa59ed015b17504d1"
)


# ── Local HuggingFace model ───────────────────────────────────────────────────

class HFClient(BaseLLM):
    """Local HuggingFace model client using AutoTokenizer + AutoModelForCausalLM."""

    def __init__(self, model_id: str = MODEL) -> None:
        super().__init__(api_key="local", model=model_id)
        # Load from local snapshot path to avoid HF hub lock-file permission issues
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        self.hf_model  = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            dtype=torch.float16,
            device_map="auto",
        )
        self.hf_model.eval()

    def generate(self, prompt: str, temperature: float = 0.0) -> LLMResponse:
        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.hf_model.device)

        input_len = inputs["input_ids"].shape[-1]

        gen_kwargs: dict = dict(max_new_tokens=512)
        if temperature > 0.0:
            gen_kwargs.update(do_sample=True, temperature=temperature)
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            outputs = self.hf_model.generate(**inputs, **gen_kwargs)

        new_tokens = outputs[0][input_len:]
        content    = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return LLMResponse(
            content=content,
            model_name=self.model,
            input_tokens=input_len,
            output_tokens=len(new_tokens),
        )


OUT_PNG       = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cot_error_amplification.png")
OUT_GRAPH_PNG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cot_thought_graphs.png")


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class PhaseCheck:
    """Evaluation result for a single GoT thought node."""
    phase:      str
    thought_id: str
    answer:     str        # Python literal extracted from content
    content:    str        # raw LLM response (truncated)
    is_correct: bool
    score:      float = 0.0
    parent_ids: List[str] = field(default_factory=list)
    status:     str = ""   # KEPT / PRUNED / AGGREGATED / REFINED / …


@dataclass
class QuestionTrace:
    """Full GoT trace for one problem."""
    qid:           str
    code:          str
    inp:           str
    ground_truth:  str
    final_answer:  str
    final_correct: bool
    phases:        List[PhaseCheck] = field(default_factory=list)
    # "correct" | "generate" | "aggregate" | "refine" | "other"
    failure_mode:  str = "other"
    elapsed_sec:   float = 0.0


# ── Evaluation helpers ────────────────────────────────────────────────────────

def check_correct(text: str, expected: str) -> bool:
    """Extract a Python literal from text and compare to expected."""
    extracted = _extract_python_literal(text)
    pred_val, pred_ok = _eval_literal(extracted)
    gt_val,   gt_ok   = _eval_literal(expected)
    if pred_ok and gt_ok:
        return pred_val == gt_val
    return extracted.strip() == expected.strip()


def classify_failure(phases: List[PhaseCheck], final_correct: bool) -> str:
    """
    Classify which phase introduced or amplified the error.

      generate  -> all Generate thoughts wrong (LLM cannot simulate execution at all)
      aggregate -> some Generate correct, but Aggregate output is wrong  (amplification)
      refine    -> Aggregate correct, but Refine corrupted it            (amplification)
      correct   -> final answer is correct
      other     -> no clear pattern
    """
    if final_correct:
        return "correct"

    gen_checks = [p for p in phases if p.phase == "generate"]
    agg_checks = [p for p in phases if p.phase == "aggregate"]
    ref_checks = [p for p in phases if p.phase == "refine"]

    any_gen_ok = any(p.is_correct for p in gen_checks)
    any_agg_ok = any(p.is_correct for p in agg_checks)

    if not any_gen_ok:
        return "generate"
    if agg_checks and not any_agg_ok:
        return "aggregate"      # correct branch was discarded / overruled
    if ref_checks and not any(p.is_correct for p in ref_checks):
        return "refine"         # Refine turned a correct answer wrong
    return "other"


# ── Graph analysis ────────────────────────────────────────────────────────────

def analyze_graph(snapshot: Dict[str, Any], expected: str) -> List[PhaseCheck]:
    """Check every thought node in the GoT graph_snapshot for correctness."""
    results: List[PhaseCheck] = []
    for tid, data in snapshot.items():
        meta       = data.get("metadata", {})
        phase      = meta.get("phase", "unknown")
        content    = data.get("content", "")
        is_correct = check_correct(content, expected)
        extracted  = _extract_python_literal(content)

        results.append(PhaseCheck(
            phase      = phase,
            thought_id = tid,
            answer     = extracted,
            content    = content[:200],
            is_correct = is_correct,
            score      = data.get("score", 0.0),
            parent_ids = data.get("parent_ids", []),
            status     = data.get("status", ""),
        ))
    return results


# ── Experiment ────────────────────────────────────────────────────────────────

def run_experiment() -> List[QuestionTrace]:
    print("=" * 64)
    print("  CoT / GoT Error Amplification: Code Execution Mental Simulation")
    print("=" * 64)
    print(f"  Model: {MODEL}  |  branches={GOT_BRANCHES}  keep={GOT_KEEP}  refine={GOT_REFINE}")
    print(f"  Up to {MAX_QUESTIONS} problems; stop after {MIN_AMPLIFICATION} amplification cases\n")

    dataset = CRUXEval()
    dataset.load_dataset()

    llm = HFClient(model_id=MODEL)
    got = GoT(llm,
              num_branches      = GOT_BRANCHES,
              keep_best         = GOT_KEEP,
              refine_rounds     = GOT_REFINE,
              gen_temperature   = 0.7,
              score_temperature = 0.0,
              agg_temperature   = 0.3)

    traces:    List[QuestionTrace] = []
    amp_count: int = 0

    for i in range(MAX_QUESTIONS):
        problem  = dataset.get_problem(i)
        gt       = problem.ground_truth
        expected = gt["expected_output"]
        qid      = gt["id"]

        print(f"--- Q{i:02d}  {qid}  expected={expected!r} " + "-" * 20)

        t0 = time.time()
        try:
            response = got.run(
                question      = problem.question,
                system_prompt = dataset.get_system_prompt(),
                instruction   = dataset.get_instruction(),
            )
        except Exception as exc:
            print(f"  [ERROR] {exc}")
            continue
        elapsed = time.time() - t0

        snapshot = response.metadata.get("graph_snapshot", {})
        phases   = analyze_graph(snapshot, expected)

        final_ans     = response.final_answer
        final_correct = check_correct(final_ans, expected)
        mode          = classify_failure(phases, final_correct)

        phase_order = {"generate": 0, "aggregate": 1, "refine": 2}
        for p in sorted(phases, key=lambda x: (phase_order.get(x.phase, 9), x.thought_id)):
            icon = "+" if p.is_correct else "x"
            print(f"  [{p.phase:10s}] {icon}  extracted={p.answer!r:25s}  score={p.score:.2f}")

        final_icon = "+" if final_correct else "x"
        print(f"  [FINAL     ] {final_icon}  answer={final_ans!r}  -> mode={mode}  ({elapsed:.1f}s)")

        traces.append(QuestionTrace(
            qid           = qid,
            code          = gt["code"],
            inp           = gt["input"],
            ground_truth  = expected,
            final_answer  = final_ans,
            final_correct = final_correct,
            phases        = phases,
            failure_mode  = mode,
            elapsed_sec   = elapsed,
        ))

        if mode in ("aggregate", "refine"):
            amp_count += 1
            print(f"  *** ERROR AMPLIFICATION #{amp_count}  mode={mode} ***")
            if amp_count >= MIN_AMPLIFICATION:
                print(f"\n  [STOP] Collected {MIN_AMPLIFICATION} amplification cases.")
                break

    print(f"\nTotal: {len(traces)} problems  |  amplification cases: {amp_count}")
    return traces


# ── Visualisation ─────────────────────────────────────────────────────────────

_BG   = "#0f1117"
_CARD = "#1a1d27"
_GRID = "#2a2d3a"

_COL = {
    "correct":   "#2ecc71",
    "generate":  "#e74c3c",
    "aggregate": "#e67e22",
    "refine":    "#9b59b6",
    "other":     "#7f8c8d",
    "gen_phase": "#3498db",
    "agg_phase": "#e67e22",
    "ref_phase": "#9b59b6",
}


def _ax_style(ax: plt.Axes, title: str = "") -> None:
    ax.set_facecolor(_CARD)
    for spine in ax.spines.values():
        spine.set_edgecolor(_GRID)
    ax.tick_params(colors="white", labelsize=8)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    if title:
        ax.set_title(title, color="white", fontsize=10, fontweight="bold", pad=8)


def visualize(traces: List[QuestionTrace]) -> None:
    if not traces:
        print("[WARN] No data to plot.")
        return

    # --- statistics ---
    mode_counts: Dict[str, int] = {m: 0 for m in ("correct", "generate", "aggregate", "refine", "other")}
    for t in traces:
        mode_counts[t.failure_mode] = mode_counts.get(t.failure_mode, 0) + 1

    phase_correct: Dict[str, List[bool]] = {"generate": [], "aggregate": [], "refine": []}
    for t in traces:
        for p in t.phases:
            if p.phase in phase_correct:
                phase_correct[p.phase].append(p.is_correct)

    phase_acc = {
        ph: (sum(v) / len(v) * 100 if v else 0.0)
        for ph, v in phase_correct.items()
    }

    amp_cases = [t for t in traces if t.failure_mode in ("aggregate", "refine")]
    n_total   = len(traces)
    n_wrong   = sum(1 for t in traces if not t.final_correct)
    n_amp     = len(amp_cases)

    # --- layout ---
    fig = plt.figure(figsize=(20, 13), facecolor=_BG)
    gs  = fig.add_gridspec(
        2, 3,
        height_ratios=[1, 1.1],
        hspace=0.42, wspace=0.35,
        left=0.06, right=0.97,
        top=0.88, bottom=0.05,
    )
    ax_bar   = fig.add_subplot(gs[0, 0])
    ax_phase = fig.add_subplot(gs[0, 1])
    ax_swim  = fig.add_subplot(gs[0, 2])
    ax_det   = fig.add_subplot(gs[1, :])

    # --- Panel 1: failure mode bar chart ---
    _ax_style(ax_bar, "1  Failure Mode Distribution")
    modes  = ["correct", "generate", "aggregate", "refine", "other"]
    labels = ["Correct", "Fail:\nGenerate\n(can't simulate)", "Fail:\nAggregate\n(amplified *)", "Fail:\nRefine\n(amplified *)", "Other"]
    vals   = [mode_counts.get(m, 0) for m in modes]
    cols   = [_COL[m] for m in modes]

    bars = ax_bar.bar(range(len(modes)), vals, color=cols, width=0.6, edgecolor="#ffffff22", linewidth=0.8)
    for bar, v in zip(bars, vals):
        if v:
            ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                        str(v), ha="center", va="bottom", color="white", fontsize=11, fontweight="bold")

    ax_bar.set_xticks(range(len(modes)))
    ax_bar.set_xticklabels(labels, color="white", fontsize=7.5)
    ax_bar.set_ylabel("Count", color="white")
    ax_bar.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax_bar.set_ylim(0, max(vals) + 1.5 if vals else 5)

    for idx, m in enumerate(modes):
        if m in ("aggregate", "refine") and vals[idx] > 0:
            ax_bar.annotate("", xy=(idx, vals[idx] + 0.8), xytext=(idx, vals[idx] + 0.3),
                            arrowprops=dict(arrowstyle="->", color="#f39c12", lw=1.5))

    # --- Panel 2: per-phase accuracy line ---
    _ax_style(ax_phase, "2  Accuracy by Reasoning Phase  (drop = error amplification)")
    ph_order  = ["generate", "aggregate", "refine"]
    ph_labels = ["Phase 0\nGenerate", "Phase 3\nAggregate", "Phase 4\nRefine"]
    ph_accs   = [phase_acc.get(ph, 0) for ph in ph_order]
    ph_colors = [_COL["gen_phase"], _COL["agg_phase"], _COL["ref_phase"]]

    ax_phase.plot(range(3), ph_accs, "o-", color="#e74c3c", lw=2.5, ms=10, zorder=3)
    ax_phase.fill_between(range(3), ph_accs, alpha=0.12, color="#e74c3c")

    for xi, (acc, col) in enumerate(zip(ph_accs, ph_colors)):
        if phase_correct[ph_order[xi]]:
            ax_phase.scatter(xi, acc, s=120, color=col, zorder=5, edgecolors="white", linewidth=1.2)
            ax_phase.annotate(f"{acc:.0f}%",
                              (xi, acc), xytext=(0, 14), textcoords="offset points",
                              ha="center", color="white", fontsize=12, fontweight="bold")

    if any(a > 0 for a in ph_accs):
        baseline = ph_accs[0]
        ax_phase.axhline(baseline, color="#2ecc71", linestyle="--", alpha=0.35, lw=1.2)
        ax_phase.text(2.45, baseline + 1.5, f"Generate baseline\n({baseline:.0f}%)",
                      color="#2ecc71", alpha=0.7, fontsize=7.5, ha="right")

    ax_phase.set_xticks(range(3))
    ax_phase.set_xticklabels(ph_labels, color="white", fontsize=9)
    ax_phase.set_ylabel("Thought accuracy (%)", color="white")
    ax_phase.set_ylim(0, 115)

    # --- Panel 3: per-question swimlane ---
    _ax_style(ax_swim, "3  Error Propagation per Question")
    n_show = min(10, len(traces))
    ph_x   = {"generate": 0, "aggregate": 1, "refine": 2, "final": 3}

    for qi, trace in enumerate(traces[:n_show]):
        by_phase: Dict[str, List[PhaseCheck]] = {"generate": [], "aggregate": [], "refine": []}
        for p in trace.phases:
            if p.phase in by_phase:
                by_phase[p.phase].append(p)

        prev_x, prev_ok = None, None

        for ph in ("generate", "aggregate", "refine"):
            checks = by_phase[ph]
            if not checks:
                continue
            majority_ok = sum(p.is_correct for p in checks) > len(checks) / 2
            x   = ph_x[ph]
            col = _COL["correct"] if majority_ok else _COL["generate"]
            ax_swim.scatter(x, qi, s=70, color=col, zorder=3, alpha=0.9)

            if prev_x is not None:
                line_col = _COL["aggregate"] if (prev_ok and not majority_ok) else col
                ax_swim.plot([prev_x, x], [qi, qi], color=line_col, alpha=0.55, lw=1.8, zorder=2)
            prev_x, prev_ok = x, majority_ok

        fc = _COL["correct"] if trace.final_correct else _COL["generate"]
        ax_swim.scatter(3, qi, s=100, marker="*", color=fc, zorder=4)
        ax_swim.text(-0.35, qi, f"Q{qi} [{trace.failure_mode[:3]}]",
                     ha="right", va="center", color="white", fontsize=7)

    ax_swim.set_xticks([0, 1, 2, 3])
    ax_swim.set_xticklabels(["Generate\n(Ph.0)", "Aggregate\n(Ph.3)", "Refine\n(Ph.4)", "Final"],
                            color="white", fontsize=8)
    ax_swim.set_yticks([])
    ax_swim.set_xlim(-0.6, 3.5)

    leg = [
        mpatches.Patch(color=_COL["correct"],   label="Correct"),
        mpatches.Patch(color=_COL["generate"],   label="Wrong"),
        mpatches.Patch(color=_COL["aggregate"],  label="Amplified (correct -> wrong)"),
    ]
    ax_swim.legend(handles=leg, loc="lower right", facecolor=_CARD,
                   edgecolor=_GRID, labelcolor="white", fontsize=7)

    # --- Panel 4: amplification case detail ---
    ax_det.set_facecolor(_CARD)
    ax_det.axis("off")
    for spine in ax_det.spines.values():
        spine.set_edgecolor(_GRID)

    if amp_cases:
        case = amp_cases[0]

        by_phase = {"generate": [], "aggregate": [], "refine": []}
        for p in case.phases:
            if p.phase in by_phase:
                by_phase[p.phase].append(p)

        title_txt = (f"4  Amplification Case Detail: {case.qid}  "
                     f"|  expected={case.ground_truth!r}  "
                     f"|  mode={case.failure_mode}")
        ax_det.text(0.5, 0.97, title_txt, transform=ax_det.transAxes,
                    color="white", fontsize=11, fontweight="bold", ha="center", va="top")

        code_short = case.code.strip().replace("\n", " | ")[:90]
        ax_det.text(0.5, 0.89, f"f({case.inp})   >>   {code_short}...",
                    transform=ax_det.transAxes,
                    color="#aaaaaa", fontsize=8, ha="center", va="top", style="italic")

        ph_xs         = {"generate": 0.14, "aggregate": 0.50, "refine": 0.76}
        node_pos: Dict[str, Tuple[float, float]] = {}
        phase_col_map = {"generate": _COL["gen_phase"], "aggregate": _COL["agg_phase"], "refine": _COL["ref_phase"]}

        for ph, checks in by_phase.items():
            if not checks:
                continue
            px       = ph_xs[ph]
            ph_label = {"generate": "Phase 0\nGenerate", "aggregate": "Phase 3\nAggregate", "refine": "Phase 4\nRefine"}[ph]
            ax_det.text(px, 0.82, ph_label, transform=ax_det.transAxes,
                        color=phase_col_map[ph], fontsize=9, ha="center", va="top", fontweight="bold")

            for j, p in enumerate(checks):
                y     = 0.62 - j * 0.22
                nc    = _COL["correct"] if p.is_correct else _COL["generate"]
                icon  = "+" if p.is_correct else "x"
                short = str(p.answer)[:18]
                box_kw = dict(boxstyle="round,pad=0.35", facecolor=nc, alpha=0.22, edgecolor=nc, linewidth=1.5)
                ax_det.text(px, y, f"{icon}  {short!r}",
                            transform=ax_det.transAxes, color="white", fontsize=9,
                            ha="center", va="center", bbox=box_kw)
                node_pos[p.thought_id] = (px, y)

        for ph, checks in by_phase.items():
            for p in checks:
                if p.thought_id not in node_pos:
                    continue
                for par_id in p.parent_ids:
                    if par_id in node_pos:
                        x1, y1 = node_pos[par_id]
                        x2, y2 = node_pos[p.thought_id]
                        ax_det.annotate(
                            "", xy=(x2 - 0.03, y2), xytext=(x1 + 0.03, y1),
                            xycoords="axes fraction", textcoords="axes fraction",
                            arrowprops=dict(arrowstyle="->", color="#888888", lw=1.3),
                        )

        fc2 = _COL["correct"] if case.final_correct else _COL["generate"]
        fi  = "+" if case.final_correct else "x"
        ax_det.text(0.94, 0.60,
                    f"FINAL {fi}\n{str(case.final_answer)[:16]!r}\n---\nExpected:\n{case.ground_truth!r}",
                    transform=ax_det.transAxes, color="white", fontsize=9,
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.5", facecolor=fc2, alpha=0.35, edgecolor=fc2, linewidth=2))

        mode_msg = {
            "aggregate": (
                "WARNING  Error Amplification  -  Failure Mode 2\n"
                "A correct thought existed after Generate, but Aggregate\n"
                "discarded or overruled it and produced the wrong answer."
            ),
            "refine": (
                "WARNING  Error Amplification  -  Failure Mode 3\n"
                "Aggregate produced the correct answer, but Refine\n"
                "introduced a new error and corrupted the final output."
            ),
        }.get(case.failure_mode, "")

        if mode_msg:
            ax_det.text(0.5, 0.12, mode_msg, transform=ax_det.transAxes,
                        color="#f39c12", fontsize=10, ha="center", va="center",
                        fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.6",
                                  facecolor="#2c1810", edgecolor="#f39c12", linewidth=2))
    else:
        ax_det.text(0.5, 0.5,
                    "4  No amplification cases found\n"
                    "(all errors occurred at Generate phase, or all answers were correct)",
                    transform=ax_det.transAxes,
                    color="#888888", fontsize=13, ha="center", va="center")

    fig.suptitle(
        f"LLM Code Execution Mental Simulation  x  GoT Chain-of-Thought Error Amplification\n"
        f"CRUXEval  -  {MODEL}  -  "
        f"k={GOT_BRANCHES} branches  keep={GOT_KEEP}  refine={GOT_REFINE}  -  "
        f"{n_total} problems  |  wrong: {n_wrong}/{n_total}  |  amplification cases: {n_amp}",
        color="white", fontsize=12, fontweight="bold", y=0.97,
    )

    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight", facecolor=_BG)
    print(f"\n[OK] Visualization saved: {OUT_PNG}")


# ── Thought-graph visualisation ───────────────────────────────────────────────

# X positions (axes fraction) for each phase column
_PH_X = {"generate": 0.16, "aggregate": 0.50, "refine": 0.76, "final": 0.96}
_PH_COL = {"generate": _COL["gen_phase"], "aggregate": _COL["agg_phase"], "refine": _COL["ref_phase"]}


def _node(ax: plt.Axes, x: float, y: float, p: PhaseCheck) -> None:
    """Draw a single thought node as a coloured rounded-rectangle with text.

    PRUNED nodes are drawn with low alpha and a dashed border so it is
    immediately obvious they never reach Aggregate.
    """
    pruned = p.status == "PRUNED"
    nc     = _COL["correct"] if p.is_correct else _COL["generate"]
    icon   = "+" if p.is_correct else "x"
    label  = f"{icon} {str(p.answer)[:13]!r}\nscore={p.score:.2f}"
    if pruned:
        label = f"[pruned]\n{label}"

    ax.text(
        x, y, label,
        transform=ax.transAxes,
        color="#aaaaaa" if pruned else "white",
        fontsize=5.0, ha="center", va="center",
        bbox=dict(
            boxstyle="round,pad=0.28",
            facecolor=nc, alpha=0.10 if pruned else 0.30,
            edgecolor=nc, linewidth=0.7 if pruned else 1.2,
            linestyle="dashed" if pruned else "solid",
        ),
    )


def _arrow(ax: plt.Axes, x1: float, y1: float, x2: float, y2: float) -> None:
    """Draw an arrow between two nodes (axes-fraction coords)."""
    ax.annotate(
        "", xy=(x2 - 0.055, y2), xytext=(x1 + 0.055, y1),
        xycoords="axes fraction", textcoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", color="#555566",
                        lw=0.8, mutation_scale=7),
    )


def _draw_question_graph(ax: plt.Axes, trace: QuestionTrace) -> None:
    """Draw one question's full GoT thought graph in *ax*."""
    ax.set_facecolor(_CARD)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    for spine in ax.spines.values():
        spine.set_edgecolor(_GRID)

    # Title bar
    mode_col = _COL.get(trace.failure_mode, "white")
    ax.text(0.5, 0.985, f"{trace.qid}  [{trace.failure_mode}]",
            transform=ax.transAxes, color=mode_col, fontsize=6.5,
            ha="center", va="top", fontweight="bold")
    exp_short = str(trace.ground_truth)[:22]
    ax.text(0.5, 0.925, f"expected: {exp_short}",
            transform=ax.transAxes, color="#888899", fontsize=5.5,
            ha="center", va="top")

    # ── Separate phases ───────────────────────────────────────────────────────
    by_phase: Dict[str, List[PhaseCheck]] = {"generate": [], "aggregate": [], "refine": []}
    for p in trace.phases:
        if p.phase in by_phase:
            by_phase[p.phase].append(p)

    # ── Dynamic x layout ──────────────────────────────────────────────────────
    # Columns: Generate | Aggregate | Refine×N | Final
    # Refine can have multiple sequential nodes (one per refine round).
    n_ref  = max(len(by_phase["refine"]), 1)
    # Distribute x positions evenly across [0.12, 0.94]
    phases_seq = ["generate", "aggregate"] + ["refine"] * n_ref + ["final"]
    n_cols  = len(phases_seq)
    xs      = [0.12 + i * (0.82 / (n_cols - 1)) for i in range(n_cols)]
    ph_xs_dyn: Dict[str, float] = {}
    ref_xs: List[float] = []
    for i, ph in enumerate(phases_seq):
        if ph == "refine":
            ref_xs.append(xs[i])
        else:
            ph_xs_dyn[ph] = xs[i]

    # ── Column headers ────────────────────────────────────────────────────────
    for ph, x in ph_xs_dyn.items():
        if ph == "final":
            ax.text(x, 0.870, "Final", transform=ax.transAxes,
                    color="white", fontsize=6, ha="center", va="top", fontweight="bold")
        else:
            ax.text(x, 0.870, ph.capitalize(), transform=ax.transAxes,
                    color=_PH_COL[ph], fontsize=6, ha="center", va="top", fontweight="bold")
    for ri, rx in enumerate(ref_xs):
        ax.text(rx, 0.870, f"Refine\n(r{ri+1})", transform=ax.transAxes,
                color=_PH_COL["refine"], fontsize=5.5, ha="center", va="top", fontweight="bold")

    # ── Generate nodes (evenly spaced vertically) ─────────────────────────────
    gen   = by_phase["generate"]
    n_g   = max(len(gen), 1)
    gen_ys = ([0.78 - i * (0.56 / (n_g - 1)) for i in range(n_g)]
              if n_g > 1 else [0.50])

    node_pos: Dict[str, Tuple[float, float]] = {}
    for j, p in enumerate(gen):
        y = gen_ys[j]
        _node(ax, ph_xs_dyn["generate"], y, p)
        node_pos[p.thought_id] = (ph_xs_dyn["generate"], y)

    # ── Aggregate node (centred) ───────────────────────────────────────────────
    for p in by_phase["aggregate"]:
        _node(ax, ph_xs_dyn["aggregate"], 0.50, p)
        node_pos[p.thought_id] = (ph_xs_dyn["aggregate"], 0.50)

    # ── Refine nodes (sequential chain, centred) ──────────────────────────────
    for ri, (p, rx) in enumerate(zip(by_phase["refine"], ref_xs)):
        _node(ax, rx, 0.50, p)
        node_pos[p.thought_id] = (rx, 0.50)

    # ── Arrows: follow parent_ids for all non-generate nodes ──────────────────
    for ph in ("aggregate", "refine"):
        for p in by_phase[ph]:
            if p.thought_id not in node_pos:
                continue
            x2, y2 = node_pos[p.thought_id]
            for par_id in p.parent_ids:
                if par_id in node_pos:
                    # Skip arrows from PRUNED generate thoughts to aggregate
                    par_check = next((q for q in trace.phases if q.thought_id == par_id), None)
                    if par_check and par_check.status == "PRUNED":
                        continue
                    x1, y1 = node_pos[par_id]
                    _arrow(ax, x1, y1, x2, y2)

    # ── Final answer box ──────────────────────────────────────────────────────
    fx    = ph_xs_dyn["final"]
    fc    = _COL["correct"] if trace.final_correct else _COL["generate"]
    fi    = "OK" if trace.final_correct else "FAIL"
    short = str(trace.final_answer)[:13]
    ax.text(fx, 0.50, f"{fi}\n{short!r}",
            transform=ax.transAxes, color="white", fontsize=5.5,
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.30", facecolor=fc,
                      alpha=0.40, edgecolor=fc, linewidth=1.5))

    # Arrow from last node to Final
    last_p = None
    for ph in ("refine", "aggregate", "generate"):
        if by_phase.get(ph):
            last_p = by_phase[ph][-1]
            break
    if last_p and last_p.thought_id in node_pos:
        x1, y1 = node_pos[last_p.thought_id]
        _arrow(ax, x1, y1, fx, 0.50)


def visualize_thought_graphs(traces: List[QuestionTrace]) -> None:
    """One subplot per question, each showing the full GoT thought graph."""
    if not traces:
        return

    ncols = 4
    nrows = math.ceil(len(traces) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 5.2, nrows * 3.8),
        facecolor=_BG,
    )
    # Normalise axes to always be a 2-D array
    if nrows == 1:
        axes = np.array([axes])
    if ncols == 1:
        axes = axes[:, np.newaxis]

    for idx, trace in enumerate(traces):
        r, c = divmod(idx, ncols)
        _draw_question_graph(axes[r, c], trace)

    # Hide unused subplots
    for idx in range(len(traces), nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    fig.suptitle(
        f"GoT Thought Graphs — CRUXEval  |  {MODEL}  |  "
        f"k={GOT_BRANCHES} branches  keep={GOT_KEEP}  refine={GOT_REFINE}  |  "
        f"{len(traces)} problems\n"
        f"Node colour:  green=correct   red=wrong   "
        f"(+) correct answer   (x) wrong answer",
        color="white", fontsize=10, fontweight="bold", y=1.01,
    )

    plt.tight_layout(pad=0.8)
    plt.savefig(OUT_GRAPH_PNG, dpi=150, bbox_inches="tight", facecolor=_BG)
    print(f"[OK] Thought-graph visualization saved: {OUT_GRAPH_PNG}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    traces = run_experiment()
    visualize(traces)
    visualize_thought_graphs(traces)
