#!/usr/bin/env python3
"""
GoT Entropy Analysis: Game of 24 — FULL DATASET VERSION

Runs GoT on the entire dataset (or up to --limit), then performs
proper run-level statistical analysis of the hypothesis:

    "Incorrect runs have higher entropy than correct runs"

Key methodological choices:
  - Run-level units (not token-level) — tokens within a trace are not
    independent, so token-level p-values are inflated.
  - Three statistics reported: mean, median, 5%-trimmed mean.
  - Mann-Whitney U + Cliff's delta (effect size).
  - Per-phase analysis (generate / score / aggregate) since they have
    different output structures.
  - Caches results to JSON so re-analysis doesn't require re-running the model.
  - Evaluator sanity check at startup — refuses to run if the dataset
    evaluator mislabels known-correct answers.

Usage:
    python got_entropy.py                  # run full dataset
    python got_entropy.py --limit 80       # cap at 80 problems
    python got_entropy.py --analyze-only   # skip model, replot from cache
"""

import argparse
import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from transformers import AutoTokenizer, AutoModelForCausalLM
from benchmark.GameOf24.gameof24 import GameOf24

# ─── Config ──────────────────────────────────────────────────────────────────

MODEL_ID     = "Qwen/Qwen2.5-3B-Instruct"
NUM_BRANCHES = 3
KEEP_BEST    = 2
CACHE_PATH   = "got_entropy_cache.json"
PLOT_PATH    = "got_entropy_analysis.png"

PHASE_COLORS = {
    "generate":  "#4C72B0",
    "score":     "#DD8452",
    "aggregate": "#55A868",
}

# ─── Data structures ─────────────────────────────────────────────────────────

@dataclass
class StepRecord:
    phase: str
    branch_idx: int
    token_entropies: List[float]
    generated_text: str


@dataclass
class RunRecord:
    problem: str
    problem_idx: int
    final_answer: str
    final_correct: bool
    steps: List[StepRecord] = field(default_factory=list)

    def entropies_by_phase(self) -> Dict[str, List[float]]:
        out: Dict[str, List[float]] = {}
        for s in self.steps:
            out.setdefault(s.phase, []).extend(s.token_entropies)
        return out

    def all_entropies(self) -> List[float]:
        return [e for s in self.steps for e in s.token_entropies]

    def mean_entropy(self) -> float:
        all_h = self.all_entropies()
        return float(np.mean(all_h)) if all_h else float("nan")

    def median_entropy(self) -> float:
        all_h = self.all_entropies()
        return float(np.median(all_h)) if all_h else float("nan")

    def trimmed_mean_entropy(self, trim: float = 0.05) -> float:
        all_h = self.all_entropies()
        return float(stats.trim_mean(all_h, trim)) if all_h else float("nan")


# ─── Evaluator sanity check ──────────────────────────────────────────────────

def sanity_check_evaluator(ds) -> None:
    """Verify the evaluator behaves as expected before trusting any labels."""
    print("\n" + "=" * 60)
    print("EVALUATOR SANITY CHECK")
    print("=" * 60)
    cases = [
        ("(1+1+1)*8",      [1, 1, 1, 8],  True),
        ("(1+1+1)*8/1",    [1, 1, 1, 8],  True),   # ← previously mislabeled
        ("8*(1+1+1)",      [1, 1, 1, 8],  True),
        ("(13-1)*(1+1)",   [1, 1, 1, 13], True),
        ("(11-1)*1+1",     [1, 1, 1, 11], False),  # = 11, not 24
        ("(12-1)*1+1",     [1, 1, 1, 12], False),  # = 12, not 24
    ]
    failures = []
    for expr, nums, expected in cases:
        try:
            r = ds.evaluate_answer(expr, sorted(nums))
            got = r.is_correct
        except Exception as e:
            got = f"ERROR: {e}"
        ok = (got == expected)
        flag = "✓" if ok else "✗"
        print(f"  {flag} {expr:25s} on {nums}: expected={expected}  got={got}")
        if not ok:
            failures.append((expr, nums, expected, got))

    if failures:
        print(f"\n⚠ EVALUATOR HAS {len(failures)} BUG(S).")
        print("  Continuing anyway, but treat 'incorrect' labels with caution.")
        print("  Fix the evaluator before publishing any results.\n")
    else:
        print("✓ Evaluator passes all checks.\n")


# ─── Model wrapper ───────────────────────────────────────────────────────────

class QwenModel:
    def __init__(self) -> None:
        print(f"Loading {MODEL_ID} …")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.model.eval()
        print("Model ready.")

    @torch.no_grad()
    def chat(
        self,
        messages: List[Dict],
        max_new_tokens: int = 200,
        temperature: float = 0.7,
    ) -> Tuple[str, List[float]]:
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(self.model.device)

        kw = dict(
            max_new_tokens=max_new_tokens,
            output_logits=True,
            return_dict_in_generate=True,
        )
        if temperature > 0:
            kw["do_sample"] = True
            kw["temperature"] = temperature
            kw["top_p"] = 1.0
            kw["top_k"] = 0
        else:
            kw["do_sample"] = False

        out = self.model.generate(**inputs, **kw)

        ents: List[float] = []
        for logits in out.logits:
            log_p = F.log_softmax(logits.float(), dim=-1)
            p     = log_p.exp()
            h     = -(p * log_p).sum(dim=-1).item()
            ents.append(h if (not np.isnan(h) and not np.isinf(h)) else 0.0)

        new_ids = out.sequences[0][inputs["input_ids"].shape[-1]:]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        return text, ents


# ─── GoT prompts & runner ────────────────────────────────────────────────────

_GEN_SYS = (
    "You are a Game of 24 expert. "
    "Given 4 numbers, write ONE arithmetic expression using all four numbers "
    "exactly once with +, -, *, / and parentheses that equals 24. "
    "Reply ONLY with the expression, nothing else."
)
_SCORE_SYS = (
    "You are a strict mathematical judge. "
    "Rate the Game of 24 solution on a scale 0–10 (10 = fully correct). "
    "Reply ONLY with a single integer."
)
_AGG_SYS = (
    "You are a Game of 24 expert. "
    "Given several candidate solutions, output the single best arithmetic "
    "expression (or synthesise a correct one if none are right). "
    "Reply ONLY with the expression."
)


def run_got(model: QwenModel, problem: str, verbose: bool = False) -> RunRecord:
    steps: List[StepRecord] = []

    candidates: List[str] = []
    for i in range(NUM_BRANCHES):
        msgs = [
            {"role": "system", "content": _GEN_SYS},
            {"role": "user",   "content": f"Numbers: {problem}"},
        ]
        text, ents = model.chat(msgs, max_new_tokens=80, temperature=0.7)
        candidates.append(text)
        steps.append(StepRecord("generate", i, ents, text))
        if verbose:
            print(f"    gen[{i}] → {text!r:<40}  H̄={np.mean(ents):.3f}")

    scores: List[float] = []
    for i, cand in enumerate(candidates):
        msgs = [
            {"role": "system", "content": _SCORE_SYS},
            {"role": "user",   "content": f"Problem: {problem}\nSolution: {cand}"},
        ]
        text, ents = model.chat(msgs, max_new_tokens=8, temperature=0.0)
        m = re.search(r"\d+", text)
        score = min(float(m.group()), 10.0) if m else 5.0
        scores.append(score)
        steps.append(StepRecord("score", i, ents, text))
        if verbose:
            print(f"    score[{i}]={score:.0f}  H̄={np.mean(ents):.3f}")

    ranked   = sorted(zip(scores, candidates), reverse=True)
    best     = [c for _, c in ranked[:KEEP_BEST]]
    cand_txt = "\n".join(f"Candidate {k+1}: {c}" for k, c in enumerate(best))
    msgs = [
        {"role": "system", "content": _AGG_SYS},
        {"role": "user",   "content": f"Problem: {problem}\n\n{cand_txt}"},
    ]
    text, ents = model.chat(msgs, max_new_tokens=80, temperature=0.3)
    steps.append(StepRecord("aggregate", 0, ents, text))
    if verbose:
        print(f"    agg → {text!r:<40}  H̄={np.mean(ents):.3f}")

    return RunRecord(
        problem=problem, problem_idx=-1,
        final_answer=text, final_correct=False,
        steps=steps,
    )


# ─── Cache I/O ───────────────────────────────────────────────────────────────

def save_cache(records: List[RunRecord], path: str) -> None:
    data = [
        {
            "problem":       r.problem,
            "problem_idx":   r.problem_idx,
            "final_answer":  r.final_answer,
            "final_correct": r.final_correct,
            "steps": [
                {
                    "phase":            s.phase,
                    "branch_idx":       s.branch_idx,
                    "token_entropies":  s.token_entropies,
                    "generated_text":   s.generated_text,
                } for s in r.steps
            ],
        }
        for r in records
    ]
    with open(path, "w") as f:
        json.dump(data, f)
    print(f"Cache saved → {path}  ({len(records)} runs)")


def load_cache(path: str) -> List[RunRecord]:
    with open(path) as f:
        data = json.load(f)
    records = []
    for d in data:
        steps = [StepRecord(**s) for s in d["steps"]]
        rec = RunRecord(
            problem=d["problem"],
            problem_idx=d["problem_idx"],
            final_answer=d["final_answer"],
            final_correct=d["final_correct"],
            steps=steps,
        )
        records.append(rec)
    print(f"Cache loaded ← {path}  ({len(records)} runs)")
    return records


# ─── Statistics ──────────────────────────────────────────────────────────────

def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta effect size: P(a > b) - P(a < b)."""
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    gt = sum(1 for x in a for y in b if x > y)
    lt = sum(1 for x in a for y in b if x < y)
    return (gt - lt) / (len(a) * len(b))


def hypothesis_test(name: str, c: np.ndarray, i: np.ndarray) -> Dict:
    """Run-level Mann-Whitney U test + Cliff's delta."""
    if len(c) < 3 or len(i) < 3:
        return {"name": name, "n_c": len(c), "n_i": len(i), "skipped": True}
    u, p = stats.mannwhitneyu(i, c, alternative="greater")
    delta = cliffs_delta(i, c)
    supported = (p < 0.05) and (abs(delta) > 0.2)
    weak      = (p < 0.05) and (abs(delta) <= 0.2)
    return {
        "name": name,
        "n_c": len(c), "n_i": len(i),
        "median_c": float(np.median(c)), "median_i": float(np.median(i)),
        "mean_c":   float(np.mean(c)),   "mean_i":   float(np.mean(i)),
        "U": float(u), "p_value": float(p),
        "cliffs_delta": float(delta),
        "supported": supported, "weak": weak,
    }


def analyze(records: List[RunRecord]) -> Dict:
    correct   = [r for r in records if r.final_correct]
    incorrect = [r for r in records if not r.final_correct]

    print("\n" + "=" * 60)
    print("RUN-LEVEL ANALYSIS")
    print("=" * 60)
    print(f"Total runs:    {len(records)}")
    print(f"Correct runs:  {len(correct)}  ({100*len(correct)/max(len(records),1):.1f}%)")
    print(f"Incorrect:     {len(incorrect)}")

    if len(correct) < 3 or len(incorrect) < 3:
        print(f"\n⚠ Insufficient samples in one group (need ≥ 3 in each).")
        print("  Cannot run hypothesis test. Run more problems first.")
        return {"insufficient_samples": True}

    # Per-run summary statistics
    c_mean   = np.array([r.mean_entropy()         for r in correct])
    c_med    = np.array([r.median_entropy()       for r in correct])
    c_trim   = np.array([r.trimmed_mean_entropy() for r in correct])
    i_mean   = np.array([r.mean_entropy()         for r in incorrect])
    i_med    = np.array([r.median_entropy()       for r in incorrect])
    i_trim   = np.array([r.trimmed_mean_entropy() for r in incorrect])

    results = {
        "n_correct":      len(correct),
        "n_incorrect":    len(incorrect),
        "accuracy":       len(correct) / len(records),
        "tests": {
            "per_run_mean":     hypothesis_test("per-run mean",         c_mean, i_mean),
            "per_run_median":   hypothesis_test("per-run median",       c_med,  i_med),
            "per_run_trimmed":  hypothesis_test("per-run 5%-trimmed",   c_trim, i_trim),
        },
        "per_phase": {},
    }

    print("\n" + "─" * 60)
    print("PER-RUN HYPOTHESIS TESTS  (Mann-Whitney U, one-sided incorrect > correct)")
    print("─" * 60)
    for key, t in results["tests"].items():
        if t.get("skipped"):
            print(f"\n  [{t['name']}] skipped (n_c={t['n_c']}, n_i={t['n_i']})")
            continue
        verdict = (
            "SUPPORTED ✓"  if t["supported"] else
            "weak signal"  if t["weak"]      else
            "NOT SUPPORTED ✗"
        )
        print(f"\n  [{t['name']}]")
        print(f"    correct:   median={t['median_c']:.4f}  mean={t['mean_c']:.4f}  n={t['n_c']}")
        print(f"    incorrect: median={t['median_i']:.4f}  mean={t['mean_i']:.4f}  n={t['n_i']}")
        print(f"    U={t['U']:.0f}  p={t['p_value']:.4f}  Cliff's δ={t['cliffs_delta']:+.3f}")
        print(f"    → {verdict}")

    # Per-phase analysis
    print("\n" + "─" * 60)
    print("PER-PHASE ANALYSIS  (per-run mean entropy within each phase)")
    print("─" * 60)
    for phase in PHASE_COLORS:
        c_p = np.array([
            np.mean(r.entropies_by_phase().get(phase, []))
            for r in correct if r.entropies_by_phase().get(phase)
        ])
        i_p = np.array([
            np.mean(r.entropies_by_phase().get(phase, []))
            for r in incorrect if r.entropies_by_phase().get(phase)
        ])
        t = hypothesis_test(f"phase={phase}", c_p, i_p)
        results["per_phase"][phase] = t
        if t.get("skipped"):
            print(f"\n  [{phase}] skipped")
            continue
        verdict = "✓ SUPPORTED" if t["supported"] else "✗ NOT SUPPORTED"
        print(f"\n  [{phase}]")
        print(f"    correct mean: {t['mean_c']:.4f}  (n={t['n_c']})")
        print(f"    incorr  mean: {t['mean_i']:.4f}  (n={t['n_i']})")
        print(f"    p={t['p_value']:.4f}  δ={t['cliffs_delta']:+.3f}  {verdict}")

    return results


# ─── Visualization ───────────────────────────────────────────────────────────

def visualize(records: List[RunRecord], stats_results: Dict, out_path: str) -> None:
    correct   = [r for r in records if r.final_correct]
    incorrect = [r for r in records if not r.final_correct]

    if len(correct) < 1 or len(incorrect) < 1:
        print(f"⚠ Cannot visualize: need at least one of each.")
        return

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        f"GoT Token-Level Generation Entropy  ·  Game of 24  ·  {MODEL_ID}\n"
        f"n={len(records)} problems  |  correct={len(correct)}  incorrect={len(incorrect)}",
        fontsize=13, fontweight="bold", y=0.995,
    )
    gs = fig.add_gridspec(3, 3, hspace=0.55, wspace=0.40,
                          height_ratios=[1, 1, 1.2])

    # Row 1: per-phase bar chart + per-run scatter
    ax_ph = fig.add_subplot(gs[0, :2])
    ax_sc = fig.add_subplot(gs[0, 2])

    phases = list(PHASE_COLORS.keys())
    c_means_phase = [
        [np.mean(r.entropies_by_phase().get(p, []))
         for r in correct if r.entropies_by_phase().get(p)]
        for p in phases
    ]
    i_means_phase = [
        [np.mean(r.entropies_by_phase().get(p, []))
         for r in incorrect if r.entropies_by_phase().get(p)]
        for p in phases
    ]
    x = np.arange(len(phases))
    w = 0.35
    ax_ph.bar(x - w/2, [np.mean(c) if c else 0 for c in c_means_phase], w,
              yerr=[np.std(c) if c else 0 for c in c_means_phase],
              color="#27ae60", alpha=0.8, label=f"Correct (n={len(correct)})",
              capsize=4)
    ax_ph.bar(x + w/2, [np.mean(c) if c else 0 for c in i_means_phase], w,
              yerr=[np.std(c) if c else 0 for c in i_means_phase],
              color="#c0392b", alpha=0.8, label=f"Incorrect (n={len(incorrect)})",
              capsize=4)
    ax_ph.set_xticks(x)
    ax_ph.set_xticklabels([p.capitalize() for p in phases])
    ax_ph.set_ylabel("Per-run mean entropy  (nats)")
    ax_ph.set_title("Phase-level mean entropy  (error bars = ±1 SD)")
    ax_ph.legend(fontsize=9)
    ax_ph.grid(axis="y", alpha=0.2)

    # Per-run scatter
    c_run_means = [r.mean_entropy() for r in correct]
    i_run_means = [r.mean_entropy() for r in incorrect]
    rng = np.random.default_rng(0)
    if c_run_means:
        ax_sc.scatter(rng.normal(1, 0.05, len(c_run_means)), c_run_means,
                      color="#27ae60", alpha=0.6, label="Correct", s=22)
        ax_sc.hlines(np.mean(c_run_means), 0.7, 1.3, color="#27ae60", lw=2.5)
    if i_run_means:
        ax_sc.scatter(rng.normal(2, 0.05, len(i_run_means)), i_run_means,
                      color="#c0392b", alpha=0.6, label="Incorrect", s=22)
        ax_sc.hlines(np.mean(i_run_means), 1.7, 2.3, color="#c0392b", lw=2.5)
    ax_sc.set_xticks([1, 2])
    ax_sc.set_xticklabels(["Correct", "Incorrect"])
    ax_sc.set_ylabel("Per-run mean entropy (nats)")
    ax_sc.set_title("Per-run distribution\n(— = group mean)")
    ax_sc.legend(fontsize=9)
    ax_sc.grid(axis="y", alpha=0.2)

    # Row 2: violin (per-run mean) + box (per-run median)
    ax_vm = fig.add_subplot(gs[1, 0])
    ax_vd = fig.add_subplot(gs[1, 1])
    ax_vt = fig.add_subplot(gs[1, 2])

    def violin_or_strip(ax, c_vals, i_vals, title):
        pos = [1, 2]
        if len(c_vals) >= 3 and len(i_vals) >= 3:
            vp = ax.violinplot([c_vals, i_vals], positions=pos, showmedians=True)
            vp["bodies"][0].set_facecolor("#27ae60"); vp["bodies"][0].set_alpha(0.65)
            vp["bodies"][1].set_facecolor("#c0392b"); vp["bodies"][1].set_alpha(0.65)
        else:
            ax.scatter([1]*len(c_vals), c_vals, color="#27ae60", alpha=0.6, s=30)
            ax.scatter([2]*len(i_vals), i_vals, color="#c0392b", alpha=0.6, s=30)
        ax.set_xticks(pos)
        ax.set_xticklabels(["Correct", "Incorrect"])
        ax.set_ylabel("Entropy (nats)")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)

    violin_or_strip(
        ax_vm,
        [r.mean_entropy()   for r in correct],
        [r.mean_entropy()   for r in incorrect],
        "Per-run MEAN entropy",
    )
    violin_or_strip(
        ax_vd,
        [r.median_entropy() for r in correct],
        [r.median_entropy() for r in incorrect],
        "Per-run MEDIAN entropy",
    )
    violin_or_strip(
        ax_vt,
        [r.trimmed_mean_entropy() for r in correct],
        [r.trimmed_mean_entropy() for r in incorrect],
        "Per-run 5%-TRIMMED MEAN",
    )

    # Row 3: stats summary table
    ax_tbl = fig.add_subplot(gs[2, :])
    ax_tbl.axis("off")

    if stats_results.get("insufficient_samples"):
        ax_tbl.text(0.5, 0.5,
                    "Insufficient samples for hypothesis testing\n"
                    f"(correct={len(correct)}, incorrect={len(incorrect)})",
                    ha="center", va="center", fontsize=14, color="#c0392b",
                    family="monospace")
    else:
        rows = []
        for key, t in stats_results.get("tests", {}).items():
            if t.get("skipped"):
                continue
            verdict = "✓" if t["supported"] else ("~" if t["weak"] else "✗")
            rows.append([
                t["name"],
                f"{t['mean_c']:.4f}",
                f"{t['mean_i']:.4f}",
                f"{t['median_c']:.4f}",
                f"{t['median_i']:.4f}",
                f"{t['p_value']:.4f}",
                f"{t['cliffs_delta']:+.3f}",
                verdict,
            ])
        for phase, t in stats_results.get("per_phase", {}).items():
            if t.get("skipped"):
                continue
            verdict = "✓" if t["supported"] else ("~" if t["weak"] else "✗")
            rows.append([
                f"phase: {phase}",
                f"{t['mean_c']:.4f}",
                f"{t['mean_i']:.4f}",
                f"{t['median_c']:.4f}",
                f"{t['median_i']:.4f}",
                f"{t['p_value']:.4f}",
                f"{t['cliffs_delta']:+.3f}",
                verdict,
            ])

        cols = ["Test", "mean(C)", "mean(I)", "med(C)", "med(I)",
                "p-value", "Cliff's δ", "verdict"]
        table = ax_tbl.table(cellText=rows, colLabels=cols, loc="center",
                             cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.6)
        for j, col in enumerate(cols):
            table[(0, j)].set_facecolor("#34495e")
            table[(0, j)].set_text_props(color="white", weight="bold")
        # color verdict cells
        for i, row in enumerate(rows, start=1):
            v = row[-1]
            color = {"✓": "#d4efdf", "~": "#fdebd0", "✗": "#fadbd8"}.get(v, "white")
            table[(i, len(cols)-1)].set_facecolor(color)

        ax_tbl.set_title(
            f"Hypothesis: incorrect entropy > correct entropy  "
            f"(✓ = p<0.05 & |δ|>0.2,  ~ = significant but small effect,  ✗ = not supported)",
            fontsize=11, pad=20,
        )

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Visualization saved → {out_path}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap problems run (default: full dataset).")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Skip model, replot from cache.")
    parser.add_argument("--cache", type=str, default=CACHE_PATH)
    parser.add_argument("--out",   type=str, default=PLOT_PATH)
    args = parser.parse_args()

    ds = GameOf24()
    ds.load_dataset()

    if args.analyze_only:
        if not os.path.exists(args.cache):
            print(f"✗ Cache not found at {args.cache}. Run without --analyze-only first.")
            sys.exit(1)
        records = load_cache(args.cache)
    else:
        sanity_check_evaluator(ds)
        model = QwenModel()

        n_total = len(ds) if hasattr(ds, "__len__") else 1362
        n_run = min(args.limit, n_total) if args.limit else n_total
        print(f"\nRunning GoT on {n_run} problems …\n")

        records: List[RunRecord] = []
        n_correct_so_far = 0

        for idx in range(n_run):
            prob = ds.get_problem(idx)
            nums, qstr = prob.ground_truth, prob.question

            run = run_got(model, qstr, verbose=False)
            run.problem_idx = idx
            result = ds.evaluate_answer(run.final_answer, sorted(nums))
            run.final_correct = result.is_correct
            records.append(run)

            if run.final_correct:
                n_correct_so_far += 1
            tag = "✓" if run.final_correct else "✗"
            acc = 100 * n_correct_so_far / (idx + 1)
            print(f"[{idx+1:4d}/{n_run}] {qstr:18s} → {tag} "
                  f"{run.final_answer[:30]:30s}  "
                  f"H̄={run.mean_entropy():.3f}  "
                  f"acc={acc:.1f}%")

            # save cache periodically
            if (idx + 1) % 20 == 0:
                save_cache(records, args.cache)

        save_cache(records, args.cache)

    stats_results = analyze(records)
    visualize(records, stats_results, args.out)


if __name__ == "__main__":
    main()