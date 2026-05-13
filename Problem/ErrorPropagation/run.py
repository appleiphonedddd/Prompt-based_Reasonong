"""
Error Propagation Preliminary Experiment
=========================================

Proves that a single wrong step in CoT reasoning cascades and kills
the final answer — demonstrated on BBH Geometric Shapes.

Pipeline
--------
1. Tracer   — run CoT on N problems, parse each numbered step,
              detect which step first commits to a wrong choice letter.
2. Injector — take only the correctly-solved problems, plant one
              wrong intermediate conclusion at early / mid / late
              positions, and record whether the final choice survives.
3. Visualizer — produce three publication-quality figures.

Output
------
  Problem/ErrorPropagation/data/
    traces.jsonl        # step traces (one per problem)
    injections.jsonl    # injection results
  Problem/ErrorPropagation/
    error_propagation.png   # combined 3-figure panel
    fig1_accuracy_bar.png
    fig2_heatmap.png
    fig3_propagation_rate.png

Usage
-----
  cd /home/infor/Code/Prompt-based-Reasoning
  python Problem/ErrorPropagation/run.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models.qwen import QwenClient
from benchmark.BigBenchHard.bigbenchhard import BigBenchHard
from tracer import CoTStepTracer
from injector import ErrorInjector
from visualize import plot_all

# ── Config ─────────────────────────────────────────────────────────────────────

N_PROBLEMS = 30
MODEL      = "qwen2.5:32b"
HERE       = Path(__file__).parent
DATA_DIR   = HERE / "data"

# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 64)
    print("  Error Propagation in CoT — BBH Geometric Shapes")
    print(f"  Model: {MODEL}  |  N problems: {N_PROBLEMS}")
    print("=" * 64)

    llm     = QwenClient(model=MODEL)
    dataset = BigBenchHard(task="geometric_shapes")
    dataset.load_dataset()

    # ── Phase 1: Step tracing ──────────────────────────────────────────────────
    print("\n[Phase 1] Step-level CoT tracing")
    tracer = CoTStepTracer(llm=llm, dataset=dataset)
    traces = tracer.run(
        n_problems=N_PROBLEMS,
        save_path=DATA_DIR / "traces.jsonl",
    )

    n_correct = sum(1 for t in traces if t.final_correct)
    print(f"\n  Result: {n_correct}/{len(traces)} correct ({n_correct/max(len(traces),1)*100:.1f}%)")

    # ── Phase 2: Error injection ───────────────────────────────────────────────
    correct_traces = [t for t in traces if t.final_correct and len(t.steps) >= 2]
    print(f"\n[Phase 2] Error injection  ({len(correct_traces)} eligible traces)")

    injector = ErrorInjector(
        llm=llm,
        system_prompt=dataset.get_system_prompt() or "",
        instruction=dataset.get_instruction() or "",
    )
    results = injector.run(
        correct_traces=correct_traces,
        save_path=DATA_DIR / "injections.jsonl",
    )

    # ── Phase 3: Visualization ─────────────────────────────────────────────────
    print("\n[Phase 3] Generating figures")
    plot_all(
        traces=traces,
        results=results,
        out_dir=HERE,
        model_name=MODEL,
        task_name="BBH Geometric Shapes",
        n_problems=N_PROBLEMS,
    )

    print("\n" + "=" * 64)
    print("  Done.")
    print(f"  Figures : {HERE}/*.png")
    print(f"  Logs    : {DATA_DIR}/*.jsonl")
    print("=" * 64)


if __name__ == "__main__":
    main()
