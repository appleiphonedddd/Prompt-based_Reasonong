"""
Error injection experiment for BBH Geometric Shapes.

For each problem that CoT solves correctly, we plant one wrong intermediate
conclusion at three positions (early / mid / late) and let the model continue
reasoning from that corrupted step.

Corruption strategy (tried in order):
  1. Corrupt a small integer  (e.g. "3 segments" → "4 segments")
  2. Swap a shape name        (e.g. "triangle" → "quadrilateral")
  3. Flip the choice letter   (e.g. "(J)" → "(H)")
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models.base import BaseLLM
from tracer import StepTrace, _extract_final_choice


# ── Prompt ─────────────────────────────────────────────────────────────────────

_CONTINUE_PROMPT = """\
{system_prompt}

{instruction}

Question:
{question}

A student started solving this problem but may have made an error.
Continue the reasoning from Step {next_step} and determine the correct answer.

{partial_steps}

Continue from Step {next_step}:\
"""


# ── Corruption strategies ──────────────────────────────────────────────────────

_SMALL_INT = re.compile(r"\b([2-9]|1[0-2])\b")

_SHAPE_SWAPS = {
    "triangle":      "quadrilateral",
    "quadrilateral": "triangle",
    "pentagon":      "hexagon",
    "hexagon":       "pentagon",
    "heptagon":      "octagon",
    "octagon":       "heptagon",
    "rectangle":     "pentagon",
    "kite":          "triangle",
    "circle":        "octagon",
    "sector":        "circle",
    "line":          "triangle",
}

_CHOICE_LETTERS = list("ABCDEFGHIJ")
_CHOICE_RE      = re.compile(r"\(([A-Ja-j])\)")


def _corrupt(text: str, correct_choice: str) -> Tuple[str, bool]:
    """Try the three corruption strategies in order; return (result, success)."""
    correct_letter = correct_choice.strip("()").upper()

    # Strategy 1: corrupt a small integer (segment / vertex counts)
    ints = list(_SMALL_INT.finditer(text))
    if ints:
        m   = ints[-1]
        val = int(m.group())
        wrong = (val % 8) + 2
        if wrong == val:
            wrong = (val % 7) + 3
        return text[: m.start()] + str(wrong) + text[m.end() :], True

    # Strategy 2: swap a shape name
    for shape, replacement in _SHAPE_SWAPS.items():
        if re.search(rf"\b{shape}\b", text, re.IGNORECASE):
            corrupted = re.sub(
                rf"\b{shape}\b", replacement, text, count=1, flags=re.IGNORECASE
            )
            return corrupted, True

    # Strategy 3: flip a choice letter
    choices = list(_CHOICE_RE.finditer(text))
    if choices:
        m = choices[-1]
        for letter in _CHOICE_LETTERS:
            if letter != correct_letter and letter != m.group(1).upper():
                corrupted = text[: m.start()] + f"({letter})" + text[m.end() :]
                return corrupted, True

    return text, False


def _pick_step(n_steps: int, position: str) -> Optional[int]:
    if n_steps < 2:
        return None
    return {
        "early": max(1, round(n_steps * 0.25)),
        "mid":   max(1, round(n_steps * 0.50)),
        "late":  max(1, n_steps - 1),
    }.get(position)


# ── Data container ─────────────────────────────────────────────────────────────

@dataclass
class InjectionResult:
    problem_index: int
    question: str
    ground_truth: str
    position: str
    inject_step: Optional[int]
    total_steps: int
    continuation: str
    final_answer: Optional[str]
    final_correct: bool

    def to_dict(self) -> dict:
        return {
            "problem_index": self.problem_index,
            "ground_truth": self.ground_truth,
            "position": self.position,
            "inject_step": self.inject_step,
            "total_steps": self.total_steps,
            "final_answer": self.final_answer,
            "final_correct": self.final_correct,
        }


# ── Injector ───────────────────────────────────────────────────────────────────

class ErrorInjector:
    POSITIONS = ("clean", "early", "mid", "late")

    def __init__(self, llm: BaseLLM, system_prompt: str, instruction: str) -> None:
        self.llm = llm
        self.system_prompt = system_prompt
        self.instruction   = instruction

    def _clean_result(self, trace: StepTrace) -> InjectionResult:
        return InjectionResult(
            problem_index=trace.problem_index,
            question=trace.question,
            ground_truth=trace.ground_truth,
            position="clean",
            inject_step=None,
            total_steps=len(trace.steps),
            continuation="",
            final_answer=trace.final_answer,
            final_correct=trace.final_correct,
        )

    def _inject_at(self, trace: StepTrace, position: str) -> Optional[InjectionResult]:
        n = len(trace.steps)
        k = _pick_step(n, position)
        if k is None or k > n:
            return None

        lines: List[str] = []
        for step in trace.steps:
            if step.index < k:
                lines.append(f"Step {step.index}: {step.text}")
            elif step.index == k:
                corrupted, ok = _corrupt(step.text, trace.ground_truth)
                if not ok:
                    return None
                lines.append(f"Step {step.index}: {corrupted}")
                break

        prompt = _CONTINUE_PROMPT.format(
            system_prompt=self.system_prompt,
            instruction=self.instruction,
            question=trace.question,
            partial_steps="\n".join(lines),
            next_step=k + 1,
        )

        content        = self.llm.generate(prompt, temperature=0.0).content
        final_letter   = _extract_final_choice(content)
        correct_letter = trace.ground_truth.strip("()").upper()
        final_correct  = (
            final_letter is not None
            and final_letter.upper() == correct_letter
        )

        return InjectionResult(
            problem_index=trace.problem_index,
            question=trace.question,
            ground_truth=trace.ground_truth,
            position=position,
            inject_step=k,
            total_steps=n,
            continuation=content[:300],
            final_answer=final_letter,
            final_correct=final_correct,
        )

    def run(
        self,
        correct_traces: List[StepTrace],
        save_path: Optional[Path] = None,
    ) -> List[InjectionResult]:
        results: List[InjectionResult] = []

        for trace in correct_traces:
            for pos in self.POSITIONS:
                print(f"  [injector] Q{trace.problem_index:03d} pos={pos:5s}", end="  ", flush=True)
                try:
                    r = self._clean_result(trace) if pos == "clean" else self._inject_at(trace, pos)
                    if r is None:
                        print("SKIP (too few steps or no corruption found)")
                        continue
                    mark = "+" if r.final_correct else "x"
                    print(f"[{mark}]  pred={r.final_answer}  GT={r.ground_truth.strip('()')}")
                    results.append(r)
                except Exception as exc:
                    print(f"ERROR: {exc}")

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                for r in results:
                    f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
            print(f"  [injector] Saved {len(results)} results → {save_path}")

        return results
