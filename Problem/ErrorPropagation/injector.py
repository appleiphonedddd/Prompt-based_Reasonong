"""
Error injection experiment for CRUXEval (code output prediction).

For each problem that CoT solves correctly, we plant one wrong intermediate
value at three positions (early / mid / late) and let the model continue
reasoning from that corrupted step.

Corruption strategy (tried in order):
  1. Corrupt a small integer (e.g. "result = 3" → "result = 4")
  2. Flip a boolean value   (True ↔ False)
  3. Corrupt a string value (append/change one character)
"""

from __future__ import annotations

import ast
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models.base import BaseLLM
from tracer import StepTrace
from benchmark.CRUXEval.cruxeval import _extract_python_literal, _eval_literal


# ── Prompt ─────────────────────────────────────────────────────────────────────

_CONTINUE_PROMPT = """\
{system_prompt}

{instruction}

{question}

A student started tracing through this function but may have made an error.
Continue the execution trace from Step {next_step} and determine the correct return value.

{partial_steps}

Continue from Step {next_step}:\
"""


# ── Corruption strategies ──────────────────────────────────────────────────────

_SMALL_INT = re.compile(r"\b([2-9]|1[0-2])\b")
_BOOL_RE   = re.compile(r"\b(True|False)\b")
_STR_RE    = re.compile(r"'([^']{1,20})'|\"([^\"]{1,20})\"")


def _corrupt(text: str) -> Tuple[str, bool]:
    """Try the three corruption strategies in order; return (result, success)."""

    # Strategy 1: corrupt a small integer
    ints = list(_SMALL_INT.finditer(text))
    if ints:
        m   = ints[-1]
        val = int(m.group())
        wrong = (val % 8) + 2
        if wrong == val:
            wrong = (val % 7) + 3
        return text[: m.start()] + str(wrong) + text[m.end():], True

    # Strategy 2: flip a boolean
    m = _BOOL_RE.search(text)
    if m:
        original  = m.group(1)
        flipped   = "False" if original == "True" else "True"
        corrupted = text[: m.start()] + flipped + text[m.end():]
        return corrupted, True

    # Strategy 3: corrupt a short string literal
    m = _STR_RE.search(text)
    if m:
        quote    = "'" if m.group(1) is not None else '"'
        content  = m.group(1) if m.group(1) is not None else m.group(2)
        corrupted_content = content + "x" if content else "x"
        replacement = f"{quote}{corrupted_content}{quote}"
        corrupted = text[: m.start()] + replacement + text[m.end():]
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
    ground_truth: Any
    position: str
    inject_step: Optional[int]
    total_steps: int
    continuation: str
    final_answer: Optional[str]
    final_correct: bool

    def to_dict(self) -> dict:
        gt = self.ground_truth
        expected = gt["expected_output"] if isinstance(gt, dict) else str(gt)
        return {
            "problem_index": self.problem_index,
            "ground_truth": expected,
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

    def _expected_output(self, ground_truth: Any) -> str:
        if isinstance(ground_truth, dict):
            return ground_truth.get("expected_output", "")
        return str(ground_truth)

    def _values_equal(self, predicted: str, expected: str) -> bool:
        pred_val, pred_ok = _eval_literal(predicted)
        exp_val,  exp_ok  = _eval_literal(expected)
        if pred_ok and exp_ok:
            return pred_val == exp_val
        return predicted.strip() == expected.strip()

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
                corrupted, ok = _corrupt(step.text)
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

        content       = self.llm.generate(prompt, temperature=0.0).content
        final_answer  = _extract_python_literal(content)
        expected      = self._expected_output(trace.ground_truth)
        final_correct = (
            final_answer is not None
            and self._values_equal(final_answer, expected)
        )

        return InjectionResult(
            problem_index=trace.problem_index,
            question=trace.question,
            ground_truth=trace.ground_truth,
            position=position,
            inject_step=k,
            total_steps=n,
            continuation=content[:300],
            final_answer=final_answer,
            final_correct=final_correct,
        )

    def run(
        self,
        correct_traces: List[StepTrace],
        save_path: Optional[Path] = None,
    ) -> List[InjectionResult]:
        results: List[InjectionResult] = []

        for trace in correct_traces:
            expected = self._expected_output(trace.ground_truth)
            for pos in self.POSITIONS:
                print(f"  [injector] Q{trace.problem_index:03d} pos={pos:5s}", end="  ", flush=True)
                try:
                    r = self._clean_result(trace) if pos == "clean" else self._inject_at(trace, pos)
                    if r is None:
                        print("SKIP (too few steps or no corruption found)")
                        continue
                    mark = "+" if r.final_correct else "x"
                    print(f"[{mark}]  pred={r.final_answer!r}  GT={expected!r}")
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
