"""
Step-level CoT tracer for CRUXEval (code output prediction).

Runs zero-shot CoT, parses each numbered step, and marks whether the step
is consistent with the correct return value.

For CRUXEval, intermediate steps trace Python execution so "step correctness"
is mostly ambiguous (None) — we only mark a step wrong when it explicitly
states a wrong Python literal as the final value.  The key signal is whether
the model's final extracted literal matches the expected output.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models.base import BaseLLM
from benchmark.datasetbase import DatasetBase
from benchmark.CRUXEval.cruxeval import _extract_python_literal, _eval_literal


# ── Regex helpers ──────────────────────────────────────────────────────────────

_STEP_SPLIT = re.compile(r"(?:^|\n)\s*Step\s+(\d+)\s*[:\.]?\s*", re.IGNORECASE)

_STEP_PROMPT = """\
{system_prompt}

{instruction}

{question}

Reason step by step. Number every step clearly (Step 1:, Step 2:, ...).
Trace through the function execution systematically.
At the very end write exactly one line: "The answer is X." \
where X is the Python literal return value.\
"""


# ── Step verification ──────────────────────────────────────────────────────────

def verify_step_value(step_text: str, expected_output: str) -> Optional[bool]:
    """Check whether a reasoning step is consistent with the expected output.

    Returns:
        True  — step explicitly states the correct Python literal.
        False — step explicitly states a different Python literal as the value.
        None  — ambiguous (can't determine from intermediate computation).
    """
    exp_val, exp_ok = _eval_literal(expected_output)
    if not exp_ok:
        return None

    # Look for "= X" or "returns X" or "is X" patterns at the end of the step
    value_re = re.compile(
        r"(?:=|returns?|is)\s+(.+)$",
        re.IGNORECASE | re.MULTILINE,
    )
    candidates: list[str] = []
    for m in value_re.finditer(step_text):
        candidate = m.group(1).strip().rstrip(".,;")
        val, ok = _eval_literal(candidate)
        if ok:
            candidates.append((val, ok))

    if not candidates:
        return None

    # If any candidate matches expected, the step is on track
    for val, _ in candidates:
        if val == exp_val:
            return True

    # All candidates are literals but none match → step committed to a wrong value
    return False


def _extract_final_value(text: str) -> Optional[str]:
    """Pull the Python literal return value out of the model's response."""
    return _extract_python_literal(text)


# ── Data containers ────────────────────────────────────────────────────────────

@dataclass
class Step:
    index: int
    text: str
    is_correct: Optional[bool]


@dataclass
class StepTrace:
    problem_index: int
    question: str
    ground_truth: Any              # dict with 'expected_output', 'code', 'input', 'id'
    raw_response: str
    steps: List[Step]
    final_answer: Optional[str]
    final_correct: bool
    first_error_step: Optional[int]

    def to_dict(self) -> dict:
        gt = self.ground_truth
        expected = gt["expected_output"] if isinstance(gt, dict) else str(gt)
        return {
            "problem_index": self.problem_index,
            "question": self.question[:120],
            "ground_truth": expected,
            "steps": [
                {"index": s.index, "text": s.text[:150], "is_correct": s.is_correct}
                for s in self.steps
            ],
            "final_answer": self.final_answer,
            "final_correct": self.final_correct,
            "first_error_step": self.first_error_step,
        }


# ── Tracer ─────────────────────────────────────────────────────────────────────

class CoTStepTracer:
    """Runs CoT on CRUXEval and tracks step-level correctness."""

    def __init__(self, llm: BaseLLM, dataset: DatasetBase) -> None:
        self.llm = llm
        self.dataset = dataset

    def _build_prompt(self, problem) -> str:
        return _STEP_PROMPT.format(
            system_prompt=self.dataset.get_system_prompt() or "",
            instruction=self.dataset.get_instruction() or "",
            question=problem.question,
        )

    def _expected_output(self, ground_truth: Any) -> str:
        if isinstance(ground_truth, dict):
            return ground_truth.get("expected_output", "")
        return str(ground_truth)

    def _parse_steps(self, response: str, expected_output: str) -> List[Step]:
        parts = _STEP_SPLIT.split(response)
        steps: List[Step] = []
        i = 1
        while i + 1 < len(parts):
            step_num  = int(parts[i])
            step_text = parts[i + 1].strip()
            steps.append(Step(
                index=step_num,
                text=step_text,
                is_correct=verify_step_value(step_text, expected_output),
            ))
            i += 2
        return steps

    def _values_equal(self, predicted: str, expected: str) -> bool:
        """Compare two Python literal strings by value, with string fallback."""
        pred_val, pred_ok = _eval_literal(predicted)
        exp_val,  exp_ok  = _eval_literal(expected)
        if pred_ok and exp_ok:
            return pred_val == exp_val
        return predicted.strip() == expected.strip()

    def trace(self, problem_index: int) -> StepTrace:
        problem  = self.dataset.get_problem(problem_index)
        gt       = problem.ground_truth
        expected = self._expected_output(gt)

        response = self.llm.generate(self._build_prompt(problem), temperature=0.0)
        content  = response.content

        steps         = self._parse_steps(content, expected)
        final_answer  = _extract_final_value(content)
        final_correct = (
            final_answer is not None
            and self._values_equal(final_answer, expected)
        )
        first_error = next(
            (s.index for s in steps if s.is_correct is False), None
        )

        return StepTrace(
            problem_index=problem_index,
            question=problem.question,
            ground_truth=gt,
            raw_response=content,
            steps=steps,
            final_answer=final_answer,
            final_correct=final_correct,
            first_error_step=first_error,
        )

    def run(
        self,
        n_problems: int = 30,
        save_path: Optional[Path] = None,
    ) -> List[StepTrace]:
        traces: List[StepTrace] = []
        for i in range(min(n_problems, len(self.dataset))):
            print(f"  [tracer] Q{i:03d}", end="  ", flush=True)
            try:
                t = self.trace(i)
                status   = "+" if t.final_correct else "x"
                expected = self._expected_output(t.ground_truth)
                err_info = f"  first_err=step{t.first_error_step}" if t.first_error_step else ""
                print(
                    f"[{status}]  GT={expected!r}  pred={t.final_answer!r}"
                    f"  steps={len(t.steps)}{err_info}"
                )
                traces.append(t)
            except Exception as exc:
                print(f"[ERROR] {exc}")

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                for t in traces:
                    f.write(json.dumps(t.to_dict(), ensure_ascii=False) + "\n")
            print(f"  [tracer] Saved {len(traces)} traces → {save_path}")

        return traces
