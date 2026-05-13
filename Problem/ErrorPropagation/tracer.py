"""
Step-level CoT tracer for BBH Geometric Shapes.

Runs zero-shot CoT, parses each numbered step, and marks whether the step
is steering toward the correct answer.

For geometric shapes (multiple-choice), "step correctness" is determined by
checking whether the step commits to a choice letter:
    • Correct letter (or no letter yet) → True / None
    • Wrong letter mentioned confidently  → False
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models.base import BaseLLM
from benchmark.datasetbase import DatasetBase


# ── Regex helpers ──────────────────────────────────────────────────────────────

_STEP_SPLIT   = re.compile(r"(?:^|\n)\s*Step\s+(\d+)\s*[:\.]?\s*", re.IGNORECASE)
_CHOICE       = re.compile(r"\(([A-Ja-j])\)")
_FINAL_CHOICE = re.compile(
    r"(?:answer|choose|option|select)\s*(?:is)?\s*[:\s]*\(?([A-Ja-j])\)?",
    re.IGNORECASE,
)

_STEP_PROMPT = """\
{system_prompt}

{instruction}

Question:
{question}

Reason step by step. Number every step clearly (Step 1:, Step 2:, ...).
Work through the problem systematically before committing to an answer.
At the very end write exactly one line: "The answer is (X)." where X is the option letter.\
"""


# ── Step verification ──────────────────────────────────────────────────────────

def verify_step_choice(step_text: str, correct_choice: str) -> Optional[bool]:
    """Check whether a reasoning step is consistent with the correct choice.

    Returns:
        True  — step mentions no letter, or explicitly matches correct choice.
        False — step explicitly mentions a wrong choice letter.
        None  — ambiguous (multiple letters, or can't determine confidence).
    """
    correct_letter = correct_choice.strip("()").lower()
    mentions = [m.group(1).lower() for m in _CHOICE.finditer(step_text)]

    if not mentions:
        return None

    wrong = [l for l in mentions if l != correct_letter]
    right = [l for l in mentions if l == correct_letter]

    if wrong and not right:
        return False
    if right:
        return True
    return None


def _extract_final_choice(text: str) -> Optional[str]:
    """Pull the choice letter out of the model's final sentence."""
    m = _FINAL_CHOICE.search(text)
    if m:
        return m.group(1).upper()
    all_choices = _CHOICE.findall(text)
    return all_choices[-1].upper() if all_choices else None


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
    ground_truth: str              # e.g. "(J)"
    raw_response: str
    steps: List[Step]
    final_answer: Optional[str]
    final_correct: bool
    first_error_step: Optional[int]

    def to_dict(self) -> dict:
        return {
            "problem_index": self.problem_index,
            "question": self.question[:120],
            "ground_truth": self.ground_truth,
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
    """Runs CoT on BBH Geometric Shapes and tracks step-level correctness."""

    def __init__(self, llm: BaseLLM, dataset: DatasetBase) -> None:
        self.llm = llm
        self.dataset = dataset

    def _build_prompt(self, problem) -> str:
        return _STEP_PROMPT.format(
            system_prompt=self.dataset.get_system_prompt() or "",
            instruction=self.dataset.get_instruction() or "",
            question=problem.question,
        )

    def _parse_steps(self, response: str, ground_truth: str) -> List[Step]:
        parts = _STEP_SPLIT.split(response)
        steps: List[Step] = []
        i = 1
        while i + 1 < len(parts):
            step_num  = int(parts[i])
            step_text = parts[i + 1].strip()
            steps.append(Step(
                index=step_num,
                text=step_text,
                is_correct=verify_step_choice(step_text, ground_truth),
            ))
            i += 2
        return steps

    def trace(self, problem_index: int) -> StepTrace:
        problem = self.dataset.get_problem(problem_index)
        gt      = str(problem.ground_truth)

        response = self.llm.generate(self._build_prompt(problem), temperature=0.0)
        content  = response.content

        steps         = self._parse_steps(content, gt)
        final_letter  = _extract_final_choice(content)
        final_correct = (
            final_letter is not None
            and final_letter.upper() == gt.strip("()").upper()
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
            final_answer=final_letter,
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
                err_info = f"  first_err=step{t.first_error_step}" if t.first_error_step else ""
                print(f"[{status}]  GT={t.ground_truth}  pred={t.final_answer}  steps={len(t.steps)}{err_info}")
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
