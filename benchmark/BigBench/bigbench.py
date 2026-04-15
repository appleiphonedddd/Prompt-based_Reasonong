"""
BigBench Benchmark Dataset.

Supports four BIG-Bench Hard (BBH) reasoning tasks:
    1. Geometric Shapes  — Geometric reasoning with multi-hop logic
    2. Multi-Step Arithmetic — Complex arithmetic with multiple steps
    3. Word Sorting      — Alphabetical or custom ordering of words
    4. Checkmate-in-One  — Chess problem: find move that leads to checkmate

HuggingFace source:
    dataset: "tasksource/bigbench"
    config: "geometric_shapes", "multistep_arithmetic_two", "word_sorting",
            "checkmate_in_one"

Evaluation strategy:
    - Extract answer from model output (handle markdown, LaTeX, etc.)
    - Compare against ground truth using exact string match
    - Task-specific logic for answer extraction and normalization

Author: Egor Morozov
"""

import re
from typing import Any
from enum import Enum

from benchmark.datasetbase import DatasetBase, EvaluationResult, Problem


class BigBenchTask(Enum):
    """Enumeration of supported BigBench Hard tasks."""
    GEOMETRIC_SHAPES = "geometric_shapes"
    MULTI_STEP_ARITHMETIC = "multistep_arithmetic_two"
    WORD_SORTING = "word_sorting"
    CHECKMATE_IN_ONE = "checkmate_in_one"


def _normalize_whitespace(text: str) -> str:
    """Normalize whitespace: strip and collapse multiple spaces."""
    return " ".join(text.split())


def _extract_answer_from_text(text: str) -> str:
    """
    Extract the answer from model output.

    Handles:
    - Markdown code blocks (```...```)
    - LaTeX delimiters (\(...\), \[...\], $...$)
    - Common prefix patterns ("The answer is...", "Answer:", etc.)

    Returns:
        The extracted answer string, stripped and normalized.
    """
    # Remove markdown code blocks
    text = re.sub(r"```[a-zA-Z0-9]*\n(.*?)\n```", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"```", "", text)

    # Remove LaTeX delimiters (extract content)
    text = re.sub(r"\$\$(.*?)\$\$", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"\$", "", text)

    # Remove inline \(...\) and \[...\]
    text = re.sub(r"\\\((.*?)\\\)", r"\1", text)
    text = re.sub(r"\\\[(.*?)\\\]", r"\1", text, flags=re.DOTALL)

    # Try to extract answer after common patterns
    patterns = [
        r"[Aa]nswer\s*(?:is)?\s*[:\-]?\s*([^\n]+)",
        r"[Ff]inal\s+answer\s*[:\-]?\s*([^\n]+)",
        r"[Tt]he\s+answer\s+is\s*[:\-]?\s*([^\n]+)",
        r"[Rr]esult\s*[:\-]?\s*([^\n]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return _normalize_whitespace(match.group(1).strip())

    # Fallback: return the last non-empty line
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if lines:
        return _normalize_whitespace(lines[-1])

    return _normalize_whitespace(text)


class BigBench(DatasetBase):
    """Benchmark wrapper for BIG-Bench Hard (BBH) reasoning tasks.

    Supports four reasoning tasks:
    - Geometric Shapes: Logical reasoning about geometric properties
    - Multi-Step Arithmetic: Complex arithmetic requiring multiple steps
    - Word Sorting: Ordering words by various criteria
    - Checkmate-in-One: Chess positions requiring mate-in-one solutions

    Args:
        task: Which BigBench task to load. Must be one of:
              'geometric_shapes', 'multistep_arithmetic_two',
              'word_sorting', 'checkmate_in_one'
        split: HuggingFace split to load (default: "validation").

    Example::

        ds = BigBench(task="geometric_shapes")
        ds.load_dataset()
        problem = ds.get_problem(0)
        result = ds.evaluate_answer("Yes", problem.ground_truth)
    """

    HF_DATASET_ID = "tasksource/bigbench"

    # Map task names to HuggingFace config names (some datasets use different names)
    TASK_NAME_MAPPING = {
        "geometric_shapes": "geometric_shapes",
        "multistep_arithmetic_two": "arithmetic",
        "word_sorting": "word_sorting",
        "checkmate_in_one": "checkmate_in_one",
    }

    def __init__(self, task: str = "geometric_shapes", split: str = "validation"):
        """Initialize BigBench dataset for a specific task.

        Args:
            task: Task name (must be one of: 'geometric_shapes',
                  'multistep_arithmetic_two', 'word_sorting', 'checkmate_in_one').
            split: HuggingFace split ("validation" or "train").

        Raises:
            ValueError: If task is not recognized.
        """
        # Validate task
        try:
            self.task_enum = BigBenchTask(task)
            self.task = task
        except ValueError as exc:
            valid_tasks = [t.value for t in BigBenchTask]
            raise ValueError(
                f"Invalid task '{task}'. Valid tasks: {valid_tasks}"
            ) from exc

        # Get the HuggingFace config name for this task
        self.hf_config_name = self.TASK_NAME_MAPPING[task]
        super().__init__(split=split, dataset_name=f"BigBench[{task}]")

    # ── Abstract method implementations ───────────────────────────────────

    def load_dataset(self) -> None:
        """Download the BigBench dataset from HuggingFace.

        Loads the specified task and split into self._data.

        Raises:
            ImportError: If 'datasets' package is not installed.
            RuntimeError: If download or validation fails.
        """
        try:
            from datasets import load_dataset as hf_load
        except ImportError as exc:
            raise ImportError(
                "The 'datasets' package is required. "
                "Install it with: pip install datasets"
            ) from exc

        try:
            raw = hf_load(
                self.HF_DATASET_ID,
                name=self.hf_config_name,
                split=self.split,
            )
        except Exception as exc:
            raise RuntimeError(
                f"[{self.dataset_name}] Failed to load '{self.HF_DATASET_ID}' "
                f"(name='{self.task}', split='{self.split}'): {exc}"
            ) from exc

        self._data = raw
        print(
            f"[{self.dataset_name}] Loaded {len(self._data)} problems "
            f"from '{self.HF_DATASET_ID}' (name='{self.task}', split='{self.split}')."
        )

    def get_problem(self, index: int) -> Problem:
        """Return the BigBench problem at the given index.

        Args:
            index: Zero-based index into the dataset.

        Returns:
            Problem with 'question' as the task description and
            'ground_truth' as the expected answer.

        Raises:
            RuntimeError: If dataset has not been loaded.
            IndexError: If index is out of range.
        """
        self._ensure_loaded()

        if index < 0 or index >= len(self._data):
            raise IndexError(
                f"[{self.dataset_name}] Index {index} out of range "
                f"[0, {len(self._data) - 1}]."
            )

        row = self._data[index]

        # BigBench standard columns
        question = row.get("input", "")
        answer = row.get("target", [""])[0]  # target is often a list

        return Problem(
            index=index,
            question=question.strip(),
            ground_truth=answer.strip() if isinstance(answer, str) else answer,
            metadata={
                "raw_row": dict(row),
                "task": self.task,
            },
        )

    def evaluate_answer(
        self,
        prediction: str,
        ground_truth: Any,
    ) -> EvaluationResult:
        """Evaluate a prediction against the ground truth.

        Default evaluation uses case-insensitive exact string match
        after normalization. Subclasses can override for task-specific logic.

        Args:
            prediction: Model's raw output.
            ground_truth: Expected answer from dataset.

        Returns:
            EvaluationResult with is_correct and score.
        """
        # Extract answer from prediction
        extracted = _extract_answer_from_text(prediction)

        # Normalize both for comparison
        extracted_normalized = _normalize_whitespace(extracted).lower()
        truth_normalized = _normalize_whitespace(
            str(ground_truth)
        ).lower()

        # Task-specific post-processing
        if self.task == "word_sorting":
            # Word Sorting: might be comma-separated or space-separated
            extracted_normalized = self._normalize_word_list(extracted_normalized)
            truth_normalized = self._normalize_word_list(truth_normalized)

        elif self.task == "checkmate_in_one":
            # Chess moves: normalize notation (remove extra spaces, lowercase)
            extracted_normalized = self._normalize_chess_move(extracted_normalized)
            truth_normalized = self._normalize_chess_move(truth_normalized)

        # Exact match evaluation
        is_correct = extracted_normalized == truth_normalized

        details = {
            "raw_prediction": prediction,
            "extracted_answer": extracted,
            "extracted_normalized": extracted_normalized,
            "truth_normalized": truth_normalized,
            "task": self.task,
        }

        return EvaluationResult(
            is_correct=is_correct,
            score=1.0 if is_correct else 0.0,
            prediction=prediction,
            ground_truth=ground_truth,
            details=details,
        )

    # ── Task-specific normalization helpers ────────────────────────────────

    @staticmethod
    def _normalize_word_list(text: str) -> str:
        """Normalize word lists (comma or space separated).

        Returns:
            Sorted space-separated words for consistent comparison.
        """
        # Handle both comma-separated and space-separated
        if "," in text:
            words = [w.strip() for w in text.split(",")]
        else:
            words = text.split()

        # Sort and rejoin for canonical form
        words = sorted(set(w for w in words if w))
        return " ".join(words)

    @staticmethod
    def _normalize_chess_move(text: str) -> str:
        """Normalize chess move notation.

        Handles both algebraic (e.g., 'Nf3') and coordinate notation.

        Returns:
            Normalized move string.
        """
        # Remove spaces and normalize
        move = "".join(text.split())
        # Lowercase everything for case-insensitive comparison
        return move.lower()

    # ── Optional hook overrides ────────────────────────────────────────────

    def get_instruction(self) -> str:
        """Return task-specific instruction."""
        instructions = {
            "geometric_shapes": (
                "Reason about geometric shapes and their properties. "
                "Answer with 'yes', 'no', or the requested property."
            ),
            "multistep_arithmetic_two": (
                "Solve this arithmetic problem step-by-step. "
                "Provide the final numerical answer."
            ),
            "word_sorting": (
                "Sort the given words alphabetically or as requested. "
                "Provide the sorted words in order."
            ),
            "checkmate_in_one": (
                "Given a chess position, find the move that delivers checkmate in one. "
                "Respond with the move in algebraic notation."
            ),
        }
        return instructions.get(self.task, "")

    def get_system_prompt(self) -> str:
        """Return task-specific system prompt."""
        system_prompts = {
            "geometric_shapes": (
                "You are an expert in geometric reasoning. "
                "Answer precisely and concisely."
            ),
            "multistep_arithmetic_two": (
                "You are an expert mathematician. "
                "Show your work and provide the final answer."
            ),
            "word_sorting": (
                "You are an expert in language and ordering. "
                "Sort words accurately."
            ),
            "checkmate_in_one": (
                "You are a master chess player. "
                "Identify the winning move in standard algebraic notation."
            ),
        }
        return system_prompts.get(self.task, "")
