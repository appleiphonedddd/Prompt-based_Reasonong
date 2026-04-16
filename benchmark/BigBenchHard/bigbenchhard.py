"""
BigBenchHard (BBH) Benchmark Dataset.

Supports all 27 BIG-Bench Hard reasoning tasks:
    1. boolean_expressions — Logical operators (True/False)
    2. causal_judgement — Cause-effect reasoning
    3. date_understanding — Date arithmetic and reasoning
    4. disambiguation_qa — Resolving ambiguous references
    5. dyck_languages — Bracket matching
    6. formal_fallacies — Logical fallacy detection
    7. geometric_shapes — Geometric reasoning
    8. hyperbaton — Word order variations
    9. logical_deduction_three_objects — 3-object deduction
    10. logical_deduction_five_objects — 5-object deduction
    11. logical_deduction_seven_objects — 7-object deduction
    12. movie_recommendation — Movie recommendation task
    13. multistep_arithmetic_two — Multi-step arithmetic
    14. navigate — Navigation instructions
    15. object_counting — Count objects in description
    16. penguins_in_a_table — Table reasoning
    17. reasoning_about_colored_objects — Color logic
    18. ruin_names — Word manipulation
    19. salient_translation_error_detection — Translation errors
    20. snarks — Logic puzzle (snarks)
    21. sports_understanding — Sports rules and facts
    22. temporal_sequences — Time sequence reasoning
    23. tracking_shuffled_objects_three_objects — 3-object tracking
    24. tracking_shuffled_objects_five_objects — 5-object tracking
    25. tracking_shuffled_objects_seven_objects — 7-object tracking
    26. web_of_lies — Logic puzzle (lies/truth)
    27. word_sorting — Alphabetical word sorting

HuggingFace source:
    dataset: "lukaemon/bbh"
    split: "test" (only available split, 250 per task)
    target: string (either "True"/"False" or specific answer)

Evaluation strategy:
    - Extract answer from model output (handle markdown, LaTeX, etc.)
    - Task-specific normalization (multiple-choice, yes/no, numeric, etc.)
    - Compare against ground truth using normalized exact match

Author: Egor Morozov (refactored for BigBenchHard)
"""

import re
from typing import Any, Optional
from enum import Enum

from benchmark.datasetbase import DatasetBase, EvaluationResult, Problem


class BigBenchHardTask(Enum):
    """Enumeration of all 27 BIG-Bench Hard tasks."""
    BOOLEAN_EXPRESSIONS = "boolean_expressions"
    CAUSAL_JUDGEMENT = "causal_judgement"
    DATE_UNDERSTANDING = "date_understanding"
    DISAMBIGUATION_QA = "disambiguation_qa"
    DYCK_LANGUAGES = "dyck_languages"
    FORMAL_FALLACIES = "formal_fallacies"
    GEOMETRIC_SHAPES = "geometric_shapes"
    HYPERBATON = "hyperbaton"
    LOGICAL_DEDUCTION_THREE = "logical_deduction_three_objects"
    LOGICAL_DEDUCTION_FIVE = "logical_deduction_five_objects"
    LOGICAL_DEDUCTION_SEVEN = "logical_deduction_seven_objects"
    MOVIE_RECOMMENDATION = "movie_recommendation"
    MULTISTEP_ARITHMETIC = "multistep_arithmetic_two"
    NAVIGATE = "navigate"
    OBJECT_COUNTING = "object_counting"
    PENGUINS_IN_TABLE = "penguins_in_a_table"
    REASONING_COLORED_OBJECTS = "reasoning_about_colored_objects"
    RUIN_NAMES = "ruin_names"
    SALIENT_TRANSLATION_ERROR = "salient_translation_error_detection"
    SNARKS = "snarks"
    SPORTS_UNDERSTANDING = "sports_understanding"
    TEMPORAL_SEQUENCES = "temporal_sequences"
    TRACKING_SHUFFLED_THREE = "tracking_shuffled_objects_three_objects"
    TRACKING_SHUFFLED_FIVE = "tracking_shuffled_objects_five_objects"
    TRACKING_SHUFFLED_SEVEN = "tracking_shuffled_objects_seven_objects"
    WEB_OF_LIES = "web_of_lies"
    WORD_SORTING = "word_sorting"


# Classification of tasks by answer type for evaluation.
#
# BBH ground-truth formats (verified against lukaemon/bbh dataset):
#   boolean  – True/False, Yes/No, valid/invalid
#   choice   – multiple-choice letter such as (A), (B), …
#   numeric  – integer or decimal number
#   word_list – space-separated sorted words
#   default  – bracket sequence or other free-form text
TASK_ANSWER_TYPES = {
    # Boolean/Yes-No/valid-invalid tasks  (ground truth is a keyword, not a letter)
    "boolean_expressions": "boolean",       # True / False
    "causal_judgement": "boolean",          # Yes / No
    "formal_fallacies": "boolean",          # valid / invalid
    "navigate": "boolean",                  # Yes / No
    "sports_understanding": "boolean",      # Yes / No
    "web_of_lies": "boolean",               # Yes / No

    # Multiple-choice tasks  (ground truth is a letter like "(A)")
    "date_understanding": "choice",
    "disambiguation_qa": "choice",
    "geometric_shapes": "choice",
    "hyperbaton": "choice",
    "logical_deduction_three_objects": "choice",
    "logical_deduction_five_objects": "choice",
    "logical_deduction_seven_objects": "choice",
    "movie_recommendation": "choice",
    "penguins_in_a_table": "choice",
    "reasoning_about_colored_objects": "choice",
    "ruin_names": "choice",
    "salient_translation_error_detection": "choice",
    "snarks": "choice",
    "temporal_sequences": "choice",
    "tracking_shuffled_objects_three_objects": "choice",
    "tracking_shuffled_objects_five_objects": "choice",
    "tracking_shuffled_objects_seven_objects": "choice",

    # Numeric tasks
    "multistep_arithmetic_two": "numeric",
    "object_counting": "numeric",

    # Free-form / special tasks
    "dyck_languages": "default",    # bracket sequences, e.g. "[ ] ( )"
    "word_sorting": "word_list",    # space-separated alphabetically sorted words
}


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


class BigBenchHard(DatasetBase):
    """Benchmark wrapper for all 27 BIG-Bench Hard reasoning tasks.

    Args:
        task: Which BigBenchHard task to load. Must be one of the 27 task names.
        split: HuggingFace split to load (default: "train", only available split).

    Example::

        ds = BigBenchHard(task="boolean_expressions")
        ds.load_dataset()
        problem = ds.get_problem(0)
        result = ds.evaluate_answer("False", problem.ground_truth)
    """

    HF_DATASET_ID = "lukaemon/bbh"

    def __init__(self, task: str = "boolean_expressions", split: str = "test"):
        """Initialize BigBenchHard dataset for a specific task.

        Args:
            task: Task name (must be one of the 27 BBH tasks).
            split: HuggingFace split ("test" only for this dataset).

        Raises:
            ValueError: If task is not recognized.
        """
        # Validate task
        try:
            self.task_enum = BigBenchHardTask(task)
            self.task = task
        except ValueError as exc:
            valid_tasks = [t.value for t in BigBenchHardTask]
            raise ValueError(
                f"Invalid task '{task}'. Valid tasks: {valid_tasks}"
            ) from exc

        super().__init__(split=split, dataset_name=f"BigBenchHard[{task}]")

    # ── Abstract method implementations ───────────────────────────────────

    def load_dataset(self) -> None:
        """Download the BigBenchHard dataset from HuggingFace.

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
                name=self.task,
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
            f"from '{self.HF_DATASET_ID}' (task='{self.task}', split='{self.split}')."
        )

    def get_problem(self, index: int) -> Problem:
        """Return the BigBenchHard problem at the given index.

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

        # BigBenchHard standard columns (target is a plain string, not list)
        question = row.get("input", "")
        answer = row.get("target", "")

        return Problem(
            index=index,
            question=question.strip(),
            ground_truth=answer.strip() if isinstance(answer, str) else str(answer),
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

        Task-specific evaluation logic based on answer type classification.

        Args:
            prediction: Model's raw output.
            ground_truth: Expected answer from dataset.

        Returns:
            EvaluationResult with is_correct and score.
        """
        # Extract answer from prediction
        extracted = _extract_answer_from_text(prediction)

        # Determine task answer type
        answer_type = TASK_ANSWER_TYPES.get(self.task, "default")

        # Normalize both for comparison
        extracted_normalized = self._normalize_answer(extracted, answer_type)
        truth_normalized = self._normalize_answer(str(ground_truth), answer_type)

        # Exact match evaluation
        is_correct = extracted_normalized == truth_normalized

        details = {
            "raw_prediction": prediction,
            "extracted_answer": extracted,
            "extracted_normalized": extracted_normalized,
            "truth_normalized": truth_normalized,
            "task": self.task,
            "answer_type": answer_type,
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
    def _normalize_answer(text: str, answer_type: str) -> str:
        """Normalize answer based on task type.

        Args:
            text: Raw answer text.
            answer_type: One of: "boolean", "numeric", "choice", "word_list", "default".

        Returns:
            Normalized answer for comparison.
        """
        # Always start with basic normalization
        text = _normalize_whitespace(text).lower().strip()

        if answer_type == "boolean":
            # Normalize boolean answers — use word-boundary search so extra
            # words after the answer ("True, because...") don't cause false negatives.
            # Also handles valid/invalid for formal_fallacies.
            if re.search(r"\b(true|yes|correct|valid)\b", text):
                return "true"
            elif re.search(r"\b(false|no|incorrect|invalid)\b", text):
                return "false"
            # Numeric shorthands
            if text.strip() in ("1", "(a)"):
                return "true"
            if text.strip() in ("0", "(b)"):
                return "false"
            return text

        elif answer_type == "numeric":
            # Extract first number (or return as-is if it's numeric)
            match = re.search(r"-?\d+\.?\d*", text)
            if match:
                return match.group()
            return text

        elif answer_type == "choice":
            # Extract choice letter (A–Z).  Try parenthesised form first
            # ("(A)", "(b)"), then bare letter at word boundary.
            match = re.search(r"\(([a-z])\)", text)
            if match:
                return match.group(1)
            match = re.search(r"\b([a-z])\b", text)
            if match:
                return match.group(1)
            return text

        elif answer_type == "word_list":
            # Sort words for consistent comparison
            return BigBenchHard._normalize_word_list(text)

        else:  # "default"
            # Default: just lowercase and normalize whitespace
            return text

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

    # ── Optional hook overrides ────────────────────────────────────────────

    def get_instruction(self) -> Optional[str]:
        """Return task-specific instruction."""
        instructions = {
            "boolean_expressions": "Evaluate the boolean expression and answer True or False.",
            "causal_judgement": "Determine if the causal statement is correct. Answer True or False.",
            "date_understanding": "Answer questions about dates and calendars.",
            "disambiguation_qa": "Answer the multiple-choice question.",
            "dyck_languages": "Check if the bracket sequence is valid.",
            "formal_fallacies": "Identify if the argument contains a logical fallacy. Answer True or False.",
            "geometric_shapes": "Answer questions about geometric shapes. Answer True, False, or the requested property.",
            "hyperbaton": "Analyze word order and meaning.",
            "logical_deduction_three_objects": "Solve the logical deduction puzzle with three objects.",
            "logical_deduction_five_objects": "Solve the logical deduction puzzle with five objects.",
            "logical_deduction_seven_objects": "Solve the logical deduction puzzle with seven objects.",
            "movie_recommendation": "Select the correct movie based on the description.",
            "multistep_arithmetic_two": "Solve this multi-step arithmetic problem. Provide the final numerical answer.",
            "navigate": "Follow the navigation instructions and describe the result.",
            "object_counting": "Count the objects in the description.",
            "penguins_in_a_table": "Answer the question about the table of penguins. Answer True or False.",
            "reasoning_about_colored_objects": "Reason about objects and their colors. Answer True or False.",
            "ruin_names": "Complete or manipulate the word/name.",
            "salient_translation_error_detection": "Identify the translation error.",
            "snarks": "Solve the logic puzzle.",
            "sports_understanding": "Answer the question about sports. Answer True or False.",
            "temporal_sequences": "Reason about temporal sequences.",
            "tracking_shuffled_objects_three_objects": "Track object positions through shuffling (3 objects).",
            "tracking_shuffled_objects_five_objects": "Track object positions through shuffling (5 objects).",
            "tracking_shuffled_objects_seven_objects": "Track object positions through shuffling (7 objects).",
            "web_of_lies": "Solve the logic puzzle about liars and truth-tellers.",
            "word_sorting": "Sort the words alphabetically or as requested.",
        }
        return instructions.get(self.task, "")

    def get_system_prompt(self) -> Optional[str]:
        """Return task-specific system prompt."""
        system_prompts = {
            "boolean_expressions": "You are an expert in logic and boolean expressions.",
            "causal_judgement": "You are an expert in causal reasoning and judgment.",
            "date_understanding": "You are an expert in calendars and date arithmetic.",
            "disambiguation_qa": "You are skilled at resolving ambiguous references.",
            "dyck_languages": "You are an expert in formal languages and syntax.",
            "formal_fallacies": "You are an expert in logic and detecting fallacies.",
            "geometric_shapes": "You are an expert in geometry and spatial reasoning.",
            "hyperbaton": "You are skilled in linguistic analysis and word order.",
            "logical_deduction_three_objects": "You are an expert in logical deduction.",
            "logical_deduction_five_objects": "You are an expert in logical deduction.",
            "logical_deduction_seven_objects": "You are an expert in logical deduction.",
            "movie_recommendation": "You are knowledgeable about movies and entertainment.",
            "multistep_arithmetic_two": "You are an expert mathematician. Show your work.",
            "navigate": "You are skilled at following and describing directions.",
            "object_counting": "You are skilled at counting objects in descriptions.",
            "penguins_in_a_table": "You are skilled at table analysis and data reasoning.",
            "reasoning_about_colored_objects": "You are skilled at spatial and color reasoning.",
            "ruin_names": "You are skilled at word manipulation and analysis.",
            "salient_translation_error_detection": "You are an expert in languages and translation.",
            "snarks": "You are skilled at solving logic puzzles.",
            "sports_understanding": "You are knowledgeable about sports and rules.",
            "temporal_sequences": "You are skilled at temporal reasoning and sequences.",
            "tracking_shuffled_objects_three_objects": "You are skilled at tracking object positions.",
            "tracking_shuffled_objects_five_objects": "You are skilled at tracking object positions.",
            "tracking_shuffled_objects_seven_objects": "You are skilled at tracking object positions.",
            "web_of_lies": "You are skilled at solving logic puzzles.",
            "word_sorting": "You are skilled in language and word ordering.",
        }
        return system_prompts.get(self.task, "")
