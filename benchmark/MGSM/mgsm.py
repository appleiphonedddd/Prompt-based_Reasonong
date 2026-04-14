"""
MGSM (Multilingual Grade School Math) Benchmark Dataset.

MGSM extends the English GSM8K dataset with professional translations into
10 languages.  Each sample is a grade-school math word problem together
with a numeric answer.

HuggingFace source:
    dataset: "juletxara/mgsm"  (https://huggingface.co/datasets/juletxara/mgsm)
    config:  one of the language codes below
    split:   "test" (250 problems per language)

Supported languages:
    en  – English          zh  – Chinese (Simplified)
    de  – German           ja  – Japanese
    fr  – French           th  – Thai
    es  – Spanish          sw  – Swahili
    ru  – Russian          bn  – Bengali

Evaluation strategy:
    Extract the final numeric value from the model's prediction and
    compare it to the ground-truth integer.  The comparison is exact
    (no floating-point tolerance needed for GSM-style problems).

Author: Egor Morozov
"""

import re
from typing import Any, Optional

from benchmark.datasetbase import DatasetBase, EvaluationResult, Problem


# Maps the CLI-friendly language names to HuggingFace config names
LANGUAGE_MAP: dict[str, str] = {
    "en": "en", "english":    "en",
    "de": "de", "german":     "de",
    "fr": "fr", "french":     "fr",
    "es": "es", "spanish":    "es",
    "ru": "ru", "russian":    "ru",
    "zh": "zh", "chinese":    "zh",
    "ja": "ja", "japanese":   "ja",
    "th": "th", "thai":       "th",
    "sw": "sw", "swahili":    "sw",
    "bn": "bn", "bengali":    "bn",
}

# Default number word to digit mapping used by the extractor
_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12,
}


def _extract_number(text: str) -> Optional[float]:
    """Extract the last number mentioned in a string.

    Handles:
    - Integers and decimals ("42", "3.14")
    - Comma-formatted numbers ("1,234")
    - Negative numbers ("-7")
    - Common answer patterns ("The answer is 42", "= 42")

    Returns:
        The extracted numeric value, or ``None`` if none found.
    """
    # Normalise: remove commas used as thousands separators
    cleaned = text.replace(",", "")

    # Priority: look for explicit answer markers
    marker_pattern = re.compile(
        r"(?:answer(?:\s+is)?|result(?:\s+is)?|=)\s*(-?\d+(?:\.\d+)?)",
        re.IGNORECASE,
    )
    marker_match = marker_pattern.search(cleaned)
    if marker_match:
        return float(marker_match.group(1))

    # Fallback: collect all numbers and return the last one
    all_numbers = re.findall(r"-?\d+(?:\.\d+)?", cleaned)
    if all_numbers:
        return float(all_numbers[-1])

    # Last resort: English number words — return the value of the word that
    # appears LAST in the text (consistent with the numeric-fallback behaviour
    # above which also takes the last number found).
    lower = text.lower()
    last_pos: int = -1
    last_val: Optional[float] = None
    for word, value in _NUMBER_WORDS.items():
        m = re.search(rf"\b{word}\b", lower)
        if m and m.start() > last_pos:
            last_pos = m.start()
            last_val = float(value)

    return last_val


class MGSM(DatasetBase):
    """Benchmark wrapper for the MGSM multilingual math dataset.

    Each problem is a grade-school math word problem (potentially in a
    non-English language) with a numeric answer.

    Args:
        language: Language code or name (default: ``"en"``).
                  See ``LANGUAGE_MAP`` for all supported values.
        split:    HuggingFace dataset split (default: ``"test"``).

    Example::

        ds = MGSM(language="zh")
        ds.load_dataset()
        print(len(ds))                          # 250

        problem = ds.get_problem(0)
        print(problem.question)
        print(problem.ground_truth)             # e.g. 72

        result = ds.evaluate_answer("The answer is 72.", problem.ground_truth)
        print(result.is_correct, result.score)  # True  1.0
    """

    HF_DATASET_ID = "juletxara/mgsm"

    def __init__(self, language: str = "en", split: str = "test"):
        lang_code = LANGUAGE_MAP.get(language.lower())
        if lang_code is None:
            raise ValueError(
                f"Unsupported language '{language}'. "
                f"Choose from: {sorted(set(LANGUAGE_MAP.values()))}"
            )
        self.language = lang_code
        super().__init__(split=split, dataset_name=f"MGSM-{lang_code.upper()}")

    # ── Abstract method implementations ───────────────────────────────────

    def load_dataset(self) -> None:
        """Download the MGSM dataset from Hugging Face.

        Populates ``self._data`` with the HuggingFace Dataset object
        for the selected language and split.

        Raises:
            ImportError:  If the ``datasets`` package is not installed.
            RuntimeError: If the download or validation fails.
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
                self.language,
                split=self.split,
            )
        except Exception as exc:
            raise RuntimeError(
                f"[{self.dataset_name}] Failed to load "
                f"'{self.HF_DATASET_ID}' (lang='{self.language}', "
                f"split='{self.split}'): {exc}"
            ) from exc

        self._data = raw
        print(
            f"[{self.dataset_name}] Loaded {len(self._data)} problems "
            f"(lang='{self.language}', split='{self.split}')."
        )

    def get_problem(self, index: int) -> Problem:
        """Return the MGSM problem at the given index.

        Args:
            index: Zero-based index into the dataset split.

        Returns:
            Problem whose ``question`` is the word problem string and
            ``ground_truth`` is the integer answer.

        Raises:
            RuntimeError: If the dataset has not been loaded yet.
            IndexError:   If ``index`` is out of range.
        """
        self._ensure_loaded()

        if index < 0 or index >= len(self._data):
            raise IndexError(
                f"[{self.dataset_name}] Index {index} out of range "
                f"[0, {len(self._data) - 1}]."
            )

        row = self._data[index]

        # juletxara/mgsm schema columns: "question", "answer", "answer_number"
        question: str     = row.get("question", "")
        answer_number     = row.get("answer_number", None)

        # answer_number is the clean integer; fall back to parsing "answer"
        if answer_number is None:
            raw_answer: str = row.get("answer", "")
            extracted = _extract_number(raw_answer)
            answer_number = int(extracted) if extracted is not None else None

        return Problem(
            index=index,
            question=question.strip(),
            ground_truth=answer_number,
            metadata={
                "language":    self.language,
                "raw_answer":  row.get("answer", ""),
                "raw_row":     dict(row),
            },
        )

    def evaluate_answer(
        self,
        prediction: str,
        ground_truth: Any,
    ) -> EvaluationResult:
        """Evaluate a predicted answer against the ground-truth integer.

        The prediction is deemed correct when the extracted number
        matches ``ground_truth`` exactly (integer equality after rounding).

        Args:
            prediction:   The model's raw answer string.
            ground_truth: The integer answer from ``get_problem()``.

        Returns:
            EvaluationResult with ``is_correct`` and ``score`` in {0.0, 1.0}.
        """
        details: dict = {
            "raw_prediction": prediction,
            "ground_truth":   ground_truth,
        }

        if ground_truth is None:
            details["error"] = "ground_truth is None — cannot evaluate."
            return EvaluationResult(
                is_correct=False,
                score=0.0,
                prediction=prediction,
                ground_truth=ground_truth,
                details=details,
            )

        extracted = _extract_number(prediction)
        details["extracted_number"] = extracted

        if extracted is None:
            details["error"] = "No numeric value found in prediction."
            return EvaluationResult(
                is_correct=False,
                score=0.0,
                prediction=prediction,
                ground_truth=ground_truth,
                details=details,
            )

        is_correct = round(extracted) == int(ground_truth)
        details["is_correct"] = is_correct

        return EvaluationResult(
            is_correct=is_correct,
            score=1.0 if is_correct else 0.0,
            prediction=prediction,
            ground_truth=ground_truth,
            details=details,
        )

    # ── Optional hook overrides ────────────────────────────────────────────

    def get_instruction(self) -> str:
        return (
            "Solve the math word problem step by step. "
            "At the end of your solution, state the final numeric answer "
            "clearly, for example: 'The answer is 42.'"
        )

    def get_system_prompt(self) -> str:
        return (
            "You are a precise mathematical problem solver. "
            "Always end your response with 'The answer is [number].'"
        )
