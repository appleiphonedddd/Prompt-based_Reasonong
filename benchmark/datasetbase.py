"""
Abstract base class for Benchmark Datasets.

This module defines the DatasetBase abstract class which enforces a
consistent interface for all benchmark dataset implementations.

Every concrete dataset must implement:
- load_dataset():       Download/load from Hugging Face
- get_problem(index):   Retrieve a single problem by index
- evaluate_answer():    Score a predicted answer against ground truth

Design Notes:
    Follows the same ABC pattern as models/base.py (BaseLLM),
    keeping the project architecture consistent.

Author: Egor Morozov
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Data container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Problem:
    """Standardised container for a single benchmark problem.

    Attributes:
        index:        Position in the dataset split.
        question:     The raw question / puzzle string shown to the model.
        ground_truth: The expected correct answer (string or structured data).
        metadata:     Any dataset-specific extras (e.g. language, difficulty).
    """
    index: int
    question: str
    ground_truth: Any
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        preview = self.question[:60] + "..." if len(self.question) > 60 else self.question
        return f"Problem(index={self.index}, question='{preview}')"


@dataclass
class EvaluationResult:
    """Standardised container for a single evaluation outcome.

    Attributes:
        is_correct:   Whether the prediction is considered correct.
        score:        Numeric score in [0.0, 1.0].
        prediction:   The model's raw answer string.
        ground_truth: The reference answer used for comparison.
        details:      Dataset-specific diagnostic information.
    """
    is_correct: bool
    score: float
    prediction: str
    ground_truth: Any
    details: Dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base class
# ─────────────────────────────────────────────────────────────────────────────

class DatasetBase(ABC):
    """Abstract base class for all benchmark dataset implementations.

    Subclasses must implement the three abstract methods below.
    Optional hook methods (``get_instruction``, ``get_system_prompt``,
    ``__len__``) may be overridden to provide richer behaviour.

    Attributes:
        dataset_name: Human-readable name shown in logs and reports.
        split:        HuggingFace dataset split to load (e.g. "test").
        _data:        The raw HuggingFace dataset object (set by load_dataset).

    Example::

        class MyDataset(DatasetBase):
            def load_dataset(self): ...
            def get_problem(self, index): ...
            def evaluate_answer(self, prediction, ground_truth): ...

        ds = MyDataset()
        problem = ds.get_problem(0)
        result  = ds.evaluate_answer("42", problem.ground_truth)
    """

    def __init__(self, split: str = "test", dataset_name: str = "DatasetBase"):
        """Initialise the dataset wrapper.

        Args:
            split:        HuggingFace split to load (default: ``"test"``).
            dataset_name: Identifier used in logs and reports.
        """
        self.split = split
        self.dataset_name = dataset_name
        self._data = None           # populated by load_dataset()

    # ── Abstract interface ─────────────────────────────────────────────────

    @abstractmethod
    def load_dataset(self) -> None:
        """Download and cache the dataset from Hugging Face.

        Must assign the loaded data to ``self._data``.

        Raises:
            ImportError:  If the ``datasets`` package is unavailable.
            RuntimeError: If the download fails.
        """

    @abstractmethod
    def get_problem(self, index: int) -> Problem:
        """Return the problem at the given position.

        Args:
            index: Zero-based index into the current split.

        Returns:
            A ``Problem`` instance with ``question`` and ``ground_truth``.

        Raises:
            IndexError:   If ``index`` is out of range.
            RuntimeError: If ``load_dataset()`` has not been called yet.
        """

    @abstractmethod
    def evaluate_answer(
        self,
        prediction: str,
        ground_truth: Any,
    ) -> EvaluationResult:
        """Score a model prediction against the reference answer.

        Args:
            prediction:   The model's raw output string.
            ground_truth: The reference answer (type may vary per dataset).

        Returns:
            An ``EvaluationResult`` with ``is_correct`` and ``score``.
        """

    # ── Optional hook methods ──────────────────────────────────────────────

    def get_instruction(self) -> Optional[str]:
        """Return a task-specific instruction prepended to every prompt.

        Override in subclasses to inject dataset-specific guidance.
        Returns ``None`` by default (no extra instruction).
        """
        return None

    def get_system_prompt(self) -> Optional[str]:
        """Return an optional system-level prompt for the LLM.

        Override in subclasses to set a persona or constraint.
        Returns ``None`` by default.
        """
        return None

    def __len__(self) -> int:
        """Return the number of problems in the loaded split.

        Returns:
            0 if ``load_dataset()`` has not yet been called.
        """
        return len(self._data) if self._data is not None else 0

    def _ensure_loaded(self) -> None:
        """Guard helper: raise RuntimeError if data is not yet loaded."""
        if self._data is None:
            raise RuntimeError(
                f"[{self.dataset_name}] Dataset not loaded. "
                "Call load_dataset() before accessing problems."
            )

    def __repr__(self) -> str:
        size = len(self) if self._data is not None else "not loaded"
        return (
            f"{self.__class__.__name__}("
            f"dataset_name='{self.dataset_name}', "
            f"split='{self.split}', "
            f"size={size})"
        )