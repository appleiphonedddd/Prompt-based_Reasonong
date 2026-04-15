"""
Unit tests for BigBenchHard benchmark dataset.

Tests cover:
- Task enumeration and validation
- Answer extraction from various formats
- Task-specific answer normalization
- HuggingFace dataset loading
- Evaluation results
"""

import unittest
from benchmark.BigBenchHard.bigbenchhard import (
    BigBenchHard,
    BigBenchHardTask,
    _extract_answer_from_text,
    _normalize_whitespace,
    TASK_ANSWER_TYPES,
)


class TestBigBenchHardTasks(unittest.TestCase):
    """Test BigBenchHardTask enumeration and task initialization."""

    def test_enum_count(self):
        """Verify that all 27 BBH tasks are in the enum."""
        tasks = list(BigBenchHardTask)
        self.assertEqual(len(tasks), 27, "Should have exactly 27 tasks")

    def test_valid_task_initialization(self):
        """Test initialization with valid task names."""
        for task in BigBenchHardTask:
            ds = BigBenchHard(task=task.value)
            self.assertEqual(ds.task, task.value)

    def test_invalid_task_raises_error(self):
        """Test that invalid task names raise ValueError."""
        with self.assertRaises(ValueError) as context:
            BigBenchHard(task="invalid_task_name")
        self.assertIn("Invalid task", str(context.exception))

    def test_default_task(self):
        """Test default task initialization."""
        ds = BigBenchHard()
        self.assertEqual(ds.task, "boolean_expressions")

    def test_custom_split(self):
        """Test initialization with custom split."""
        ds = BigBenchHard(task="boolean_expressions", split="train")
        self.assertEqual(ds.split, "train")


class TestBigBenchHardAnswerExtraction(unittest.TestCase):
    """Test answer extraction from various model output formats."""

    def test_extract_markdown_codeblock(self):
        """Test extraction from markdown code blocks."""
        text = "```\nFalse\n```"
        result = _extract_answer_from_text(text)
        self.assertIn("False", result)

    def test_extract_latex_dollars(self):
        """Test extraction from LaTeX $...$ delimiters."""
        text = "The answer is $42$."
        result = _extract_answer_from_text(text)
        self.assertIn("42", result)

    def test_extract_answer_prefix(self):
        """Test extraction with 'Answer:' prefix."""
        text = "After careful reasoning, Answer: True"
        result = _extract_answer_from_text(text)
        self.assertIn("True", result)

    def test_extract_final_answer(self):
        """Test extraction with 'Final answer:' prefix."""
        text = "Working through the problem...\nFinal answer: C"
        result = _extract_answer_from_text(text)
        self.assertIn("C", result)

    def test_extract_fallback_last_line(self):
        """Test fallback: extract last non-empty line."""
        text = "Some reasoning here\n\nB"
        result = _extract_answer_from_text(text)
        self.assertEqual(result.strip(), "B")

    def test_normalize_whitespace(self):
        """Test whitespace normalization."""
        text = "  word1   word2   word3  "
        result = _normalize_whitespace(text)
        self.assertEqual(result, "word1 word2 word3")


class TestBigBenchHardEvaluation(unittest.TestCase):
    """Test task-specific evaluation and normalization."""

    def test_boolean_normalization(self):
        """Test boolean answer normalization."""
        ds = BigBenchHard(task="boolean_expressions")

        # Test True variants
        for pred in ["true", "True", "yes", "YES", "correct", "1"]:
            result = ds.evaluate_answer(pred, "True")
            self.assertTrue(result.is_correct, f"Failed for prediction: {pred}")

        # Test False variants
        for pred in ["false", "False", "no", "NO", "incorrect", "0"]:
            result = ds.evaluate_answer(pred, "False")
            self.assertTrue(result.is_correct, f"Failed for prediction: {pred}")

        # Test mismatch
        result = ds.evaluate_answer("True", "False")
        self.assertFalse(result.is_correct)

    def test_numeric_normalization(self):
        """Test numeric answer extraction."""
        ds = BigBenchHard(task="multistep_arithmetic_two")

        # Test number extraction
        result = ds.evaluate_answer("The answer is 42", "42")
        self.assertTrue(result.is_correct)

        # Test negative number
        result = ds.evaluate_answer("Answer: -5", "-5")
        self.assertTrue(result.is_correct)

        # Test number in sentence
        result = ds.evaluate_answer("After calculation, 100 is the result", "100")
        self.assertTrue(result.is_correct)

    def test_choice_normalization(self):
        """Test multiple-choice answer extraction."""
        ds = BigBenchHard(task="disambiguation_qa")

        # Test letter extraction
        result = ds.evaluate_answer("The answer is (B)", "B")
        self.assertTrue(result.is_correct)

        # Test case-insensitive
        result = ds.evaluate_answer("Answer: c", "C")
        self.assertTrue(result.is_correct)

        # Test standalone letter
        result = ds.evaluate_answer("A", "A")
        self.assertTrue(result.is_correct)

    def test_word_list_normalization(self):
        """Test word list normalization for word_sorting."""
        ds = BigBenchHard(task="word_sorting")

        # Test comma-separated
        result = ds.evaluate_answer("apple, banana, cherry", "apple banana cherry")
        self.assertTrue(result.is_correct)

        # Test space-separated
        result = ds.evaluate_answer("dog cat bird", "bird cat dog")
        self.assertTrue(result.is_correct)

        # Test order doesn't matter (they get sorted)
        result = ds.evaluate_answer("zebra apple mouse", "apple mouse zebra")
        self.assertTrue(result.is_correct)

    def test_answer_extraction_with_formatting(self):
        """Test answer extraction combined with formatting."""
        ds = BigBenchHard(task="boolean_expressions")

        # LaTeX + answer prefix
        result = ds.evaluate_answer(
            "The answer is $True$",
            "True"
        )
        self.assertTrue(result.is_correct)

        # Markdown code block
        result = ds.evaluate_answer(
            "```\nFalse\n```",
            "False"
        )
        self.assertTrue(result.is_correct)

    def test_evaluation_details(self):
        """Test that evaluation result contains details."""
        ds = BigBenchHard(task="boolean_expressions")
        result = ds.evaluate_answer("True", "True")

        self.assertIn("raw_prediction", result.details)
        self.assertIn("extracted_answer", result.details)
        self.assertIn("extracted_normalized", result.details)
        self.assertIn("truth_normalized", result.details)
        self.assertIn("task", result.details)
        self.assertIn("answer_type", result.details)


class TestBigBenchHardDatasetLoading(unittest.TestCase):
    """Test HuggingFace dataset loading."""

    def test_load_dataset(self):
        """Test loading dataset from HuggingFace (network dependent)."""
        ds = BigBenchHard(task="boolean_expressions", split="train")
        try:
            ds.load_dataset()
            self.assertIsNotNone(ds._data)
            self.assertGreater(len(ds._data), 0)
        except RuntimeError as e:
            if "Failed to load" in str(e):
                self.skipTest("HuggingFace dataset not accessible (network issue)")
            else:
                raise

    def test_get_problem(self):
        """Test retrieving a problem after loading."""
        ds = BigBenchHard(task="boolean_expressions", split="train")
        try:
            ds.load_dataset()
            problem = ds.get_problem(0)

            self.assertIsNotNone(problem.index)
            self.assertIsNotNone(problem.question)
            self.assertIsNotNone(problem.ground_truth)
            self.assertIsNotNone(problem.metadata)
        except RuntimeError as e:
            if "Failed to load" in str(e):
                self.skipTest("HuggingFace dataset not accessible (network issue)")
            else:
                raise

    def test_get_problem_out_of_range(self):
        """Test that out-of-range index raises IndexError."""
        ds = BigBenchHard(task="boolean_expressions", split="train")
        try:
            ds.load_dataset()
            with self.assertRaises(IndexError):
                ds.get_problem(999999)
        except RuntimeError as e:
            if "Failed to load" in str(e):
                self.skipTest("HuggingFace dataset not accessible (network issue)")
            else:
                raise

    def test_get_problem_before_load(self):
        """Test that accessing problem before load raises RuntimeError."""
        ds = BigBenchHard(task="boolean_expressions")
        with self.assertRaises(RuntimeError):
            ds.get_problem(0)

    def test_dataset_name(self):
        """Test dataset name formatting."""
        ds = BigBenchHard(task="boolean_expressions")
        self.assertIn("boolean_expressions", ds.dataset_name)
        self.assertIn("BigBenchHard", ds.dataset_name)


class TestTaskAnswerTypes(unittest.TestCase):
    """Test task answer type classification."""

    def test_all_tasks_have_type(self):
        """Verify all tasks have an answer type classification."""
        for task in BigBenchHardTask:
            self.assertIn(task.value, TASK_ANSWER_TYPES,
                         f"Task {task.value} missing from TASK_ANSWER_TYPES")

    def test_valid_answer_types(self):
        """Verify answer types are valid."""
        valid_types = {"boolean", "numeric", "choice", "word_list", "default"}
        for task, answer_type in TASK_ANSWER_TYPES.items():
            self.assertIn(answer_type, valid_types,
                         f"Task {task} has invalid answer type: {answer_type}")


if __name__ == "__main__":
    unittest.main()
