"""
Test suite for BigBench benchmark implementation.

Tests loading, problem retrieval, and answer evaluation for all four
BigBench Hard tasks.

Author: Egor Morozov
"""

import unittest
from benchmark.BigBench.bigbench import BigBench, BigBenchTask


class TestBigBenchTasks(unittest.TestCase):
    """Test BigBench task initialization and validation."""

    def test_valid_tasks(self):
        """Test that all valid tasks can be initialized."""
        valid_tasks = [
            "geometric_shapes",
            "multistep_arithmetic_two",
            "word_sorting",
            "checkmate_in_one",
        ]
        for task in valid_tasks:
            bb = BigBench(task=task)
            self.assertEqual(bb.task, task)
            self.assertIsNotNone(bb.get_instruction())
            self.assertIsNotNone(bb.get_system_prompt())

    def test_invalid_task_raises_error(self):
        """Test that invalid task names raise ValueError."""
        with self.assertRaises(ValueError):
            BigBench(task="invalid_task_name")

    def test_task_enum(self):
        """Test that BigBenchTask enum contains all valid tasks."""
        expected = {
            "geometric_shapes",
            "multistep_arithmetic_two",
            "word_sorting",
            "checkmate_in_one",
        }
        actual = {t.value for t in BigBenchTask}
        self.assertEqual(actual, expected)


class TestBigBenchAnswerExtraction(unittest.TestCase):
    """Test answer extraction from various model output formats."""

    def setUp(self):
        """Initialize a BigBench instance for testing."""
        self.bb = BigBench(task="geometric_shapes")

    def test_simple_answer_extraction(self):
        """Test extraction of simple answers."""
        test_cases = [
            ("Yes", "yes"),
            ("NO", "no"),
            ("  The answer is yes  ", "yes"),
            ("Answer: Yes", "yes"),
            ("Final answer: no", "no"),
        ]
        from benchmark.BigBench.bigbench import _extract_answer_from_text

        for input_text, expected in test_cases:
            result = _extract_answer_from_text(input_text).lower()
            self.assertEqual(result, expected, f"Failed for input: {input_text}")

    def test_markdown_code_block_extraction(self):
        """Test extraction from markdown code blocks."""
        from benchmark.BigBench.bigbench import _extract_answer_from_text

        # With code block
        text = "```\nYes\n```"
        result = _extract_answer_from_text(text).lower()
        self.assertEqual(result, "yes")

        # Python code block
        text = "```python\nYes\n```"
        result = _extract_answer_from_text(text).lower()
        self.assertEqual(result, "yes")

    def test_latex_extraction(self):
        """Test extraction from LaTeX formatted text."""
        from benchmark.BigBench.bigbench import _extract_answer_from_text

        # Inline LaTeX
        text = "The answer is $5 + 3 = 8$"
        result = _extract_answer_from_text(text)
        self.assertIn("8", result)

        # Equation block
        text = r"The result: \[5 + 3 = 8\]"
        result = _extract_answer_from_text(text)
        self.assertIn("8", result)

    def test_whitespace_normalization(self):
        """Test whitespace normalization."""
        from benchmark.BigBench.bigbench import _normalize_whitespace

        test_cases = [
            ("  hello   world  ", "hello world"),
            ("hello\n\nworld", "hello world"),
            ("hello\t\tworld", "hello world"),
        ]
        for input_text, expected in test_cases:
            result = _normalize_whitespace(input_text)
            self.assertEqual(result, expected)


class TestBigBenchEvaluation(unittest.TestCase):
    """Test answer evaluation for various BigBench tasks."""

    def test_geometric_shapes_evaluation(self):
        """Test evaluation for geometric shapes task."""
        bb = BigBench(task="geometric_shapes")

        # Exact match
        result = bb.evaluate_answer("Yes", "Yes")
        self.assertTrue(result.is_correct)
        self.assertEqual(result.score, 1.0)

        # Case-insensitive match
        result = bb.evaluate_answer("yes", "YES")
        self.assertTrue(result.is_correct)

        # Mismatch
        result = bb.evaluate_answer("No", "Yes")
        self.assertFalse(result.is_correct)
        self.assertEqual(result.score, 0.0)

    def test_arithmetic_evaluation(self):
        """Test evaluation for multi-step arithmetic task."""
        bb = BigBench(task="multistep_arithmetic_two")

        # Exact match
        result = bb.evaluate_answer("42", "42")
        self.assertTrue(result.is_correct)

        # Answer wrapped in text
        result = bb.evaluate_answer("The answer is 42", "42")
        self.assertTrue(result.is_correct)

        # Mismatch
        result = bb.evaluate_answer("41", "42")
        self.assertFalse(result.is_correct)

    def test_word_sorting_evaluation(self):
        """Test evaluation for word sorting task."""
        bb = BigBench(task="word_sorting")

        # Exact match (space-separated)
        result = bb.evaluate_answer("apple banana cherry", "apple banana cherry")
        self.assertTrue(result.is_correct)

        # Different separator (comma-separated)
        result = bb.evaluate_answer("apple, banana, cherry", "apple banana cherry")
        self.assertTrue(result.is_correct)

        # Words are sorted for comparison (order doesn't matter)
        result = bb.evaluate_answer("cherry banana apple", "apple banana cherry")
        self.assertTrue(result.is_correct)  # Normalized to sorted form

    def test_checkmate_evaluation(self):
        """Test evaluation for checkmate-in-one task."""
        bb = BigBench(task="checkmate_in_one")

        # Exact match (lowercase)
        result = bb.evaluate_answer("Nf3", "Nf3")
        self.assertTrue(result.is_correct)

        # Case-insensitive
        result = bb.evaluate_answer("nf3", "NF3")
        self.assertTrue(result.is_correct)

        # Chess move normalization (spaces removed)
        result = bb.evaluate_answer("N f 3", "Nf3")
        self.assertTrue(result.is_correct)

        # Mismatch
        result = bb.evaluate_answer("e4", "Nf3")
        self.assertFalse(result.is_correct)


class TestBigBenchDatasetLoading(unittest.TestCase):
    """Test dataset loading (requires internet for HuggingFace download)."""

    def test_load_dataset_succeeds(self):
        """Test that dataset loading works (will download from HF)."""
        bb = BigBench(task="geometric_shapes", split="validation")
        try:
            bb.load_dataset()
            self.assertIsNotNone(bb._data)
            self.assertGreater(len(bb), 0)
        except RuntimeError as e:
            # Skip if network is unavailable
            if "Failed to load" in str(e):
                self.skipTest("HuggingFace dataset unavailable (network issue)")
            raise

    def test_get_problem_before_loading_raises_error(self):
        """Test that accessing problems before loading raises RuntimeError."""
        bb = BigBench(task="geometric_shapes")
        with self.assertRaises(RuntimeError):
            bb.get_problem(0)

    def test_get_problem_out_of_range_raises_error(self):
        """Test that out-of-range index raises IndexError."""
        bb = BigBench(task="geometric_shapes")
        try:
            bb.load_dataset()
            with self.assertRaises(IndexError):
                bb.get_problem(999999)
        except RuntimeError as e:
            if "Failed to load" in str(e):
                self.skipTest("HuggingFace dataset unavailable")
            raise


if __name__ == "__main__":
    unittest.main()
