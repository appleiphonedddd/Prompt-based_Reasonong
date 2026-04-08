"""
Tests for the Benchmark module.

Covers:
- DatasetBase abstract interface enforcement
- GameOf24: problem retrieval, expression evaluation, edge cases
- MGSM: problem retrieval, number extraction, evaluation
- Registry completeness

Author: Egor Morozov
"""

import unittest
from unittest.mock import MagicMock, patch, PropertyMock

from benchmark.datasetbase import DatasetBase, Problem, EvaluationResult
from benchmark.GameOf24.gameof24 import GameOf24, _safe_eval, _extract_numbers_from_expr
from benchmark.MGSM.mgsm import MGSM, _extract_number, LANGUAGE_MAP
from benchmark import DATASET_REGISTRY


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_hf_row_g24(puzzle="2 5 8 11"):
    """Minimal HuggingFace-style row for GameOf24."""
    return {"Puzzles": puzzle}


def _make_hf_row_mgsm(question="Bob has 3 apples.", answer="3", answer_number=3):
    return {"question": question, "answer": answer, "answer_number": answer_number}


def _mock_dataset(rows):
    """Create a list-like mock that behaves like a HuggingFace Dataset."""
    mock = MagicMock()
    mock.__len__.return_value = len(rows)
    mock.__getitem__.side_effect = lambda i: rows[i]
    return mock


# ─────────────────────────────────────────────────────────────────────────────
# DatasetBase abstract interface
# ─────────────────────────────────────────────────────────────────────────────

class TestDatasetBaseAbstract(unittest.TestCase):

    def test_cannot_instantiate_directly(self):
        with self.assertRaises(TypeError):
            DatasetBase()

    def test_concrete_subclass_must_implement_all_methods(self):
        """A class missing any abstract method should still raise TypeError."""
        class Incomplete(DatasetBase):
            def load_dataset(self): ...
            def get_problem(self, index): ...
            # evaluate_answer intentionally omitted

        with self.assertRaises(TypeError):
            Incomplete()

    def test_ensure_loaded_raises_before_load(self):
        class Concrete(DatasetBase):
            def load_dataset(self): self._data = []
            def get_problem(self, i): return Problem(i, "", None)
            def evaluate_answer(self, p, g): return EvaluationResult(False, 0.0, p, g)

        ds = Concrete()
        with self.assertRaises(RuntimeError):
            ds._ensure_loaded()

    def test_len_returns_zero_before_load(self):
        class Concrete(DatasetBase):
            def load_dataset(self): self._data = [1, 2, 3]
            def get_problem(self, i): return Problem(i, "", None)
            def evaluate_answer(self, p, g): return EvaluationResult(False, 0.0, p, g)

        ds = Concrete()
        self.assertEqual(len(ds), 0)
        ds.load_dataset()
        self.assertEqual(len(ds), 3)

    def test_default_hooks_return_none(self):
        class Concrete(DatasetBase):
            def load_dataset(self): ...
            def get_problem(self, i): ...
            def evaluate_answer(self, p, g): ...

        ds = Concrete()
        self.assertIsNone(ds.get_instruction())
        self.assertIsNone(ds.get_system_prompt())


# ─────────────────────────────────────────────────────────────────────────────
# GameOf24 helper functions
# ─────────────────────────────────────────────────────────────────────────────

class TestGameOf24Helpers(unittest.TestCase):

    def test_safe_eval_basic_arithmetic(self):
        self.assertAlmostEqual(_safe_eval("2+5"), 7.0)
        self.assertAlmostEqual(_safe_eval("(11-5)*(8/2)"), 24.0)
        self.assertAlmostEqual(_safe_eval("3*8"), 24.0)

    def test_safe_eval_division_by_zero(self):
        with self.assertRaises(ZeroDivisionError):
            _safe_eval("1/0")

    def test_safe_eval_invalid_expression(self):
        with self.assertRaises(ValueError):
            _safe_eval("import os")

    def test_safe_eval_rejects_function_calls(self):
        with self.assertRaises(ValueError):
            _safe_eval("__import__('os')")

    def test_extract_numbers_basic(self):
        self.assertEqual(_extract_numbers_from_expr("(2+5)*8/11"), [2, 5, 8, 11])

    def test_extract_numbers_with_spaces(self):
        self.assertEqual(_extract_numbers_from_expr("3 * 8"), [3, 8])


# ─────────────────────────────────────────────────────────────────────────────
# GameOf24 dataset
# ─────────────────────────────────────────────────────────────────────────────

class TestGameOf24Dataset(unittest.TestCase):

    def _make_loaded_ds(self, rows):
        ds = GameOf24()
        ds._data = _mock_dataset(rows)
        return ds

    # ── get_problem ──

    def test_get_problem_returns_correct_structure(self):
        ds = self._make_loaded_ds([_make_hf_row_g24("2 5 8 11")])
        problem = ds.get_problem(0)

        self.assertIsInstance(problem, Problem)
        self.assertEqual(problem.question, "2 5 8 11")
        self.assertEqual(problem.ground_truth, sorted([2, 5, 8, 11]))
        self.assertEqual(problem.index, 0)

    def test_get_problem_raises_before_load(self):
        ds = GameOf24()
        with self.assertRaises(RuntimeError):
            ds.get_problem(0)

    def test_get_problem_raises_on_out_of_range(self):
        ds = self._make_loaded_ds([_make_hf_row_g24()])
        with self.assertRaises(IndexError):
            ds.get_problem(5)

    # ── evaluate_answer ──

    def test_evaluate_correct_answer(self):
        ds = self._make_loaded_ds([])
        result = ds.evaluate_answer(
            prediction="(11 - 5) * (8 / 2)",
            ground_truth=sorted([2, 5, 8, 11]),
        )
        # numbers used: 11,5,8,2 → sorted [2,5,8,11] ✓, result=24 ✓
        self.assertTrue(result.is_correct)
        self.assertEqual(result.score, 1.0)

    def test_evaluate_wrong_result(self):
        ds = self._make_loaded_ds([])
        result = ds.evaluate_answer(
            prediction="2 + 5 + 8 + 11",   # = 26, not 24
            ground_truth=sorted([2, 5, 8, 11]),
        )
        self.assertFalse(result.is_correct)

    def test_evaluate_wrong_numbers(self):
        ds = self._make_loaded_ds([])
        # (4+4)*3 = 24 but uses wrong numbers
        result = ds.evaluate_answer(
            prediction="(4+4)*3",
            ground_truth=sorted([2, 5, 8, 11]),
        )
        self.assertFalse(result.is_correct)
        self.assertFalse(result.details["numbers_match"])

    def test_evaluate_invalid_expression(self):
        ds = self._make_loaded_ds([])
        result = ds.evaluate_answer(
            prediction="not a number at all !!",
            ground_truth=sorted([2, 5, 8, 11]),
        )
        self.assertFalse(result.is_correct)
        self.assertIn("error", result.details)

    def test_evaluate_strips_markdown(self):
        ds = self._make_loaded_ds([])
        result = ds.evaluate_answer(
            prediction="```\n(11-5)*(8/2)\n```",
            ground_truth=sorted([2, 5, 8, 11]),
        )
        # After stripping markdown, expression should evaluate to 24
        self.assertTrue(result.details.get("reaches_24", False))

    # ── hooks ──

    def test_hooks_return_strings(self):
        ds = GameOf24()
        self.assertIsInstance(ds.get_instruction(), str)
        self.assertIsInstance(ds.get_system_prompt(), str)


# ─────────────────────────────────────────────────────────────────────────────
# MGSM helper functions
# ─────────────────────────────────────────────────────────────────────────────

class TestMGSMHelpers(unittest.TestCase):

    def test_extract_number_plain_integer(self):
        self.assertEqual(_extract_number("42"), 42.0)

    def test_extract_number_with_answer_marker(self):
        self.assertEqual(_extract_number("The answer is 72."), 72.0)

    def test_extract_number_with_equals(self):
        self.assertEqual(_extract_number("result = 15"), 15.0)

    def test_extract_number_comma_formatted(self):
        self.assertEqual(_extract_number("1,234"), 1234.0)

    def test_extract_number_negative(self):
        self.assertEqual(_extract_number("-7"), -7.0)

    def test_extract_number_returns_none_when_absent(self):
        self.assertIsNone(_extract_number("no numbers here"))

    def test_extract_number_prefers_marker(self):
        # "The answer is 42" should beat the "3" in the sentence
        result = _extract_number("She bought 3 apples. The answer is 42.")
        self.assertEqual(result, 42.0)


# ─────────────────────────────────────────────────────────────────────────────
# MGSM dataset
# ─────────────────────────────────────────────────────────────────────────────

class TestMGSMDataset(unittest.TestCase):

    def _make_loaded_ds(self, rows, language="en"):
        ds = MGSM(language=language)
        ds._data = _mock_dataset(rows)
        return ds

    # ── init ──

    def test_invalid_language_raises(self):
        with self.assertRaises(ValueError):
            MGSM(language="klingon")

    def test_all_language_codes_accepted(self):
        for code in set(LANGUAGE_MAP.values()):
            ds = MGSM(language=code)
            self.assertEqual(ds.language, code)

    # ── get_problem ──

    def test_get_problem_returns_correct_structure(self):
        row = _make_hf_row_mgsm("Bob has 3 apples.", "3", 3)
        ds = self._make_loaded_ds([row])
        problem = ds.get_problem(0)

        self.assertIsInstance(problem, Problem)
        self.assertEqual(problem.question, "Bob has 3 apples.")
        self.assertEqual(problem.ground_truth, 3)

    def test_get_problem_falls_back_to_answer_string(self):
        row = {"question": "Q?", "answer": "The answer is 99.", "answer_number": None}
        ds = self._make_loaded_ds([row])
        problem = ds.get_problem(0)
        self.assertEqual(problem.ground_truth, 99)

    def test_get_problem_raises_before_load(self):
        ds = MGSM()
        with self.assertRaises(RuntimeError):
            ds.get_problem(0)

    def test_get_problem_raises_on_out_of_range(self):
        ds = self._make_loaded_ds([_make_hf_row_mgsm()])
        with self.assertRaises(IndexError):
            ds.get_problem(10)

    # ── evaluate_answer ──

    def test_evaluate_correct_integer(self):
        ds = self._make_loaded_ds([])
        result = ds.evaluate_answer("The answer is 72.", 72)
        self.assertTrue(result.is_correct)
        self.assertEqual(result.score, 1.0)

    def test_evaluate_wrong_integer(self):
        ds = self._make_loaded_ds([])
        result = ds.evaluate_answer("The answer is 71.", 72)
        self.assertFalse(result.is_correct)

    def test_evaluate_none_ground_truth(self):
        ds = self._make_loaded_ds([])
        result = ds.evaluate_answer("42", None)
        self.assertFalse(result.is_correct)
        self.assertIn("error", result.details)

    def test_evaluate_no_number_in_prediction(self):
        ds = self._make_loaded_ds([])
        result = ds.evaluate_answer("I don't know.", 42)
        self.assertFalse(result.is_correct)
        self.assertIsNone(result.details["extracted_number"])

    def test_evaluate_float_rounds_correctly(self):
        ds = self._make_loaded_ds([])
        result = ds.evaluate_answer("The answer is 3.0", 3)
        self.assertTrue(result.is_correct)

    # ── hooks ──

    def test_hooks_return_strings(self):
        ds = MGSM()
        self.assertIsInstance(ds.get_instruction(), str)
        self.assertIsInstance(ds.get_system_prompt(), str)


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

class TestRegistry(unittest.TestCase):

    def test_registry_contains_gameof24(self):
        self.assertIn("gameof24", DATASET_REGISTRY)
        cls, _ = DATASET_REGISTRY["gameof24"]
        self.assertIs(cls, GameOf24)

    def test_registry_contains_mgsm(self):
        self.assertIn("mgsm", DATASET_REGISTRY)
        cls, _ = DATASET_REGISTRY["mgsm"]
        self.assertIs(cls, MGSM)

    def test_all_registry_entries_are_dataset_subclasses(self):
        for name, (cls, _) in DATASET_REGISTRY.items():
            with self.subTest(dataset=name):
                self.assertTrue(
                    issubclass(cls, DatasetBase),
                    f"{cls} must be a subclass of DatasetBase",
                )


if __name__ == "__main__":
    unittest.main()
