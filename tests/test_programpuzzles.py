"""
Unit tests for the ProgrammingPuzzles benchmark dataset.

Tests cover:
- Helper functions: _extract_python_literal, _clean_docstring,
  _call_sat_with_timeout, _build_sat_namespace
- Dataset: load_dataset, get_problem, evaluate_answer
- Registry: presence and subclass correctness
"""

import ast
import time
import unittest

from benchmark.ProgrammingPuzzles.programpuzzles import (
    ProgrammingPuzzles,
    _build_sat_namespace,
    _call_sat_with_timeout,
    _clean_docstring,
    _extract_python_literal,
)
from benchmark.datasetbase import DatasetBase, EvaluationResult, Problem
from benchmark import DATASET_REGISTRY


# ─────────────────────────────────────────────────────────────────────────────
# Factories
# ─────────────────────────────────────────────────────────────────────────────

def _gt(
    sat_src: str = "def sat(x: int):\n    return x == 42",
    ans_type: str = "int",
    name: str = "Test:0",
) -> dict:
    return {"sat": sat_src, "ans_type": ans_type, "name": name}


def _loaded(num_samples: int = 5, module: str = "study.py") -> ProgrammingPuzzles:
    ds = ProgrammingPuzzles(num_samples=num_samples, module=module)
    ds.load_dataset()
    return ds


# ─────────────────────────────────────────────────────────────────────────────
# _extract_python_literal
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractPythonLiteral(unittest.TestCase):

    def _parse(self, text: str, ans_type: str = "") -> object:
        result = _extract_python_literal(text, ans_type)
        self.assertIsNotNone(result, f"Expected a literal from: {text!r}")
        return ast.literal_eval(result)

    def test_plain_integer(self):
        self.assertEqual(self._parse("42", "int"), 42)

    def test_plain_negative_integer(self):
        self.assertEqual(self._parse("-7", "int"), -7)

    def test_plain_float(self):
        self.assertAlmostEqual(self._parse("3.14", "float"), 3.14)

    def test_plain_string_double_quotes(self):
        self.assertEqual(self._parse('"hello"', "str"), "hello")

    def test_plain_string_single_quotes(self):
        self.assertEqual(self._parse("'world'", "str"), "world")

    def test_plain_list_of_ints(self):
        self.assertEqual(self._parse("[1, 2, 3]", "List[int]"), [1, 2, 3])

    def test_plain_bool_true(self):
        self.assertIs(self._parse("True", "bool"), True)

    def test_plain_bool_false(self):
        self.assertIs(self._parse("False", "bool"), False)

    def test_from_fenced_code_block(self):
        self.assertEqual(self._parse("```python\n42\n```", "int"), 42)

    def test_from_plain_code_block(self):
        self.assertEqual(self._parse("```\n[1, 2]\n```", "List[int]"), [1, 2])

    def test_return_inside_code_block(self):
        self.assertEqual(self._parse("```python\nreturn [1, 2, 3]\n```", "List[int]"), [1, 2, 3])

    def test_inline_backtick(self):
        self.assertEqual(self._parse("The answer is `42`", "int"), 42)

    def test_answer_prefix(self):
        val = self._parse("Answer: [1, 2, 3]", "List[int]")
        self.assertEqual(val, [1, 2, 3])

    def test_result_prefix(self):
        self.assertEqual(self._parse("Result: 99", "int"), 99)

    def test_type_hint_str_extracts_quoted(self):
        self.assertEqual(self._parse("My answer is 'hohoho'", "str"), "hohoho")

    def test_type_hint_bool_extracts_keyword(self):
        self.assertIs(self._parse("The answer is True because ...", "bool"), True)

    def test_type_hint_list_extracts_brackets(self):
        self.assertEqual(self._parse("Use this: [4, 5, 6]", "List[int]"), [4, 5, 6])

    def test_last_line_fallback(self):
        text = "Let me reason through this.\n42"
        self.assertEqual(self._parse(text, "int"), 42)

    def test_returns_none_for_unparseable(self):
        result = _extract_python_literal("I cannot solve this puzzle", "int")
        self.assertIsNone(result)

    def test_returns_none_for_empty(self):
        result = _extract_python_literal("", "int")
        self.assertIsNone(result)


# ─────────────────────────────────────────────────────────────────────────────
# _clean_docstring
# ─────────────────────────────────────────────────────────────────────────────

class TestCleanDocstring(unittest.TestCase):

    def test_triple_double_quotes(self):
        self.assertEqual(_clean_docstring('    """Find a string."""'), "Find a string.")

    def test_triple_single_quotes(self):
        self.assertEqual(_clean_docstring("    '''Find a value.'''"), "Find a value.")

    def test_no_quotes(self):
        self.assertEqual(_clean_docstring("   plain text   "), "plain text")

    def test_multiline_docstring(self):
        doc = '    """\n    Find a value.\n    Extra context.\n    """'
        result = _clean_docstring(doc)
        self.assertIn("Find a value.", result)

    def test_empty_string(self):
        self.assertEqual(_clean_docstring(""), "")


# ─────────────────────────────────────────────────────────────────────────────
# _call_sat_with_timeout
# ─────────────────────────────────────────────────────────────────────────────

class TestCallSatWithTimeout(unittest.TestCase):

    def test_correct_answer_returns_true(self):
        def sat(x: int):
            return x == 42

        result, error = _call_sat_with_timeout(sat, 42)
        self.assertIsNone(error)
        self.assertTrue(result)

    def test_wrong_answer_returns_false(self):
        def sat(x: int):
            return x == 42

        result, error = _call_sat_with_timeout(sat, 0)
        self.assertIsNone(error)
        self.assertFalse(result)

    def test_exception_is_captured(self):
        def sat(x):
            raise ValueError("bad")

        result, error = _call_sat_with_timeout(sat, "anything")
        self.assertIsNone(result)
        self.assertIsInstance(error, ValueError)

    def test_assertion_error_is_captured(self):
        def sat(x: int):
            assert x > 0
            return x == 42

        result, error = _call_sat_with_timeout(sat, -1)
        self.assertIsNone(result)
        self.assertIsInstance(error, AssertionError)

    def test_timeout_returns_timeout_error(self):
        def sat(x):
            time.sleep(10)
            return True

        result, error = _call_sat_with_timeout(sat, None, timeout=0.1)
        self.assertIsNone(result)
        self.assertIsInstance(error, TimeoutError)

    def test_sat_with_default_params(self):
        """sat functions may have extra params with defaults; only first is provided."""
        def sat(lst: list, n: int = 3):
            return len(lst) == n

        result, error = _call_sat_with_timeout(sat, [1, 2, 3])
        self.assertIsNone(error)
        self.assertTrue(result)


# ─────────────────────────────────────────────────────────────────────────────
# _build_sat_namespace
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildSatNamespace(unittest.TestCase):

    def test_contains_list(self):
        ns = _build_sat_namespace()
        self.assertIn("List", ns)

    def test_contains_dict(self):
        ns = _build_sat_namespace()
        self.assertIn("Dict", ns)

    def test_exec_with_list_annotation(self):
        ns = _build_sat_namespace()
        exec("def sat(lst: List[int]):\n    return sum(lst) == 6", ns)
        self.assertTrue(ns["sat"]([1, 2, 3]))


# ─────────────────────────────────────────────────────────────────────────────
# ProgrammingPuzzles dataset
# ─────────────────────────────────────────────────────────────────────────────

class TestProgrammingPuzzlesLoad(unittest.TestCase):

    def test_load_all_puzzles(self):
        ds = ProgrammingPuzzles()
        ds.load_dataset()
        self.assertEqual(len(ds), 1715)

    def test_load_with_module_filter(self):
        ds = ProgrammingPuzzles(module="study.py")
        ds.load_dataset()
        self.assertEqual(len(ds), 30)
        for row in ds._data:
            self.assertEqual(row["module"], "study.py")

    def test_load_with_num_samples(self):
        ds = ProgrammingPuzzles(num_samples=10)
        ds.load_dataset()
        self.assertLessEqual(len(ds), 10)

    def test_load_combined_filter(self):
        ds = ProgrammingPuzzles(num_samples=3, module="study.py")
        ds.load_dataset()
        self.assertEqual(len(ds), 3)

    def test_len_before_load_is_zero(self):
        ds = ProgrammingPuzzles()
        self.assertEqual(len(ds), 0)

    def test_missing_file_raises_runtime_error(self):
        ds = ProgrammingPuzzles()
        ds._data = None
        import unittest.mock as mock
        with mock.patch("benchmark.ProgrammingPuzzles.programpuzzles.Path.exists", return_value=False):
            with self.assertRaises(RuntimeError):
                ds.load_dataset()


class TestProgrammingPuzzlesGetProblem(unittest.TestCase):

    def setUp(self):
        self.ds = _loaded(num_samples=5, module="study.py")

    def test_raises_before_load(self):
        ds = ProgrammingPuzzles()
        with self.assertRaises(RuntimeError):
            ds.get_problem(0)

    def test_raises_on_negative_index(self):
        with self.assertRaises(IndexError):
            self.ds.get_problem(-1)

    def test_raises_on_out_of_range(self):
        with self.assertRaises(IndexError):
            self.ds.get_problem(100)

    def test_returns_problem_instance(self):
        problem = self.ds.get_problem(0)
        self.assertIsInstance(problem, Problem)

    def test_question_contains_task_section(self):
        problem = self.ds.get_problem(0)
        self.assertIn("Task:", problem.question)

    def test_question_contains_return_type(self):
        problem = self.ds.get_problem(0)
        self.assertIn("Return type:", problem.question)

    def test_question_contains_sat_function(self):
        problem = self.ds.get_problem(0)
        self.assertIn("def sat(", problem.question)

    def test_ground_truth_has_sat_key(self):
        problem = self.ds.get_problem(0)
        self.assertIn("sat", problem.ground_truth)

    def test_ground_truth_has_ans_type_key(self):
        problem = self.ds.get_problem(0)
        self.assertIn("ans_type", problem.ground_truth)

    def test_ground_truth_has_name_key(self):
        problem = self.ds.get_problem(0)
        self.assertIn("name", problem.ground_truth)

    def test_metadata_has_module(self):
        problem = self.ds.get_problem(0)
        self.assertIn("module", problem.metadata)
        self.assertEqual(problem.metadata["module"], "study.py")

    def test_index_matches(self):
        for i in range(len(self.ds)):
            self.assertEqual(self.ds.get_problem(i).index, i)


# ─────────────────────────────────────────────────────────────────────────────
# ProgrammingPuzzles evaluation
# ─────────────────────────────────────────────────────────────────────────────

class TestProgrammingPuzzlesEvaluate(unittest.TestCase):

    def setUp(self):
        self.ds = ProgrammingPuzzles()

    def test_correct_integer(self):
        result = self.ds.evaluate_answer("42", _gt())
        self.assertTrue(result.is_correct)
        self.assertEqual(result.score, 1.0)

    def test_wrong_integer(self):
        result = self.ds.evaluate_answer("99", _gt())
        self.assertFalse(result.is_correct)
        self.assertEqual(result.score, 0.0)

    def test_correct_string(self):
        gt = _gt("def sat(s: str):\n    return s == 'hello'", "str")
        result = self.ds.evaluate_answer('"hello"', gt)
        self.assertTrue(result.is_correct)

    def test_wrong_string(self):
        gt = _gt("def sat(s: str):\n    return s == 'hello'", "str")
        result = self.ds.evaluate_answer('"world"', gt)
        self.assertFalse(result.is_correct)

    def test_correct_list(self):
        gt = _gt(
            "def sat(lst: List[int]):\n    return sum(lst) == 10 and len(lst) == 4",
            "List[int]",
        )
        result = self.ds.evaluate_answer("[1, 2, 3, 4]", gt)
        self.assertTrue(result.is_correct)

    def test_correct_bool(self):
        gt = _gt("def sat(b: bool):\n    return b is True", "bool")
        result = self.ds.evaluate_answer("True", gt)
        self.assertTrue(result.is_correct)

    def test_answer_in_code_block(self):
        result = self.ds.evaluate_answer("```python\n42\n```", _gt())
        self.assertTrue(result.is_correct)

    def test_answer_after_reasoning(self):
        result = self.ds.evaluate_answer(
            "Let me think. The number that equals 42 is:\n42", _gt()
        )
        self.assertTrue(result.is_correct)

    def test_unparseable_returns_false_with_error(self):
        result = self.ds.evaluate_answer("I have no idea what to return", _gt())
        self.assertFalse(result.is_correct)
        self.assertIn("error", result.details)

    def test_invalid_ground_truth_type(self):
        result = self.ds.evaluate_answer("42", "not a dict")
        self.assertFalse(result.is_correct)
        self.assertIn("error", result.details)

    def test_invalid_ground_truth_missing_sat(self):
        result = self.ds.evaluate_answer("42", {"ans_type": "int"})
        self.assertFalse(result.is_correct)
        self.assertIn("error", result.details)

    def test_sat_with_typing_annotation(self):
        gt = _gt(
            "def sat(lst: List[int], n: int = 3):\n    return len(lst) == n and all(x > 0 for x in lst)",
            "List[int]",
        )
        result = self.ds.evaluate_answer("[1, 2, 3]", gt)
        self.assertTrue(result.is_correct)

    def test_details_contain_extracted_literal(self):
        result = self.ds.evaluate_answer("42", _gt())
        self.assertIn("extracted_literal", result.details)

    def test_details_contain_sat_result_on_success(self):
        result = self.ds.evaluate_answer("42", _gt())
        self.assertIn("sat_result", result.details)
        self.assertTrue(result.details["sat_result"])

    def test_real_study_puzzle_correct_answer(self):
        """Verify a known-correct answer passes for the first study puzzle."""
        ds = _loaded(num_samples=1, module="study.py")
        problem = ds.get_problem(0)
        # study puzzle 0: s.count('o') == 1000 and s.count('oo') == 0
        valid = repr("ho" * 1000)
        result = ds.evaluate_answer(valid, problem.ground_truth)
        self.assertTrue(result.is_correct)

    def test_real_study_puzzle_wrong_answer(self):
        ds = _loaded(num_samples=1, module="study.py")
        problem = ds.get_problem(0)
        wrong = repr("o" * 1000)   # has 'oo', so sat returns False
        result = ds.evaluate_answer(wrong, problem.ground_truth)
        self.assertFalse(result.is_correct)


# ─────────────────────────────────────────────────────────────────────────────
# Hooks
# ─────────────────────────────────────────────────────────────────────────────

class TestProgrammingPuzzlesHooks(unittest.TestCase):

    def setUp(self):
        self.ds = ProgrammingPuzzles()

    def test_get_instruction_returns_str(self):
        self.assertIsInstance(self.ds.get_instruction(), str)

    def test_get_instruction_mentions_sat(self):
        self.assertIn("sat", self.ds.get_instruction())

    def test_get_system_prompt_returns_str(self):
        self.assertIsInstance(self.ds.get_system_prompt(), str)

    def test_get_system_prompt_mentions_puzzle(self):
        self.assertIn("puzzle", self.ds.get_system_prompt().lower())


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

class TestProgrammingPuzzlesRegistry(unittest.TestCase):

    def test_registry_contains_key(self):
        self.assertIn("programmingpuzzles", DATASET_REGISTRY)

    def test_registry_class_is_correct(self):
        cls, _ = DATASET_REGISTRY["programmingpuzzles"]
        self.assertIs(cls, ProgrammingPuzzles)

    def test_registry_class_is_dataset_subclass(self):
        cls, _ = DATASET_REGISTRY["programmingpuzzles"]
        self.assertTrue(issubclass(cls, DatasetBase))

    def test_kwargs_extractor_passes_num_samples(self):
        _, extractor = DATASET_REGISTRY["programmingpuzzles"]
        import argparse
        args = argparse.Namespace(pp_num_samples=10, pp_module=None)
        kwargs = extractor(args)
        self.assertEqual(kwargs["num_samples"], 10)

    def test_kwargs_extractor_passes_module(self):
        _, extractor = DATASET_REGISTRY["programmingpuzzles"]
        import argparse
        args = argparse.Namespace(pp_num_samples=50, pp_module="basic.py")
        kwargs = extractor(args)
        self.assertEqual(kwargs["module"], "basic.py")


if __name__ == "__main__":
    unittest.main()
