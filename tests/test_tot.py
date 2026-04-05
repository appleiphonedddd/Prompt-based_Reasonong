"""
Tests for the Tree of Thoughts (ToT) baseline implementation.

Tests cover:
- Value enum parsing
- ThoughtNode helpers (path, terminal check)
- Thought generation & parsing
- Remaining-number extraction
- State evaluation with majority vote
- BFS search algorithm
- DFS search algorithm with pruning
- Answer extraction
- Full end-to-end pipeline (BFS & DFS)
- Counter reset between runs
- Metadata completeness

Author: (your name)
"""

import unittest
from typing import List

from models.base import BaseLLM, LLMResponse
from baseline.ToT.tot import ToT, ThoughtNode, Value


# ─────────────────────────────────────────────
#  Mock LLM helpers
# ─────────────────────────────────────────────

class SequentialMockLLM(BaseLLM):
    """Returns pre-defined responses one by one in sequence."""

    def __init__(self, responses: List[str]):
        super().__init__(api_key="dummy", model="mock-tot")
        self.responses = responses
        self.call_counter = 0

    def generate(self, prompt: str, temperature: float = 0) -> LLMResponse:
        content = (
            self.responses[self.call_counter]
            if self.call_counter < len(self.responses)
            else ""
        )
        self.call_counter += 1
        return LLMResponse(
            content=content,
            model_name="mock-tot",
            input_tokens=10,
            output_tokens=10,
        )


class FixedMockLLM(BaseLLM):
    """Always returns the same fixed response."""

    def __init__(self, fixed: str = "likely"):
        super().__init__(api_key="dummy", model="mock-tot-fixed")
        self.fixed = fixed
        self.call_counter = 0

    def generate(self, prompt: str, temperature: float = 0) -> LLMResponse:
        self.call_counter += 1
        return LLMResponse(
            content=self.fixed,
            model_name="mock-tot-fixed",
            input_tokens=10,
            output_tokens=10,
        )


# ─────────────────────────────────────────────
#  1. Value enum
# ─────────────────────────────────────────────

class TestValue(unittest.TestCase):
    """Test Value enum parsing from raw LLM text."""

    def test_sure(self):
        self.assertEqual(Value.from_text("sure"), Value.SURE)
        self.assertEqual(Value.from_text("Sure"), Value.SURE)
        self.assertEqual(Value.from_text("  SURE  "), Value.SURE)
        self.assertEqual(Value.from_text("this is sure"), Value.SURE)

    def test_likely(self):
        self.assertEqual(Value.from_text("likely"), Value.LIKELY)
        self.assertEqual(Value.from_text("Likely"), Value.LIKELY)
        self.assertEqual(Value.from_text("it looks likely"), Value.LIKELY)

    def test_impossible(self):
        self.assertEqual(Value.from_text("impossible"), Value.IMPOSSIBLE)
        self.assertEqual(Value.from_text("IMPOSSIBLE"), Value.IMPOSSIBLE)
        self.assertEqual(Value.from_text("this is impossible"), Value.IMPOSSIBLE)

    def test_ambiguous_defaults_to_likely(self):
        """Unknown text should default to LIKELY."""
        self.assertEqual(Value.from_text("maybe"), Value.LIKELY)
        self.assertEqual(Value.from_text(""), Value.LIKELY)
        self.assertEqual(Value.from_text("I don't know"), Value.LIKELY)

    def test_numeric_values(self):
        self.assertEqual(Value.SURE.value, 3)
        self.assertEqual(Value.LIKELY.value, 2)
        self.assertEqual(Value.IMPOSSIBLE.value, 0)


# ─────────────────────────────────────────────
#  2. ThoughtNode
# ─────────────────────────────────────────────

class TestThoughtNode(unittest.TestCase):
    """Test ThoughtNode dataclass helpers."""

    def test_default_fields(self):
        node = ThoughtNode(state="4 9 10 13")
        self.assertEqual(node.state, "4 9 10 13")
        self.assertEqual(node.thought, "")
        self.assertEqual(node.depth, 0)
        self.assertIsNone(node.parent)
        self.assertEqual(node.value_score, 0.0)
        self.assertEqual(node.children, [])

    def test_path_thoughts_single_node(self):
        """Root node with no thought → empty path."""
        root = ThoughtNode(state="4 9 10 13", thought="", depth=0)
        self.assertEqual(root.path_thoughts(), [])

    def test_path_thoughts_chain(self):
        """Path should collect thoughts from root to current node."""
        root  = ThoughtNode(state="4 9 10 13", thought="",                   depth=0)
        child = ThoughtNode(state="4 4 10",    thought="13 - 9 = 4",         depth=1, parent=root)
        grand = ThoughtNode(state="4 6",       thought="10 - 4 = 6",         depth=2, parent=child)
        leaf  = ThoughtNode(state="24",        thought="4 * 6 = 24 (left: 24)", depth=3, parent=grand)

        path = leaf.path_thoughts()
        self.assertEqual(len(path), 3)
        self.assertEqual(path[0], "13 - 9 = 4")
        self.assertEqual(path[1], "10 - 4 = 6")
        self.assertEqual(path[2], "4 * 6 = 24 (left: 24)")

    def test_is_terminal_at_max_depth(self):
        node = ThoughtNode(state="4 6", depth=3)
        self.assertTrue(node.is_terminal(max_depth=3))

    def test_is_not_terminal_below_max_depth(self):
        node = ThoughtNode(state="4 6", depth=2)
        self.assertFalse(node.is_terminal(max_depth=3))


# ─────────────────────────────────────────────
#  3. Initialization
# ─────────────────────────────────────────────

class TestToTInitialization(unittest.TestCase):
    """Test ToT constructor and attribute assignment."""

    def test_default_params(self):
        llm = FixedMockLLM()
        tot = ToT(llm)

        self.assertEqual(tot.baseline_name, "ToT")
        self.assertEqual(tot.n_generate_sample, 5)
        self.assertEqual(tot.n_evaluate_sample, 3)
        self.assertEqual(tot.breadth_limit, 5)
        self.assertEqual(tot.max_steps, 3)
        self.assertEqual(tot.search_algorithm, "bfs")
        self.assertEqual(tot.value_threshold, 1.0)
        self.assertAlmostEqual(tot.propose_temperature, 0.7)
        self.assertAlmostEqual(tot.value_temperature, 0.0)

    def test_custom_params(self):
        llm = FixedMockLLM()
        tot = ToT(
            llm,
            n_generate_sample=3,
            n_evaluate_sample=1,
            breadth_limit=2,
            max_steps=5,
            search_algorithm="dfs",
            value_threshold=2.0,
            propose_temperature=0.5,
            value_temperature=0.1,
        )

        self.assertEqual(tot.n_generate_sample, 3)
        self.assertEqual(tot.n_evaluate_sample, 1)
        self.assertEqual(tot.breadth_limit, 2)
        self.assertEqual(tot.max_steps, 5)
        self.assertEqual(tot.search_algorithm, "dfs")
        self.assertAlmostEqual(tot.value_threshold, 2.0)
        self.assertAlmostEqual(tot.propose_temperature, 0.5)
        self.assertAlmostEqual(tot.value_temperature, 0.1)

    def test_custom_prompts(self):
        llm = FixedMockLLM()
        tot = ToT(
            llm,
            propose_prompt="custom propose {numbers}",
            value_prompt="custom value {state}",
            final_answer_prompt="custom answer {numbers} {steps}",
        )

        self.assertIn("custom propose", tot.propose_prompt)
        self.assertIn("custom value", tot.value_prompt)
        self.assertIn("custom answer", tot.final_answer_prompt)


# ─────────────────────────────────────────────
#  4. _parse_thoughts()
# ─────────────────────────────────────────────

class TestParseThoughts(unittest.TestCase):
    """Test thought-line parsing from raw LLM output."""

    def setUp(self):
        self.tot = ToT(FixedMockLLM())

    def test_parses_arithmetic_lines(self):
        raw = (
            "13 - 9 = 4 (left: 4 4 10)\n"
            "10 + 4 = 14 (left: 4 9 14)\n"
            "9 * 4 = 36 (left: 10 13 36)\n"
        )
        thoughts = self.tot.parse_thoughts(raw)
        self.assertEqual(len(thoughts), 3)
        self.assertIn("13 - 9 = 4 (left: 4 4 10)", thoughts)

    def test_skips_blank_lines(self):
        raw = "\n\n13 - 9 = 4 (left: 4 4 10)\n\n\n"
        thoughts = self.tot.parse_thoughts(raw)
        self.assertEqual(len(thoughts), 1)

    def test_skips_lines_without_digits(self):
        raw = (
            "Here are the possible steps:\n"
            "13 - 9 = 4 (left: 4 4 10)\n"
            "Note: These are just examples.\n"
        )
        thoughts = self.tot.parse_thoughts(raw)
        # Only the arithmetic line should survive
        self.assertEqual(len(thoughts), 1)
        self.assertIn("13 - 9 = 4", thoughts[0])

    def test_empty_input(self):
        thoughts = self.tot.parse_thoughts("")
        self.assertEqual(thoughts, [])


# ─────────────────────────────────────────────
#  5. _extract_remaining()
# ─────────────────────────────────────────────

class TestExtractRemaining(unittest.TestCase):
    """Test parsing of remaining numbers from thought strings."""

    def setUp(self):
        self.tot = ToT(FixedMockLLM())

    def test_standard_left_pattern(self):
        thought = "13 - 9 = 4 (left: 4 4 10)"
        result = self.tot.extract_remaining(thought, "4 9 10 13")
        self.assertEqual(result, "4 4 10")

    def test_left_with_colon_space(self):
        thought = "10 - 4 = 6 (left: 4 6)"
        result = self.tot.extract_remaining(thought, "4 4 10")
        self.assertEqual(result, "4 6")

    def test_terminal_equals_24(self):
        thought = "4 * 6 = 24 (left: 24)"
        result = self.tot.extract_remaining(thought, "4 6")
        self.assertEqual(result, "24")

    def test_equals_24_without_left(self):
        thought = "4 * 6 = 24"
        result = self.tot.extract_remaining(thought, "4 6")
        self.assertEqual(result, "24")

    def test_fallback_to_current_state(self):
        """When no pattern is found, fall back to the current state."""
        thought = "some ambiguous text"
        result = self.tot.extract_remaining(thought, "4 6")
        self.assertEqual(result, "4 6")


# ─────────────────────────────────────────────
#  6. _evaluate_state()
# ─────────────────────────────────────────────

class TestEvaluateState(unittest.TestCase):
    """Test state evaluation with majority voting."""

    def test_terminal_state_24_no_llm_call(self):
        """State '24' should return SURE score without any LLM calls."""
        llm = FixedMockLLM("impossible")   # would give 0 if called
        tot = ToT(llm, n_evaluate_sample=3)
        node = ThoughtNode(state="24")

        score = tot.evaluate_state(node)

        self.assertEqual(score, float(Value.SURE.value))
        self.assertEqual(llm.call_counter, 0)  # no LLM calls

    def test_sure_majority(self):
        """Three 'sure' responses → average = 3.0."""
        llm = SequentialMockLLM(["sure", "sure", "sure"])
        tot = ToT(llm, n_evaluate_sample=3)
        node = ThoughtNode(state="4 6")

        score = tot.evaluate_state(node)

        self.assertAlmostEqual(score, 3.0)
        self.assertEqual(llm.call_counter, 3)

    def test_impossible_majority(self):
        """Three 'impossible' responses → average = 0.0."""
        llm = SequentialMockLLM(["impossible", "impossible", "impossible"])
        tot = ToT(llm, n_evaluate_sample=3)
        node = ThoughtNode(state="1 2 3")

        score = tot.evaluate_state(node)

        self.assertAlmostEqual(score, 0.0)

    def test_mixed_votes(self):
        """sure=3, likely=2, impossible=0 → average = (3+2+0)/3 ≈ 1.67."""
        llm = SequentialMockLLM(["sure", "likely", "impossible"])
        tot = ToT(llm, n_evaluate_sample=3)
        node = ThoughtNode(state="4 10")

        score = tot.evaluate_state(node)

        self.assertAlmostEqual(score, (3 + 2 + 0) / 3, places=5)

    def test_single_evaluate_sample(self):
        """n_evaluate_sample=1 → exactly 1 LLM call."""
        llm = SequentialMockLLM(["likely"])
        tot = ToT(llm, n_evaluate_sample=1)
        node = ThoughtNode(state="4 6")

        score = tot.evaluate_state(node)

        self.assertAlmostEqual(score, 2.0)
        self.assertEqual(llm.call_counter, 1)


# ─────────────────────────────────────────────
#  7. BFS
# ─────────────────────────────────────────────

class TestBFS(unittest.TestCase):
    """Test the BFS search algorithm."""

    def _make_tot_bfs(self, responses: List[str], breadth: int = 1) -> ToT:
        llm = SequentialMockLLM(responses)
        return ToT(
            llm,
            n_generate_sample=1,
            n_evaluate_sample=1,
            breadth_limit=breadth,
            max_steps=3,
            search_algorithm="bfs",
        )

    def test_bfs_finds_solution_in_3_steps(self):
        """
        Step 1: generate "13 - 9 = 4 (left: 4 4 10)" → eval "likely"
        Step 2: generate "10 - 4 = 6 (left: 4 6)"    → eval "sure"
        Step 3: generate "4 * 6 = 24 (left: 24)"     → state=="24", no eval
        → solved, break
        """
        responses = [
            "13 - 9 = 4 (left: 4 4 10)",   # step1 generate
            "likely",                        # step1 evaluate "4 4 10"
            "10 - 4 = 6 (left: 4 6)",       # step2 generate
            "sure",                          # step2 evaluate "4 6"
            "4 * 6 = 24 (left: 24)",         # step3 generate
            # no evaluate for "24" (handled internally)
        ]
        tot = self._make_tot_bfs(responses)
        root = ThoughtNode(state="4 9 10 13")

        best, log = tot.bfs(root)

        self.assertIsNotNone(best)
        self.assertEqual(best.state, "24")
        self.assertGreater(best.value_score, 0)

    def test_bfs_prunes_to_breadth_limit(self):
        """
        With breadth_limit=2 and n_generate=2, BFS keeps only top-2 per level.
        n_generate_sample=2, n_evaluate_sample=1, breadth_limit=2, max_steps=1
        Step 1: 2 generates + 2 evaluates
        Frontier should be 2 nodes (equal to breadth_limit).
        """
        responses = [
            # generate returns 2 thoughts in 1 call (propose prompt returns multi-line)
            "13 - 9 = 4 (left: 4 4 10)\n9 + 4 = 13 (left: 10 13 13)",
            "sure",       # eval node 1
            "likely",     # eval node 2
        ]
        llm = SequentialMockLLM(responses)
        tot = ToT(
            llm,
            n_generate_sample=2,
            n_evaluate_sample=1,
            breadth_limit=2,
            max_steps=1,
            search_algorithm="bfs",
        )
        root = ThoughtNode(state="4 9 10 13")
        best, log = tot.bfs(root)

        # frontier limited to breadth=2; best node should be the "sure" one
        self.assertIsNotNone(best)
        self.assertGreaterEqual(best.value_score, 2.0)

    def test_bfs_returns_none_on_no_candidates(self):
        """If LLM returns no parseable thoughts, BFS stops early."""
        responses = ["no digits here at all"]  # parse_thoughts returns []
        tot = self._make_tot_bfs(responses)
        root = ThoughtNode(state="4 9 10 13")

        best, log = tot.bfs(root)

        # No candidates → best_terminal stays None but frontier holds root
        # (root is initial frontier before any expansion)
        any_early_stop = any("stopping early" in l.lower() or "no candidates" in l.lower()
                             for l in log)
        self.assertTrue(any_early_stop)


# ─────────────────────────────────────────────
#  8. DFS
# ─────────────────────────────────────────────

class TestDFS(unittest.TestCase):
    """Test the DFS search algorithm with pruning."""

    def _make_tot_dfs(self, responses: List[str], threshold: float = 1.0) -> ToT:
        llm = SequentialMockLLM(responses)
        return ToT(
            llm,
            n_generate_sample=1,
            n_evaluate_sample=1,
            max_steps=3,
            search_algorithm="dfs",
            value_threshold=threshold,
        )

    def test_dfs_reaches_terminal_state(self):
        """
        DFS chain: root → "4 4 10"(likely) → "4 6"(sure) → "24"(terminal, no eval)
        """
        responses = [
            "13 - 9 = 4 (left: 4 4 10)",   # generate from root
            "likely",                        # eval "4 4 10"
            "10 - 4 = 6 (left: 4 6)",       # generate from "4 4 10"
            "sure",                          # eval "4 6"
            "4 * 6 = 24 (left: 24)",         # generate from "4 6"
            # state=="24" → no eval call
        ]
        tot = self._make_tot_dfs(responses)
        root = ThoughtNode(state="4 9 10 13")

        best, log = tot.dfs(root)

        self.assertIsNotNone(best)
        self.assertEqual(best.state, "24")

    def test_dfs_prunes_impossible_states(self):
        """
        First thought is 'impossible' (score=0 < threshold=1.0) → pruned.
        Second thought is 'sure' → explored and becomes best.
        DFS uses a stack so both children are pushed, impossible one gets pruned.
        """
        responses = [
            # Generate 2 thoughts from root (returned as multi-line)
            "1 + 2 = 3 (left: 3 10 13)\n13 - 9 = 4 (left: 4 4 10)",
            "impossible",   # eval "3 10 13" → score=0
            "sure",         # eval "4 4 10" → score=3
            # DFS pops "4 4 10" (sure, higher score pushed last → popped first)
            "4 * 10 = 40 (left: 4 40)",
            "likely",       # eval "4 40"
            "40 - 4 = 36 (left: 36)",
            "impossible",   # eval "36" → pruned
            # stack empties with "3 10 13" also pruned
        ]
        llm = SequentialMockLLM(responses)
        tot = ToT(
            llm,
            n_generate_sample=2,
            n_evaluate_sample=1,
            max_steps=3,
            search_algorithm="dfs",
            value_threshold=1.0,
        )
        root = ThoughtNode(state="4 9 10 13")
        best, log = tot.dfs(root)

        prune_logs = [l for l in log if "pruning" in l.lower()]
        self.assertGreater(len(prune_logs), 0, "Expected at least one pruning log entry")

    def test_dfs_backtracks_after_dead_end(self):
        """
        All children of first branch are impossible → DFS backtracks and logs pruning.
        """
        responses = [
            "1 + 2 = 3 (left: 3 10 13)",  # only 1 thought from root
            "impossible",                   # pruned immediately
        ]
        tot = self._make_tot_dfs(responses, threshold=1.0)
        root = ThoughtNode(state="4 9 10 13")

        best, log = tot.dfs(root)

        # No solution found; best_terminal remains None or is the root
        prune_logs = [l for l in log if "pruning" in l.lower()]
        self.assertGreater(len(prune_logs), 0)


# ─────────────────────────────────────────────
#  9. End-to-end: run() with BFS
# ─────────────────────────────────────────────

class TestToTEndToEndBFS(unittest.TestCase):
    """End-to-end tests for the full BFS pipeline via run()."""

    def _build_responses(self) -> List[str]:
        """
        Minimal response sequence for BFS:
          n_generate=1, n_evaluate=1, breadth=1, max_steps=3

        Calls (in order):
          1. generate from root         → arithmetic thought
          2. evaluate child state       → "likely"
          3. generate from child        → arithmetic thought
          4. evaluate grandchild state  → "sure"
          5. generate from grandchild   → reaches 24
          (no evaluate for "24")
          6. extract final answer       → equation string
        Total: 6 LLM calls
        """
        return [
            "13 - 9 = 4 (left: 4 4 10)",
            "likely",
            "10 - 4 = 6 (left: 4 6)",
            "sure",
            "4 * 6 = 24 (left: 24)",
            "(13 - 9) * (10 - 4) = 24",
        ]

    def _make_tot(self, responses: List[str]) -> ToT:
        return ToT(
            SequentialMockLLM(responses),
            n_generate_sample=1,
            n_evaluate_sample=1,
            breadth_limit=1,
            max_steps=3,
            search_algorithm="bfs",
        )

    def test_run_returns_baseline_response(self):
        tot = self._make_tot(self._build_responses())
        response = tot.run("4 9 10 13")

        self.assertIsInstance(response.final_answer, str)
        self.assertNotEqual(response.final_answer, "")

    def test_run_llm_call_count(self):
        """6 LLM calls expected: 3 generate + 2 evaluate + 1 extract."""
        tot = self._make_tot(self._build_responses())
        response = tot.run("4 9 10 13")

        self.assertEqual(response.num_llm_calls, 6)

    def test_run_token_tracking(self):
        """Each mock call uses 10 in + 10 out → 6 calls = 120 total tokens."""
        tot = self._make_tot(self._build_responses())
        response = tot.run("4 9 10 13")

        self.assertEqual(response.total_input_tokens, 60)
        self.assertEqual(response.total_output_tokens, 60)
        self.assertEqual(response.total_tokens, 120)

    def test_run_baseline_type(self):
        tot = self._make_tot(self._build_responses())
        response = tot.run("4 9 10 13")

        self.assertEqual(response.baseline_type, "ToT")

    def test_run_metadata_keys(self):
        tot = self._make_tot(self._build_responses())
        response = tot.run("4 9 10 13")

        for key in (
            "search_algorithm", "n_generate_sample", "n_evaluate_sample",
            "breadth_limit", "max_steps", "value_threshold",
            "propose_temperature", "value_temperature",
            "best_path", "best_node_state", "best_node_score",
        ):
            self.assertIn(key, response.metadata, f"Missing metadata key: {key}")

    def test_run_metadatavalues(self):
        tot = self._make_tot(self._build_responses())
        response = tot.run("4 9 10 13")

        self.assertEqual(response.metadata["search_algorithm"], "bfs")
        self.assertEqual(response.metadata["n_generate_sample"], 1)
        self.assertEqual(response.metadata["breadth_limit"], 1)
        self.assertEqual(response.metadata["max_steps"], 3)

    def test_run_intermediate_steps_populated(self):
        tot = self._make_tot(self._build_responses())
        response = tot.run("4 9 10 13")

        self.assertGreater(len(response.intermediate_steps), 0)

    def test_run_reasoning_trace_contains_bfs(self):
        tot = self._make_tot(self._build_responses())
        response = tot.run("4 9 10 13")

        self.assertIn("BFS", response.reasoning_trace)

    def test_counter_reset_between_runs(self):
        """Counters must reset so the second run reports its own stats."""
        responses = self._build_responses() + self._build_responses()
        tot = ToT(
            SequentialMockLLM(responses),
            n_generate_sample=1,
            n_evaluate_sample=1,
            breadth_limit=1,
            max_steps=3,
            search_algorithm="bfs",
        )

        r1 = tot.run("4 9 10 13")
        r2 = tot.run("4 9 10 13")

        self.assertEqual(r1.num_llm_calls, 6)
        self.assertEqual(r2.num_llm_calls, 6)

    def test_no_solution_returns_string(self):
        """If LLM returns no parseable thoughts, run() must still return a string."""
        responses = [
            "no digits here",   # generate step 1 → parse_thoughts returns []
            "(13 - 9) * (10 - 4) = 24",  # extract final answer fallback
        ]
        tot = self._make_tot(responses)
        response = tot.run("4 9 10 13")

        self.assertIsInstance(response.final_answer, str)


# ─────────────────────────────────────────────
#  10. End-to-end: run() with DFS
# ─────────────────────────────────────────────

class TestToTEndToEndDFS(unittest.TestCase):
    """End-to-end tests for the full DFS pipeline via run()."""

    def _make_tot_dfs(self, responses: List[str]) -> ToT:
        return ToT(
            SequentialMockLLM(responses),
            n_generate_sample=1,
            n_evaluate_sample=1,
            max_steps=3,
            search_algorithm="dfs",
            value_threshold=1.0,
        )

    def test_dfs_run_finds_answer(self):
        """
        DFS call sequence (n_gen=1, n_eval=1, max_steps=3):
          1. generate from root         → "13 - 9 = 4 (left: 4 4 10)"
          2. evaluate "4 4 10"          → "likely" (score=2 ≥ threshold)
          3. generate from "4 4 10"     → "10 - 4 = 6 (left: 4 6)"
          4. evaluate "4 6"             → "sure" (score=3 ≥ threshold)
          5. generate from "4 6"        → "4 * 6 = 24 (left: 24)"
          (state "24" → no eval call)
          6. extract final answer
        Total: 6 calls
        """
        responses = [
            "13 - 9 = 4 (left: 4 4 10)",
            "likely",
            "10 - 4 = 6 (left: 4 6)",
            "sure",
            "4 * 6 = 24 (left: 24)",
            "(13 - 9) * (10 - 4) = 24",
        ]
        tot = self._make_tot_dfs(responses)
        response = tot.run("4 9 10 13")

        self.assertEqual(response.num_llm_calls, 6)
        self.assertIsInstance(response.final_answer, str)
        self.assertNotEqual(response.final_answer, "")

    def test_dfs_metadata_algorithm(self):
        responses = [
            "13 - 9 = 4 (left: 4 4 10)",
            "likely",
            "10 - 4 = 6 (left: 4 6)",
            "sure",
            "4 * 6 = 24 (left: 24)",
            "(13 - 9) * (10 - 4) = 24",
        ]
        tot = self._make_tot_dfs(responses)
        response = tot.run("4 9 10 13")

        self.assertEqual(response.metadata["search_algorithm"], "dfs")

    def test_dfs_reasoning_trace_contains_dfs(self):
        responses = [
            "13 - 9 = 4 (left: 4 4 10)",
            "likely",
            "10 - 4 = 6 (left: 4 6)",
            "sure",
            "4 * 6 = 24 (left: 24)",
            "(13 - 9) * (10 - 4) = 24",
        ]
        tot = self._make_tot_dfs(responses)
        response = tot.run("4 9 10 13")

        self.assertIn("DFS", response.reasoning_trace)


if __name__ == "__main__":
    unittest.main()