"""
Tests for the Reversal of Thought (RoT) baseline implementation.

Tests cover:
- Candidate generation (Stage 1)
- Pairwise preference evaluation (Stage 2)
- Transitive closure in the preference matrix
- Optimal candidate selection
- Instantiation response parsing (Stage 3)
- Full pipeline end-to-end execution

Author: Egor Morozov
"""

import unittest
from typing import List

from models.base import BaseLLM, LLMResponse
from baseline.RoT.rot import RoT


class SequentialMockLLM(BaseLLM):
    """Mock LLM that returns pre-defined responses in sequence."""

    def __init__(self, responses: List[str]):
        super().__init__(api_key="dummy", model="mock-rot")
        self.responses = responses
        self.call_counter = 0

    def generate(self, prompt: str, temperature: float = 0) -> LLMResponse:
        if self.call_counter < len(self.responses):
            content = self.responses[self.call_counter]
            self.call_counter += 1
        else:
            content = ""

        return LLMResponse(
            content=content,
            model_name="mock-rot",
            input_tokens=10,
            output_tokens=10,
        )


class TestRoTInitialization(unittest.TestCase):
    """Test RoT initialization and configuration."""

    def test_default_initialization(self):
        llm = SequentialMockLLM([])
        baseline = RoT(llm)

        self.assertEqual(baseline.baseline_name, "RoT")
        self.assertEqual(baseline.warmup, 5)
        self.assertEqual(baseline.candidate_temperature, 0.7)
        self.assertEqual(baseline.instantiation_temperature, 0.1)
        self.assertIn("Input:", baseline.demos)

    def test_custom_initialization(self):
        llm = SequentialMockLLM([])
        baseline = RoT(
            llm,
            warmup=3,
            candidate_temperature=0.9,
            instantiation_temperature=0.2,
            demos="Input:test; Output:result",
        )

        self.assertEqual(baseline.warmup, 3)
        self.assertEqual(baseline.candidate_temperature, 0.9)
        self.assertEqual(baseline.instantiation_temperature, 0.2)
        self.assertEqual(baseline.demos, "Input:test; Output:result")


class TestRoTPreferenceEvaluation(unittest.TestCase):
    """Test the pairwise preference evaluation logic."""

    def test_preference_chooses_a(self):
        llm = SequentialMockLLM(["A"])
        baseline = RoT(llm, warmup=1)
        score = baseline.evaluate_preference("candidate_a", "candidate_b")
        self.assertEqual(score, 1.0)

    def test_preference_chooses_b(self):
        llm = SequentialMockLLM(["B"])
        baseline = RoT(llm, warmup=1)
        score = baseline.evaluate_preference("candidate_a", "candidate_b")
        self.assertEqual(score, 0.0)

    def test_preference_ambiguous_fallback(self):
        llm = SequentialMockLLM(["I think both are good"])
        baseline = RoT(llm, warmup=1)
        score = baseline.evaluate_preference("candidate_a", "candidate_b")
        # No A or B found, should return 0.5
        self.assertEqual(score, 0.5)

    def test_preference_with_verbose_response(self):
        llm = SequentialMockLLM(["A is clearly better because..."])
        baseline = RoT(llm, warmup=1)
        score = baseline.evaluate_preference("candidate_a", "candidate_b")
        self.assertEqual(score, 1.0)


class TestRoTPreferenceMatrix(unittest.TestCase):
    """Test preference matrix construction and transitive closure."""

    def test_two_candidates_matrix(self):
        """With 2 candidates, only 1 pairwise comparison is needed."""
        # Candidate generation: 2 candidates
        # Then 1 pairwise comparison: A vs B → "A"
        responses = ["cand1", "cand2", "A"]
        llm = SequentialMockLLM(responses)
        baseline = RoT(llm, warmup=2)

        candidates = baseline.generate_candidates()
        self.assertEqual(len(candidates), 2)

        p_pre = baseline.build_preference_matrix(candidates)
        self.assertEqual(p_pre[(0, 1)], 1.0)
        self.assertEqual(p_pre[(1, 0)], 0.0)

    def test_three_candidates_transitivity(self):
        """With 3 candidates: 3 pairwise comparisons + transitivity."""
        # 3 pairwise: (0,1)→A, (0,2)→B, (1,2)→A
        responses = ["cand0", "cand1", "cand2", "A", "B", "A"]
        llm = SequentialMockLLM(responses)
        baseline = RoT(llm, warmup=3)

        candidates = baseline.generate_candidates()
        p_pre = baseline.build_preference_matrix(candidates)

        # Direct preferences (before transitivity)
        self.assertEqual(p_pre[(0, 1)], 1.0)  # 0 > 1
        self.assertEqual(p_pre[(1, 2)], 1.0)  # 1 > 2

        # After transitivity: 0>1 and 1>2 implies 0>2
        # p_pre[(0,2)] = max(initial 0.0, 1.0*1.0) = 1.0
        self.assertEqual(p_pre[(0, 2)], 1.0)

        # Transitivity also strengthens (2,0): 2>1 (0.0) * 1>0 (0.0) = 0.0
        # so (2,0) stays at its direct value of 1.0 (from B chosen in 0v2)
        self.assertEqual(p_pre[(2, 0)], 1.0)


class TestRoTOptimalSelection(unittest.TestCase):
    """Test the optimal candidate selection."""

    def test_selects_preferred_candidate(self):
        """Should select the candidate with highest average preference."""
        llm = SequentialMockLLM([])
        baseline = RoT(llm, warmup=2)

        candidates = ["weak_candidate", "strong_candidate"]
        # strong_candidate (index 1) is preferred over weak (index 0)
        p_pre = {(0, 1): 0.0, (1, 0): 1.0}

        idx, text = baseline.select_optimal(candidates, p_pre)
        self.assertEqual(idx, 1)
        self.assertEqual(text, "strong_candidate")

    def test_single_candidate(self):
        """With only 1 candidate, it should be selected."""
        llm = SequentialMockLLM([])
        baseline = RoT(llm, warmup=1)

        candidates = ["only_candidate"]
        p_pre = {}

        idx, text = baseline.select_optimal(candidates, p_pre)
        self.assertEqual(idx, 0)
        self.assertEqual(text, "only_candidate")


class TestRoTResponseParsing(unittest.TestCase):
    """Test the instantiation response parsing."""

    def test_parse_standard_format(self):
        llm = SequentialMockLLM([])
        baseline = RoT(llm)

        response = (
            "** Thinking **: First we add 2+5=7, then 8-7=1, then 11*1=11.\n\n"
            "** Answer **: (2+5) * 8 - 11 = 29"
        )
        answer, thinking = baseline.parse_instantiation_response(response)

        self.assertIn("2+5", thinking)
        self.assertIn("(2+5)", answer)

    def test_parse_no_thinking(self):
        llm = SequentialMockLLM([])
        baseline = RoT(llm)

        response = "** Answer **: 42"
        answer, thinking = baseline.parse_instantiation_response(response)

        self.assertEqual(answer, "42")
        self.assertEqual(thinking, "")

    def test_parse_fallback_no_markers(self):
        """When no format markers are found, use the last line."""
        llm = SequentialMockLLM([])
        baseline = RoT(llm)

        response = "The solution is 24."
        answer, thinking = baseline.parse_instantiation_response(response)

        self.assertEqual(answer, "The solution is 24.")


class TestRoTEndToEnd(unittest.TestCase):
    """End-to-end tests for the full RoT pipeline."""

    def test_full_pipeline_warmup_2(self):
        """Test full pipeline with warmup=2 (simplest non-trivial case).

        Calls:
        1-2: Candidate generation (2 calls)
        3:   Pairwise preference (1 comparison)
        4:   Instantiation (1 call)
        Total: 4 LLM calls
        """
        responses = [
            # Stage 1: 2 candidates
            "Task: Use 4 numbers to make 24 using arithmetic.",
            "Task: Combine numbers with +,-,*,/ to reach 24.",
            # Stage 2: 1 pairwise comparison → prefer B (candidate 2)
            "B",
            # Stage 3: Instantiation answer
            "** Thinking **: 2+5=7, 8-7=1, 11+1=12, not 24. Try: (11-5)*(8/2)=24.\n\n"
            "** Answer **: (11 - 5) * (8 / 2) = 24",
        ]
        llm = SequentialMockLLM(responses)
        baseline = RoT(llm, warmup=2)

        response = baseline.run("2, 5, 8, 11")

        # Verify answer was extracted
        self.assertIn("24", response.final_answer)

        # Verify metadata
        self.assertEqual(response.baseline_type, "RoT")
        self.assertEqual(response.metadata["warmup"], 2)
        self.assertEqual(response.metadata["optimal_candidate_index"], 1)

        # Verify LLM call count: 2 candidates + 1 comparison + 1 instantiation = 4
        self.assertEqual(response.num_llm_calls, 4)

        # Verify token tracking
        self.assertEqual(response.total_tokens, 80)  # 4 calls * (10+10)

    def test_full_pipeline_warmup_3(self):
        """Test with warmup=3 → 3 candidates + 3 pairwise + 1 instantiation = 7 calls."""
        responses = [
            # Stage 1: 3 candidates
            "Candidate A definition",
            "Candidate B definition",
            "Candidate C definition",
            # Stage 2: 3 pairwise comparisons (0v1, 0v2, 1v2)
            "A",  # 0 > 1
            "A",  # 0 > 2
            "B",  # 2 > 1
            # Stage 3: Instantiation
            "** Answer **: 42",
        ]
        llm = SequentialMockLLM(responses)
        baseline = RoT(llm, warmup=3)

        response = baseline.run("test question")

        self.assertEqual(response.final_answer, "42")
        self.assertEqual(response.num_llm_calls, 7)
        # Candidate 0 should win (preferred over both 1 and 2)
        self.assertEqual(response.metadata["optimal_candidate_index"], 0)

    def test_counter_reset_between_runs(self):
        """Verify counters reset between runs."""
        responses = [
            "cand1", "cand2", "A",
            "** Answer **: first",
            "cand3", "cand4", "B",
            "** Answer **: second",
        ]
        llm = SequentialMockLLM(responses)
        baseline = RoT(llm, warmup=2)

        r1 = baseline.run("q1")
        self.assertEqual(r1.num_llm_calls, 4)

        r2 = baseline.run("q2")
        self.assertEqual(r2.num_llm_calls, 4)

    def test_intermediate_steps_populated(self):
        """Verify intermediate steps capture all three stages."""
        responses = [
            "candidate_def",
            "A",  # not used for comparison with warmup=1
            "** Answer **: done",
        ]
        llm = SequentialMockLLM(responses)
        baseline = RoT(llm, warmup=1)

        response = baseline.run("test")

        # Should have: stage 1 header, candidate, stage 2 header, stage 3
        has_stage1 = any("Stage 1" in s for s in response.intermediate_steps)
        has_stage2 = any("Stage 2" in s for s in response.intermediate_steps)
        has_stage3 = any("Stage 3" in s for s in response.intermediate_steps)

        self.assertTrue(has_stage1)
        self.assertTrue(has_stage2)
        self.assertTrue(has_stage3)


if __name__ == "__main__":
    unittest.main()
