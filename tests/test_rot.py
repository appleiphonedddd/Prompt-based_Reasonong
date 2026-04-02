"""
Tests for the Reversal of Thought (RoT) baseline implementation.

Tests cover:
- Candidate generation (Stage 1)
- Pairwise preference evaluation (Stage 2)
- Transitive closure in the preference matrix
- Optimal candidate selection
- Cognitive Preference Manager / CPM (Stage 3)
  - Task definition extraction
  - Knowledge boundary assessment (known vs unknown)
  - Solution logic aggregation (known tasks)
  - Stylistic template aggregation (unknown tasks)
  - CPM integration in full pipeline
- Instantiation response parsing (Stage 4)
- Full pipeline end-to-end execution (with and without CPM)

Author: Egor Morozov
"""

import unittest
from typing import List

from models.base import BaseLLM, LLMResponse
from baseline.RoT.rot import (
    RoT,
    BaseEmbeddingModel,
    LLMBasedSimilarity,
)


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


class MockEmbeddingModel(BaseEmbeddingModel):
    """Mock embedding model that returns a fixed similarity score."""

    def __init__(self, fixed_score: float = 0.8):
        self.fixed_score = fixed_score
        self.call_count = 0
        self.last_text_a = None
        self.last_text_b = None

    def compute_similarity(self, text_a: str, text_b: str) -> float:
        self.call_count += 1
        self.last_text_a = text_a
        self.last_text_b = text_b
        return self.fixed_score


# ─────────────────────────────────────────────────
# Original tests (Stage 1, 2, 4 — no CPM changes)
# ─────────────────────────────────────────────────

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
        # CPM defaults
        self.assertIsNone(baseline.embedding_model)
        self.assertEqual(baseline.similarity_threshold, 0.7)
        self.assertIsNone(baseline.task_prompt)
        self.assertFalse(baseline.cpm_enabled)

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

    def test_initialization_with_cpm(self):
        llm = SequentialMockLLM([])
        emb = MockEmbeddingModel(0.8)
        baseline = RoT(
            llm,
            embedding_model=emb,
            similarity_threshold=0.75,
            task_prompt="Solve the math problem.",
        )

        self.assertIs(baseline.embedding_model, emb)
        self.assertEqual(baseline.similarity_threshold, 0.75)
        self.assertEqual(baseline.task_prompt, "Solve the math problem.")
        self.assertTrue(baseline.cpm_enabled)

    def test_cpm_not_enabled_without_task_prompt(self):
        llm = SequentialMockLLM([])
        emb = MockEmbeddingModel(0.8)
        baseline = RoT(llm, embedding_model=emb)
        # embedding_model set but no task_prompt → CPM not enabled
        self.assertFalse(baseline.cpm_enabled)

    def test_cpm_not_enabled_without_embedding_model(self):
        llm = SequentialMockLLM([])
        baseline = RoT(llm, task_prompt="Some task")
        # task_prompt set but no embedding_model → CPM not enabled
        self.assertFalse(baseline.cpm_enabled)


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
        responses = ["cand0", "cand1", "cand2", "A", "B", "A"]
        llm = SequentialMockLLM(responses)
        baseline = RoT(llm, warmup=3)

        candidates = baseline.generate_candidates()
        p_pre = baseline.build_preference_matrix(candidates)

        self.assertEqual(p_pre[(0, 1)], 1.0)
        self.assertEqual(p_pre[(1, 2)], 1.0)
        self.assertEqual(p_pre[(0, 2)], 1.0)
        self.assertEqual(p_pre[(2, 0)], 1.0)


class TestRoTOptimalSelection(unittest.TestCase):
    """Test the optimal candidate selection."""

    def test_selects_preferred_candidate(self):
        llm = SequentialMockLLM([])
        baseline = RoT(llm, warmup=2)

        candidates = ["weak_candidate", "strong_candidate"]
        p_pre = {(0, 1): 0.0, (1, 0): 1.0}

        idx, text = baseline.select_optimal(candidates, p_pre)
        self.assertEqual(idx, 1)
        self.assertEqual(text, "strong_candidate")

    def test_single_candidate(self):
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
        llm = SequentialMockLLM([])
        baseline = RoT(llm)

        response = "The solution is 24."
        answer, thinking = baseline.parse_instantiation_response(response)

        self.assertEqual(answer, "The solution is 24.")


# ─────────────────────────────────────────────────
# CPM-specific tests (Stage 3)
# ─────────────────────────────────────────────────

class TestRoTTaskDefinitionExtraction(unittest.TestCase):
    """Test the task definition extraction used by CPM."""

    def test_extract_with_marker(self):
        text = (
            "Task Definition: The task is to solve a math problem.\n\n"
            "Logical Pseudocode: For each permutation..."
        )
        llm = SequentialMockLLM([])
        baseline = RoT(llm)
        result = baseline.extract_task_definition(text)
        self.assertIn("solve a math problem", result)
        self.assertNotIn("Logical Pseudocode", result)

    def test_extract_with_misspelled_marker(self):
        """The paper's outputs sometimes use 'Defination' instead of 'Definition'."""
        text = (
            "Task Defination: Find a feasible mathematical expression.\n\n"
            "Pseudocode: step 1..."
        )
        llm = SequentialMockLLM([])
        baseline = RoT(llm)
        result = baseline.extract_task_definition(text)
        self.assertIn("feasible mathematical expression", result)

    def test_extract_fallback_no_marker(self):
        text = "This is just a plain prompt without any task definition marker."
        llm = SequentialMockLLM([])
        baseline = RoT(llm)
        result = baseline.extract_task_definition(text)
        # Should return the text itself (truncated to 500 chars)
        self.assertIn("plain prompt", result)


class TestRoTKnowledgeBoundary(unittest.TestCase):
    """Test knowledge boundary assessment (CPM Algorithm 2, Eq. 7)."""

    def test_known_boundary(self):
        """Similarity >= threshold → known."""
        llm = SequentialMockLLM([])
        emb = MockEmbeddingModel(fixed_score=0.85)
        baseline = RoT(
            llm,
            embedding_model=emb,
            similarity_threshold=0.7,
            task_prompt="Solve 24 game",
        )

        boundary, score = baseline.compute_knowledge_boundary(
            "Solve the 24 game",
            "Task Definition: Find expression equaling 24.\n\nLogical Pseudocode: ...",
        )
        self.assertEqual(boundary, "known")
        self.assertAlmostEqual(score, 0.85)
        self.assertEqual(emb.call_count, 1)

    def test_unknown_boundary(self):
        """Similarity < threshold → unknown."""
        llm = SequentialMockLLM([])
        emb = MockEmbeddingModel(fixed_score=0.5)
        baseline = RoT(
            llm,
            embedding_model=emb,
            similarity_threshold=0.7,
            task_prompt="Solve math word problems",
        )

        boundary, score = baseline.compute_knowledge_boundary(
            "Solve math word problems",
            "Task Definition: Construct a Python function.\n\nLogical Pseudocode: ...",
        )
        self.assertEqual(boundary, "unknown")
        self.assertAlmostEqual(score, 0.5)

    def test_boundary_at_threshold(self):
        """Similarity exactly at threshold → known (>= check)."""
        llm = SequentialMockLLM([])
        emb = MockEmbeddingModel(fixed_score=0.7)
        baseline = RoT(
            llm,
            embedding_model=emb,
            similarity_threshold=0.7,
        )

        boundary, _ = baseline.compute_knowledge_boundary("A", "B")
        self.assertEqual(boundary, "known")

    def test_custom_threshold(self):
        """Test with a different threshold value."""
        llm = SequentialMockLLM([])
        emb = MockEmbeddingModel(fixed_score=0.65)
        baseline = RoT(
            llm,
            embedding_model=emb,
            similarity_threshold=0.6,
        )

        boundary, _ = baseline.compute_knowledge_boundary("A", "B")
        self.assertEqual(boundary, "known")


class TestRoTAggregation(unittest.TestCase):
    """Test CPM aggregation methods (known and unknown paths)."""

    def test_aggregate_known(self):
        """Known task aggregation calls LLM with CPM_KNOWN_PROMPT."""
        llm = SequentialMockLLM(["Merged prompt: solve 24 with arithmetic ops."])
        baseline = RoT(llm)

        result = baseline.aggregate_known(
            task_prompt="Let's play 24 game.",
            llm_taste="Task Definition: Find expression equaling 24.",
        )
        self.assertIn("Merged prompt", result)
        self.assertEqual(llm.call_counter, 1)

    def test_aggregate_unknown(self):
        """Unknown task aggregation calls LLM with CPM_UNKNOWN_PROMPT."""
        llm = SequentialMockLLM(["Enhanced template with meta-cognitive elements."])
        baseline = RoT(llm)

        result = baseline.aggregate_unknown(
            task_prompt="Solve multilingual math problems.",
            llm_taste="Task Definition: Parse equations.",
        )
        self.assertIn("Enhanced template", result)
        self.assertEqual(llm.call_counter, 1)

    def test_run_cpm_known_path(self):
        """Full CPM run: known task → calls _aggregate_known."""
        llm = SequentialMockLLM(["Refined known prompt"])
        emb = MockEmbeddingModel(fixed_score=0.85)
        baseline = RoT(
            llm,
            embedding_model=emb,
            similarity_threshold=0.7,
            task_prompt="Solve 24",
        )

        p_final, boundary, similarity = baseline._run_cpm(
            "Solve 24",
            "Task Definition: Make 24 from numbers.\n\nLogical Pseudocode: ...",
        )
        self.assertEqual(boundary, "known")
        self.assertAlmostEqual(similarity, 0.85)
        self.assertEqual(p_final, "Refined known prompt")

    def test_run_cpm_unknown_path(self):
        """Full CPM run: unknown task → calls _aggregate_unknown."""
        llm = SequentialMockLLM(["Refined unknown prompt"])
        emb = MockEmbeddingModel(fixed_score=0.4)
        baseline = RoT(
            llm,
            embedding_model=emb,
            similarity_threshold=0.7,
            task_prompt="Write Python puzzles",
        )

        p_final, boundary, similarity = baseline._run_cpm(
            "Write Python puzzles",
            "Task Definition: Sort words alphabetically.\n\nLogical Pseudocode: ...",
        )
        self.assertEqual(boundary, "unknown")
        self.assertAlmostEqual(similarity, 0.4)
        self.assertEqual(p_final, "Refined unknown prompt")


class TestLLMBasedSimilarity(unittest.TestCase):
    """Test the LLM-based similarity fallback."""

    def test_valid_score(self):
        llm = SequentialMockLLM(["0.85"])
        emb = LLMBasedSimilarity(llm)
        score = emb.compute_similarity("task A", "task B")
        self.assertAlmostEqual(score, 0.85)

    def test_clamping_above_one(self):
        llm = SequentialMockLLM(["1.5"])
        emb = LLMBasedSimilarity(llm)
        score = emb.compute_similarity("task A", "task B")
        self.assertAlmostEqual(score, 1.0)

    def test_clamping_below_zero(self):
        llm = SequentialMockLLM(["-0.3"])
        emb = LLMBasedSimilarity(llm)
        score = emb.compute_similarity("task A", "task B")
        self.assertAlmostEqual(score, 0.0)

    def test_unparseable_response(self):
        llm = SequentialMockLLM(["I cannot determine similarity"])
        emb = LLMBasedSimilarity(llm)
        score = emb.compute_similarity("task A", "task B")
        # Falls back to 0.5
        self.assertAlmostEqual(score, 0.5)


# ─────────────────────────────────────────────────
# End-to-end tests (with and without CPM)
# ─────────────────────────────────────────────────

class TestRoTEndToEnd(unittest.TestCase):
    """End-to-end tests for the full RoT pipeline."""

    def test_full_pipeline_warmup_2_no_cpm(self):
        """Test full pipeline without CPM (backward compatible).

        Calls:
        1-2: Candidate generation (2 calls)
        3:   Pairwise preference (1 comparison)
        4:   Instantiation (1 call)
        Total: 4 LLM calls
        """
        responses = [
            "Task: Use 4 numbers to make 24 using arithmetic.",
            "Task: Combine numbers with +,-,*,/ to reach 24.",
            "B",
            "** Thinking **: 2+5=7, 8-7=1, 11+1=12, not 24. Try: (11-5)*(8/2)=24.\n\n"
            "** Answer **: (11 - 5) * (8 / 2) = 24",
        ]
        llm = SequentialMockLLM(responses)
        baseline = RoT(llm, warmup=2)

        response = baseline.run("2, 5, 8, 11")

        self.assertIn("24", response.final_answer)
        self.assertEqual(response.baseline_type, "RoT")
        self.assertEqual(response.metadata["warmup"], 2)
        self.assertEqual(response.metadata["optimal_candidate_index"], 1)
        self.assertEqual(response.num_llm_calls, 4)
        self.assertEqual(response.total_tokens, 80)
        # CPM metadata should show disabled
        self.assertFalse(response.metadata["cpm_enabled"])
        self.assertIsNone(response.metadata["cpm_boundary"])
        self.assertIsNone(response.metadata["cpm_similarity"])

    def test_full_pipeline_warmup_2_with_cpm_known(self):
        """Test full pipeline WITH CPM (known task).

        Calls:
        1-2: Candidate generation (2 calls)
        3:   Pairwise preference (1 comparison)
        4:   CPM aggregation (1 call for _aggregate_known)
        5:   Instantiation (1 call)
        Total: 5 LLM calls
        """
        responses = [
            # Stage 1: 2 candidates
            "Task Definition: Find expression equaling 24.\n\nLogical Pseudocode: ...",
            "Task Definition: Combine 4 numbers to make 24.\n\nLogical Pseudocode: ...",
            # Stage 2: 1 comparison → prefer A
            "A",
            # Stage 3: CPM aggregation (known path)
            "Refined: Find a mathematical expression using four numbers to equal 24.",
            # Stage 4: Instantiation
            "** Answer **: (11 - 5) * (8 / 2) = 24",
        ]
        llm = SequentialMockLLM(responses)
        emb = MockEmbeddingModel(fixed_score=0.85)
        baseline = RoT(
            llm,
            warmup=2,
            embedding_model=emb,
            similarity_threshold=0.7,
            task_prompt="Let's play a game called 24.",
        )

        response = baseline.run("2, 5, 8, 11")

        self.assertIn("24", response.final_answer)
        self.assertEqual(response.num_llm_calls, 5)
        self.assertEqual(response.total_tokens, 100)  # 5 calls * 20 tokens each
        # CPM metadata
        self.assertTrue(response.metadata["cpm_enabled"])
        self.assertEqual(response.metadata["cpm_boundary"], "known")
        self.assertAlmostEqual(response.metadata["cpm_similarity"], 0.85)
        # Intermediate steps should mention CPM
        has_cpm_step = any("CPM" in s and "known" in s for s in response.intermediate_steps)
        self.assertTrue(has_cpm_step)

    def test_full_pipeline_warmup_2_with_cpm_unknown(self):
        """Test full pipeline WITH CPM (unknown task).

        Same call count as known (5), but takes the unknown aggregation path.
        """
        responses = [
            # Stage 1: 2 candidates
            "Task Definition: Parse Python code.\n\nLogical Pseudocode: ...",
            "Task Definition: Evaluate programming puzzles.\n\nLogical Pseudocode: ...",
            # Stage 2: 1 comparison → prefer B
            "B",
            # Stage 3: CPM aggregation (unknown path)
            "Enhanced: Template with meta-cognitive elements for Python puzzles.",
            # Stage 4: Instantiation
            "** Answer **: def solve(): return True",
        ]
        llm = SequentialMockLLM(responses)
        emb = MockEmbeddingModel(fixed_score=0.4)
        baseline = RoT(
            llm,
            warmup=2,
            embedding_model=emb,
            similarity_threshold=0.7,
            task_prompt="Solve Python programming puzzles.",
        )

        response = baseline.run("Find x such that x*2 == 10")

        self.assertEqual(response.num_llm_calls, 5)
        self.assertEqual(response.metadata["cpm_boundary"], "unknown")
        self.assertAlmostEqual(response.metadata["cpm_similarity"], 0.4)

    def test_full_pipeline_warmup_3(self):
        """Test with warmup=3 no CPM → 3 candidates + 3 pairwise + 1 instantiation = 7 calls."""
        responses = [
            "Candidate A definition",
            "Candidate B definition",
            "Candidate C definition",
            "A", "A", "B",
            "** Answer **: 42",
        ]
        llm = SequentialMockLLM(responses)
        baseline = RoT(llm, warmup=3)

        response = baseline.run("test question")

        self.assertEqual(response.final_answer, "42")
        self.assertEqual(response.num_llm_calls, 7)
        self.assertEqual(response.metadata["optimal_candidate_index"], 0)

    def test_per_call_task_prompt_override(self):
        """Test that task_prompt can be overridden per run() call."""
        responses = [
            # Stage 1
            "Task Definition: Generic task.\n\nLogical Pseudocode: ...",
            # Stage 2: (warmup=1, no pairwise)
            # Stage 3: CPM aggregation
            "Refined with per-call task prompt",
            # Stage 4: Instantiation
            "** Answer **: 99",
        ]
        llm = SequentialMockLLM(responses)
        emb = MockEmbeddingModel(fixed_score=0.9)
        baseline = RoT(
            llm,
            warmup=1,
            embedding_model=emb,
            task_prompt="Default task prompt",
        )

        # Override task_prompt at call time
        response = baseline.run("Q", task_prompt="Overridden task prompt")

        self.assertEqual(response.final_answer, "99")
        self.assertEqual(response.metadata["cpm_boundary"], "known")

    def test_cpm_skipped_when_no_embedding_model(self):
        """Verify CPM is skipped when embedding_model is None."""
        responses = [
            "candidate_def",
            "** Answer **: done",
        ]
        llm = SequentialMockLLM(responses)
        baseline = RoT(llm, warmup=1, task_prompt="Some task")

        response = baseline.run("test")

        # Only 2 calls: 1 candidate + 1 instantiation (no CPM)
        self.assertEqual(response.num_llm_calls, 2)
        self.assertFalse(response.metadata["cpm_enabled"])
        has_skipped = any("Skipped" in s for s in response.intermediate_steps)
        self.assertTrue(has_skipped)

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

    def test_intermediate_steps_populated_with_cpm(self):
        """Verify intermediate steps capture all four stages when CPM is active."""
        responses = [
            "candidate_def",
            # CPM aggregation
            "Aggregated result",
            # Instantiation
            "** Answer **: done",
        ]
        llm = SequentialMockLLM(responses)
        emb = MockEmbeddingModel(fixed_score=0.9)
        baseline = RoT(
            llm,
            warmup=1,
            embedding_model=emb,
            task_prompt="Task",
        )

        response = baseline.run("test")

        has_stage1 = any("Stage 1" in s for s in response.intermediate_steps)
        has_stage2 = any("Stage 2" in s for s in response.intermediate_steps)
        has_stage3 = any("Stage 3" in s or "CPM" in s for s in response.intermediate_steps)
        has_stage4 = any("Stage 4" in s for s in response.intermediate_steps)

        self.assertTrue(has_stage1)
        self.assertTrue(has_stage2)
        self.assertTrue(has_stage3)
        self.assertTrue(has_stage4)

    def test_intermediate_steps_without_cpm(self):
        """Verify intermediate steps show CPM skipped when disabled."""
        responses = [
            "candidate_def",
            "** Answer **: done",
        ]
        llm = SequentialMockLLM(responses)
        baseline = RoT(llm, warmup=1)

        response = baseline.run("test")

        has_stage1 = any("Stage 1" in s for s in response.intermediate_steps)
        has_stage2 = any("Stage 2" in s for s in response.intermediate_steps)
        has_cpm_skipped = any("Skipped" in s for s in response.intermediate_steps)
        has_stage4 = any("Stage 4" in s for s in response.intermediate_steps)

        self.assertTrue(has_stage1)
        self.assertTrue(has_stage2)
        self.assertTrue(has_cpm_skipped)
        self.assertTrue(has_stage4)

    def test_reasoning_trace_includes_cpm_info(self):
        """Verify the reasoning trace includes CPM boundary info when enabled."""
        responses = [
            "Task Definition: Something.\n\nLogical Pseudocode: ...",
            "Aggregated prompt",
            "** Thinking **: step by step\n\n** Answer **: 42",
        ]
        llm = SequentialMockLLM(responses)
        emb = MockEmbeddingModel(fixed_score=0.85)
        baseline = RoT(
            llm,
            warmup=1,
            embedding_model=emb,
            task_prompt="Original task",
        )

        response = baseline.run("Q")

        self.assertIn("CPM", response.reasoning_trace)
        self.assertIn("known", response.reasoning_trace)
        self.assertIn("0.8500", response.reasoning_trace)
        self.assertIn("P_final", response.reasoning_trace)


class TestRoTEndToEndWithCPMWarmup3(unittest.TestCase):
    """End-to-end test with warmup=3 and CPM enabled.

    Call breakdown:
    - 3 candidate generation
    - 3 pairwise comparisons
    - 1 CPM aggregation
    - 1 instantiation
    Total: 8 LLM calls
    """

    def test_full_pipeline_warmup_3_with_cpm(self):
        responses = [
            # Stage 1: 3 candidates
            "Task Definition: Definition A.\n\nLogical Pseudocode: ...",
            "Task Definition: Definition B.\n\nLogical Pseudocode: ...",
            "Task Definition: Definition C.\n\nLogical Pseudocode: ...",
            # Stage 2: 3 pairwise (0v1, 0v2, 1v2)
            "A",   # 0 > 1
            "A",   # 0 > 2
            "B",   # 2 > 1
            # Stage 3: CPM aggregation
            "Final refined prompt combining best of both.",
            # Stage 4: Instantiation
            "** Answer **: 42",
        ]
        llm = SequentialMockLLM(responses)
        emb = MockEmbeddingModel(fixed_score=0.75)
        baseline = RoT(
            llm,
            warmup=3,
            embedding_model=emb,
            similarity_threshold=0.7,
            task_prompt="Original benchmark prompt",
        )

        response = baseline.run("test question")

        self.assertEqual(response.final_answer, "42")
        self.assertEqual(response.num_llm_calls, 8)
        self.assertEqual(response.total_tokens, 160)  # 8 * 20
        self.assertEqual(response.metadata["optimal_candidate_index"], 0)
        self.assertEqual(response.metadata["cpm_boundary"], "known")
        self.assertTrue(response.metadata["cpm_enabled"])


if __name__ == "__main__":
    unittest.main()