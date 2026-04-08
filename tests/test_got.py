"""
Unit tests for the Graph of Thoughts (GoT) baseline implementation.

Test coverage:
  - GoT.__init__          : parameter validation and defaults
  - Thought               : dataclass fields and defaults
  - GraphReasoningState   : add / score / keep / prune / volume / to_dict
  - Prompt builders       : _build_generate / score / aggregate / refine prompts
  - Parsers               : _parse_answer (structured + fallback), _parse_score
  - Thought transformations: generate, score, keep_best, aggregate, refine
  - Graph of Operations   : _execute_graph_of_operations integration
  - BaselineResponse      : run() output fields, token counts, metadata
  - Counter reset         : run() resets counters on every invocation
  - Edge cases            : single branch, zero refine rounds, keep_best clamping,
                            malformed LLM output, score-scale correction

Reference: Besta et al., "Graph of Thoughts: Solving Elaborate Problems
with Large Language Models", AAAI-24.

Author: (your name here)
"""

import unittest
from typing import List

from models.base import BaseLLM, LLMResponse
from baseline.GoT.got import (
    GoT,
    Thought,
    ThoughtStatus,
    GraphReasoningState,
)


# ─────────────────────────────────────────────────────────────────────────────
# Test helpers
# ─────────────────────────────────────────────────────────────────────────────

class SequentialMockLLM(BaseLLM):
    """
    Mock LLM that returns pre-defined responses in sequence.

    Once the list is exhausted every subsequent call returns an empty string,
    so tests that accidentally over-call the LLM still produce deterministic
    (and usually failing) behaviour rather than crashing.
    """

    def __init__(self, responses: List[str]) -> None:
        super().__init__(api_key="dummy", model="mock-got")
        self.responses    = responses
        self.call_counter = 0
        self.prompts_seen: List[str] = []

    def generate(self, prompt: str, temperature: float = 0.0) -> LLMResponse:
        self.prompts_seen.append(prompt)
        if self.call_counter < len(self.responses):
            content = self.responses[self.call_counter]
            self.call_counter += 1
        else:
            content = ""
        return LLMResponse(
            content=content,
            model_name="mock-got",
            input_tokens=10,
            output_tokens=10,
        )


def _generate_response(answer: str = "42", reasoning: str = "Step 1: Think.") -> str:
    """Return a well-formed Generate/Aggregate/Refine LLM reply."""
    return f"Reasoning: {reasoning}\nAnswer: {answer}"


def _score_response(score: str = "0.9") -> str:
    """Return a well-formed Score reply."""
    return f"Score: {score}"


def _build_responses(
    *,
    branches:       int   = 3,
    kept:           int   = 2,
    refine_rounds:  int   = 1,
    gen_answer:     str   = "42",
    scores:         List[str] | None = None,
    agg_answer:     str   = "aggregated",
    refine_answer:  str   = "refined",
) -> List[str]:
    """
    Build a canonical response list for a GoT run with the given parameters.

    Layout:
      [0 .. branches-1]              : Generate responses
      [branches .. 2*branches-1]     : Score responses
      [2*branches]                   : Aggregate response   (skipped if kept==1)
      [2*branches+1 .. +refine_rounds]: Refine responses
    """
    if scores is None:
        # Assign decreasing scores so the first branch always wins
        scores = [str(round(0.9 - 0.1 * i, 1)) for i in range(branches)]

    resps: List[str] = []

    # Phase 0 – Generate
    for _ in range(branches):
        resps.append(_generate_response(gen_answer))

    # Phase 1 – Score
    for s in scores:
        resps.append(_score_response(s))

    # Phase 3 – Aggregate  (only when more than one thought survives)
    if kept > 1:
        resps.append(_generate_response(agg_answer, "Aggregate reasoning."))

    # Phase 4 – Refine
    for _ in range(refine_rounds):
        resps.append(_generate_response(refine_answer, "Refined reasoning."))

    return resps


def _expected_calls(branches: int, kept: int, refine_rounds: int) -> int:
    """Return the exact number of LLM calls a GoT run should make."""
    aggregate_calls = 1 if kept > 1 else 0
    return branches + branches + aggregate_calls + refine_rounds


# ─────────────────────────────────────────────────────────────────────────────
# 1. Initialisation
# ─────────────────────────────────────────────────────────────────────────────

class TestGoTInitialization(unittest.TestCase):
    """GoT.__init__ parameter handling and validation."""

    def _make(self, **kwargs) -> GoT:
        return GoT(SequentialMockLLM([]), **kwargs)

    def test_default_values(self):
        got = self._make()
        self.assertEqual(got.baseline_name,    "GoT")
        self.assertEqual(got.num_branches,     5)
        self.assertEqual(got.keep_best,        2)
        self.assertEqual(got.refine_rounds,    1)
        self.assertAlmostEqual(got.gen_temperature,   0.7)
        self.assertAlmostEqual(got.score_temperature, 0.0)
        self.assertAlmostEqual(got.agg_temperature,   0.3)

    def test_custom_values(self):
        got = self._make(
            num_branches=8, keep_best=3, refine_rounds=2,
            gen_temperature=0.9, score_temperature=0.1, agg_temperature=0.5,
        )
        self.assertEqual(got.num_branches,  8)
        self.assertEqual(got.keep_best,     3)
        self.assertEqual(got.refine_rounds, 2)
        self.assertAlmostEqual(got.gen_temperature,   0.9)
        self.assertAlmostEqual(got.score_temperature, 0.1)
        self.assertAlmostEqual(got.agg_temperature,   0.5)

    def test_keep_best_clamped_to_num_branches(self):
        """keep_best must not exceed num_branches."""
        got = self._make(num_branches=3, keep_best=10)
        self.assertEqual(got.keep_best, 3)

    def test_invalid_num_branches_raises(self):
        with self.assertRaises(ValueError):
            self._make(num_branches=0)

    def test_invalid_keep_best_raises(self):
        with self.assertRaises(ValueError):
            self._make(keep_best=0)

    def test_invalid_refine_rounds_raises(self):
        with self.assertRaises(ValueError):
            self._make(refine_rounds=-1)

    def test_llm_stored(self):
        llm = SequentialMockLLM([])
        got = GoT(llm)
        self.assertIs(got.llm, llm)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Thought dataclass
# ─────────────────────────────────────────────────────────────────────────────

class TestThought(unittest.TestCase):
    """Thought vertex fields, defaults, and uniqueness."""

    def test_default_status_is_pending(self):
        t = Thought()
        self.assertEqual(t.status, ThoughtStatus.PENDING)

    def test_default_score_is_zero(self):
        self.assertAlmostEqual(Thought().score, 0.0)

    def test_unique_ids(self):
        ids = {Thought().id for _ in range(50)}
        self.assertEqual(len(ids), 50)

    def test_custom_fields(self):
        t = Thought(content="hello", score=0.8, parent_ids=["p1"])
        self.assertEqual(t.content,    "hello")
        self.assertAlmostEqual(t.score, 0.8)
        self.assertEqual(t.parent_ids, ["p1"])

    def test_repr_contains_id_and_status(self):
        t = Thought(content="test content")
        r = repr(t)
        self.assertIn(t.id, r)
        self.assertIn("PENDING", r)


# ─────────────────────────────────────────────────────────────────────────────
# 3. GraphReasoningState
# ─────────────────────────────────────────────────────────────────────────────

class TestGraphReasoningState(unittest.TestCase):
    """GRS add / score / keep / prune / volume / to_dict."""

    def setUp(self):
        self.grs = GraphReasoningState()

    # ── basic operations ─────────────────────────────────────────────────

    def test_add_and_retrieve(self):
        t = Thought(content="root")
        self.grs.add(t)
        self.assertIs(self.grs.get(t.id), t)

    def test_all_thoughts(self):
        t1, t2 = Thought(), Thought()
        self.grs.add(t1); self.grs.add(t2)
        self.assertCountEqual(self.grs.all_thoughts(), [t1, t2])

    def test_update_score_changes_status(self):
        t = Thought()
        self.grs.add(t)
        self.grs.update_score(t.id, 0.75)
        self.assertAlmostEqual(self.grs.get(t.id).score, 0.75)
        self.assertEqual(self.grs.get(t.id).status, ThoughtStatus.SCORED)

    def test_mark_kept(self):
        t = Thought()
        self.grs.add(t)
        self.grs.mark_kept(t.id)
        self.assertEqual(t.status, ThoughtStatus.KEPT)

    def test_mark_pruned(self):
        t = Thought()
        self.grs.add(t)
        self.grs.mark_pruned(t.id)
        self.assertEqual(t.status, ThoughtStatus.PRUNED)

    def test_mark_refined(self):
        t = Thought()
        self.grs.add(t)
        self.grs.mark_refined(t.id)
        self.assertEqual(t.status, ThoughtStatus.REFINED)

    def test_mark_aggregated(self):
        t = Thought()
        self.grs.add(t)
        self.grs.mark_aggregated(t.id)
        self.assertEqual(t.status, ThoughtStatus.AGGREGATED)

    # ── volume metric (§5 of the paper) ──────────────────────────────────

    def test_volume_of_root_is_zero(self):
        """A thought with no ancestors has volume 0."""
        t = Thought()
        self.grs.add(t)
        self.assertEqual(self.grs.volume(t.id), 0)

    def test_volume_linear_chain(self):
        """Chain A → B → C: volume(C) = 2, volume(B) = 1, volume(A) = 0."""
        a = Thought()
        b = Thought(parent_ids=[a.id])
        c = Thought(parent_ids=[b.id])
        for t in (a, b, c):
            self.grs.add(t)
        self.assertEqual(self.grs.volume(a.id), 0)
        self.assertEqual(self.grs.volume(b.id), 1)
        self.assertEqual(self.grs.volume(c.id), 2)

    def test_volume_aggregation_node(self):
        """Aggregate node merging 3 roots has volume 3."""
        roots = [Thought() for _ in range(3)]
        agg   = Thought(parent_ids=[r.id for r in roots])
        for t in roots:
            self.grs.add(t)
        self.grs.add(agg)
        self.assertEqual(self.grs.volume(agg.id), 3)

    def test_volume_diamond(self):
        """
        Diamond: A → B, A → C, B → D, C → D.
        volume(D) should count A, B, C = 3 (no double-counting).
        """
        a = Thought()
        b = Thought(parent_ids=[a.id])
        c = Thought(parent_ids=[a.id])
        d = Thought(parent_ids=[b.id, c.id])
        for t in (a, b, c, d):
            self.grs.add(t)
        self.assertEqual(self.grs.volume(d.id), 3)

    # ── serialisation ─────────────────────────────────────────────────────

    def test_to_dict_contains_all_ids(self):
        t1, t2 = Thought(), Thought()
        self.grs.add(t1); self.grs.add(t2)
        d = self.grs.to_dict()
        self.assertIn(t1.id, d)
        self.assertIn(t2.id, d)

    def test_to_dict_fields(self):
        t = Thought(content="hello", score=0.5)
        self.grs.add(t)
        entry = self.grs.to_dict()[t.id]
        self.assertEqual(entry["content"], "hello")
        self.assertAlmostEqual(entry["score"], 0.5)
        self.assertIn("status",     entry)
        self.assertIn("parent_ids", entry)
        self.assertIn("volume",     entry)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Prompt builders (Prompter module)
# ─────────────────────────────────────────────────────────────────────────────

class TestGoTPromptBuilders(unittest.TestCase):
    """_build_* methods embed the question and necessary context."""

    def setUp(self):
        self.got = GoT(SequentialMockLLM([]))
        self.q   = "Use 2, 3, 4, 8 to make 24."

    def test_generate_prompt_contains_question(self):
        p = self.got._build_generate_prompt(self.q)
        self.assertIn(self.q, p)
        self.assertIn("Reasoning:", p)
        self.assertIn("Answer:",    p)

    def test_score_prompt_contains_question_and_content(self):
        content = "2 * (3 + 4 + 8 - 24) is wrong."
        p = self.got._build_score_prompt(self.q, content)
        self.assertIn(self.q,   p)
        self.assertIn(content,  p)
        self.assertIn("0.0",    p)
        self.assertIn("1.0",    p)

    def test_aggregate_prompt_contains_all_candidates(self):
        thoughts = [
            Thought(content="Solution 1: 2*3*4=24."),
            Thought(content="Solution 2: 8*(4-1)=24."),
        ]
        p = self.got._build_aggregate_prompt(self.q, thoughts)
        self.assertIn(self.q,                        p)
        self.assertIn("Solution 1: 2*3*4=24.",        p)
        self.assertIn("Solution 2: 8*(4-1)=24.",      p)
        self.assertIn("[Solution 1]",                 p)
        self.assertIn("[Solution 2]",                 p)

    def test_aggregate_prompt_n_count(self):
        thoughts = [Thought() for _ in range(4)]
        p = self.got._build_aggregate_prompt(self.q, thoughts)
        self.assertIn("4", p)

    def test_refine_prompt_contains_question_and_content(self):
        content = "My current best guess: 2+3+4+8=17. Wrong."
        p = self.got._build_refine_prompt(self.q, content)
        self.assertIn(self.q,   p)
        self.assertIn(content,  p)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Parsers
# ─────────────────────────────────────────────────────────────────────────────

class TestGoTParsers(unittest.TestCase):
    """_parse_answer and _parse_score edge cases."""

    def setUp(self):
        self.got = GoT(SequentialMockLLM([]))

    # ── _parse_answer ─────────────────────────────────────────────────────

    def test_parse_answer_structured(self):
        raw = "Reasoning: Step 1: add.\nAnswer: 24"
        reasoning, answer = self.got._parse_answer(raw)
        self.assertIn("Step 1",  reasoning)
        self.assertEqual(answer, "24")

    def test_parse_answer_case_insensitive(self):
        raw = "REASONING: Do stuff.\nANSWER: 42"
        _, answer = self.got._parse_answer(raw)
        self.assertEqual(answer, "42")

    def test_parse_answer_multiline_reasoning(self):
        raw = "Reasoning: Line 1.\nLine 2.\nLine 3.\nAnswer: done"
        reasoning, answer = self.got._parse_answer(raw)
        self.assertIn("Line 2", reasoning)
        self.assertEqual(answer, "done")

    def test_parse_answer_fallback_last_line(self):
        """When structured format is absent, fall back to the last line."""
        raw = "Some text\nwithout format\nfinal_answer_here"
        _, answer = self.got._parse_answer(raw)
        self.assertEqual(answer, "final_answer_here")

    def test_parse_answer_empty_string(self):
        _, answer = self.got._parse_answer("")
        self.assertEqual(answer, "")

    # ── _parse_score ──────────────────────────────────────────────────────

    def test_parse_score_standard(self):
        self.assertAlmostEqual(self.got._parse_score("0.85"), 0.85)

    def test_parse_score_with_label(self):
        self.assertAlmostEqual(self.got._parse_score("Score: 0.6"), 0.6)

    def test_parse_score_integer_one(self):
        self.assertAlmostEqual(self.got._parse_score("1"), 1.0)

    def test_parse_score_zero(self):
        self.assertAlmostEqual(self.got._parse_score("0.0"), 0.0)

    def test_parse_score_clamp_above_one(self):
        """Values > 1 that are NOT on a 0-10 scale must be clamped."""
        # 1.5 → treated as already decimal → clamped to 1.0
        self.assertAlmostEqual(self.got._parse_score("1.5"), 1.0)

    def test_parse_score_10_scale_corrected(self):
        """A value like 8 is interpreted as 8/10 = 0.8."""
        result = self.got._parse_score("8")
        self.assertAlmostEqual(result, 0.8)

    def test_parse_score_no_number_defaults_zero(self):
        self.assertAlmostEqual(self.got._parse_score("great job!"), 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Individual thought transformations
# ─────────────────────────────────────────────────────────────────────────────

class TestGoTTransformations(unittest.TestCase):
    """Unit-test each transformation in isolation."""

    Q = "Use 1, 2, 3, 4 to make 10."

    # ── Generate ─────────────────────────────────────────────────────────

    def test_generate_creates_correct_count(self):
        resps = [_generate_response(str(i)) for i in range(4)]
        got   = GoT(SequentialMockLLM(resps), num_branches=4, keep_best=1, refine_rounds=0)
        grs   = GraphReasoningState()
        thoughts = got._transform_generate(self.Q, grs)
        self.assertEqual(len(thoughts), 4)

    def test_generate_adds_to_grs(self):
        resps = [_generate_response() for _ in range(3)]
        got   = GoT(SequentialMockLLM(resps), num_branches=3, keep_best=1, refine_rounds=0)
        grs   = GraphReasoningState()
        got._transform_generate(self.Q, grs)
        self.assertEqual(len(grs.all_thoughts()), 3)

    def test_generate_records_parent_ids(self):
        resps  = [_generate_response() for _ in range(2)]
        got    = GoT(SequentialMockLLM(resps), num_branches=2, keep_best=1, refine_rounds=0)
        grs    = GraphReasoningState()
        parent = Thought()
        grs.add(parent)
        thoughts = got._transform_generate(self.Q, grs, parent_ids=[parent.id])
        for t in thoughts:
            self.assertIn(parent.id, t.parent_ids)

    def test_generate_metadata_phase(self):
        resps = [_generate_response()]
        got   = GoT(SequentialMockLLM(resps), num_branches=1, keep_best=1, refine_rounds=0)
        grs   = GraphReasoningState()
        ts    = got._transform_generate(self.Q, grs)
        self.assertEqual(ts[0].metadata["phase"], "generate")

    # ── Score ─────────────────────────────────────────────────────────────

    def test_score_updates_grs_scores(self):
        thoughts = [Thought(), Thought()]
        grs      = GraphReasoningState()
        for t in thoughts:
            grs.add(t)
        resps = ["0.9", "0.4"]
        got   = GoT(SequentialMockLLM(resps), num_branches=2, keep_best=1, refine_rounds=0)
        got._transform_score(self.Q, thoughts, grs)
        self.assertAlmostEqual(grs.get(thoughts[0].id).score, 0.9)
        self.assertAlmostEqual(grs.get(thoughts[1].id).score, 0.4)

    def test_score_marks_status_scored(self):
        t   = Thought()
        grs = GraphReasoningState()
        grs.add(t)
        got = GoT(SequentialMockLLM(["0.7"]), num_branches=1, keep_best=1, refine_rounds=0)
        got._transform_score(self.Q, [t], grs)
        self.assertEqual(t.status, ThoughtStatus.SCORED)

    # ── KeepBest ──────────────────────────────────────────────────────────

    def test_keep_best_returns_correct_count(self):
        grs     = GraphReasoningState()
        scores  = [0.9, 0.5, 0.7, 0.3]
        thoughts = []
        for s in scores:
            t = Thought(score=s)
            t.status = ThoughtStatus.SCORED
            grs.add(t)
            thoughts.append(t)
        got  = GoT(SequentialMockLLM([]), num_branches=4, keep_best=2, refine_rounds=0)
        kept = got._transform_keep_best(thoughts, grs)
        self.assertEqual(len(kept), 2)

    def test_keep_best_selects_highest_scores(self):
        grs      = GraphReasoningState()
        thoughts = []
        for s in [0.1, 0.9, 0.5]:
            t = Thought(score=s); grs.add(t); thoughts.append(t)
        got  = GoT(SequentialMockLLM([]), num_branches=3, keep_best=1, refine_rounds=0)
        kept = got._transform_keep_best(thoughts, grs)
        self.assertAlmostEqual(kept[0].score, 0.9)

    def test_keep_best_marks_pruned(self):
        grs      = GraphReasoningState()
        thoughts = []
        for s in [0.9, 0.1]:
            t = Thought(score=s); grs.add(t); thoughts.append(t)
        got = GoT(SequentialMockLLM([]), num_branches=2, keep_best=1, refine_rounds=0)
        got._transform_keep_best(thoughts, grs)
        pruned = [t for t in grs.all_thoughts() if t.status == ThoughtStatus.PRUNED]
        self.assertEqual(len(pruned), 1)
        self.assertAlmostEqual(pruned[0].score, 0.1)

    # ── Aggregate ─────────────────────────────────────────────────────────

    def test_aggregate_creates_one_new_thought(self):
        parents = [Thought(content=f"p{i}") for i in range(3)]
        grs     = GraphReasoningState()
        for p in parents:
            grs.add(p)
        resps = [_generate_response("merged")]
        got   = GoT(SequentialMockLLM(resps), num_branches=3, keep_best=3, refine_rounds=0)
        agg   = got._transform_aggregate(self.Q, parents, grs)
        # original 3 + new aggregate = 4
        self.assertEqual(len(grs.all_thoughts()), 4)
        self.assertEqual(agg.metadata["phase"], "aggregate")

    def test_aggregate_parent_ids_correct(self):
        parents = [Thought() for _ in range(2)]
        grs     = GraphReasoningState()
        for p in parents:
            grs.add(p)
        got = GoT(SequentialMockLLM([_generate_response()]), num_branches=2, keep_best=2, refine_rounds=0)
        agg = got._transform_aggregate(self.Q, parents, grs)
        self.assertCountEqual(agg.parent_ids, [p.id for p in parents])

    def test_aggregate_marks_parents_aggregated(self):
        parents = [Thought() for _ in range(2)]
        grs     = GraphReasoningState()
        for p in parents:
            grs.add(p)
        got = GoT(SequentialMockLLM([_generate_response()]), num_branches=2, keep_best=2, refine_rounds=0)
        got._transform_aggregate(self.Q, parents, grs)
        for p in parents:
            self.assertEqual(grs.get(p.id).status, ThoughtStatus.AGGREGATED)

    # ── Refine ────────────────────────────────────────────────────────────

    def test_refine_creates_new_thought(self):
        parent = Thought(content="original")
        grs    = GraphReasoningState()
        grs.add(parent)
        got     = GoT(SequentialMockLLM([_generate_response("improved")]), num_branches=1, keep_best=1, refine_rounds=1)
        refined = got._transform_refine(self.Q, parent, grs)
        self.assertEqual(len(grs.all_thoughts()), 2)
        self.assertIn("improved", refined.content)

    def test_refine_parent_id_set(self):
        parent = Thought()
        grs    = GraphReasoningState()
        grs.add(parent)
        got     = GoT(SequentialMockLLM([_generate_response()]), num_branches=1, keep_best=1, refine_rounds=1)
        refined = got._transform_refine(self.Q, parent, grs)
        self.assertIn(parent.id, refined.parent_ids)

    def test_refine_marks_original_refined(self):
        parent = Thought()
        grs    = GraphReasoningState()
        grs.add(parent)
        got = GoT(SequentialMockLLM([_generate_response()]), num_branches=1, keep_best=1, refine_rounds=1)
        got._transform_refine(self.Q, parent, grs)
        self.assertEqual(grs.get(parent.id).status, ThoughtStatus.REFINED)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Full pipeline – run()
# ─────────────────────────────────────────────────────────────────────────────

class TestGoTFullPipeline(unittest.TestCase):
    """End-to-end run() tests with various GoT configurations."""

    Q = "Use 2, 3, 4, 8 to make 24."

    def _run(self, *, branches, kept, refine, **resp_kwargs):
        """Helper: build mock responses, instantiate GoT, call run()."""
        resps = _build_responses(
            branches=branches, kept=kept, refine_rounds=refine, **resp_kwargs
        )
        got = GoT(
            SequentialMockLLM(resps),
            num_branches=branches,
            keep_best=kept,
            refine_rounds=refine,
        )
        return got.run(self.Q)

    # ── correct LLM call counts ───────────────────────────────────────────

    def test_call_count_default_config(self):
        """branches=3, kept=2, refine=1 → 3+3+1+1 = 8 calls."""
        r = self._run(branches=3, kept=2, refine=1)
        self.assertEqual(r.num_llm_calls, _expected_calls(3, 2, 1))

    def test_call_count_no_refine(self):
        """branches=4, kept=2, refine=0 → 4+4+1 = 9 calls."""
        r = self._run(branches=4, kept=2, refine=0)
        self.assertEqual(r.num_llm_calls, _expected_calls(4, 2, 0))

    def test_call_count_single_branch(self):
        """branches=1, kept=1, refine=0 → 1+1 = 2 calls (no aggregate)."""
        r = self._run(branches=1, kept=1, refine=0)
        self.assertEqual(r.num_llm_calls, _expected_calls(1, 1, 0))

    def test_call_count_multiple_refine_rounds(self):
        """branches=2, kept=1, refine=3 → 2+2+3 = 7 calls (no aggregate because kept=1)."""
        r = self._run(branches=2, kept=1, refine=3)
        self.assertEqual(r.num_llm_calls, _expected_calls(2, 1, 3))

    # ── token accounting ──────────────────────────────────────────────────

    def test_total_tokens(self):
        """Each mock LLM call costs 10 in + 10 out = 20 tokens."""
        r = self._run(branches=3, kept=2, refine=1)
        calls = _expected_calls(3, 2, 1)
        self.assertEqual(r.total_input_tokens,  calls * 10)
        self.assertEqual(r.total_output_tokens, calls * 10)
        self.assertEqual(r.total_tokens,         calls * 20)

    # ── final answer extraction ───────────────────────────────────────────

    def test_final_answer_from_refine(self):
        r = self._run(branches=3, kept=2, refine=1, refine_answer="MY_FINAL_ANSWER")
        self.assertEqual(r.final_answer, "MY_FINAL_ANSWER")

    def test_final_answer_from_aggregate_when_no_refine(self):
        r = self._run(branches=3, kept=2, refine=0, agg_answer="AGG_ANSWER")
        self.assertEqual(r.final_answer, "AGG_ANSWER")

    def test_final_answer_from_generate_when_single_branch_no_refine(self):
        """branches=1, kept=1, refine=0 → answer comes from the sole generated thought."""
        r = self._run(branches=1, kept=1, refine=0, gen_answer="SINGLE_ANSWER")
        self.assertEqual(r.final_answer, "SINGLE_ANSWER")

    # ── BaselineResponse fields ───────────────────────────────────────────

    def test_baseline_type_is_got(self):
        r = self._run(branches=2, kept=1, refine=0)
        self.assertEqual(r.baseline_type, "GoT")

    def test_reasoning_trace_non_empty(self):
        r = self._run(branches=2, kept=1, refine=0)
        self.assertIsInstance(r.reasoning_trace, str)
        self.assertGreater(len(r.reasoning_trace), 0)

    def test_intermediate_steps_length(self):
        """One entry per thought in the GRS."""
        r  = self._run(branches=3, kept=2, refine=1)
        # 3 generated + 1 aggregate + 1 refined = 5 thoughts
        self.assertEqual(len(r.intermediate_steps), 5)

    def test_intermediate_steps_length_single_branch_no_agg_no_refine(self):
        """branches=1, kept=1, refine=0 → only 1 thought."""
        r = self._run(branches=1, kept=1, refine=0)
        self.assertEqual(len(r.intermediate_steps), 1)

    # ── metadata ──────────────────────────────────────────────────────────

    def test_metadata_got_params(self):
        r = self._run(branches=4, kept=2, refine=2)
        params = r.metadata["got_params"]
        self.assertEqual(params["num_branches"],  4)
        self.assertEqual(params["keep_best"],     2)
        self.assertEqual(params["refine_rounds"], 2)

    def test_metadata_total_thoughts(self):
        r = self._run(branches=3, kept=2, refine=1)
        # 3 generated + 1 aggregate + 1 refine = 5
        self.assertEqual(r.metadata["total_thoughts"], 5)

    def test_metadata_graph_snapshot_keys(self):
        r = self._run(branches=2, kept=1, refine=0)
        snap = r.metadata["graph_snapshot"]
        for tid in snap:
            entry = snap[tid]
            self.assertIn("content",    entry)
            self.assertIn("score",      entry)
            self.assertIn("status",     entry)
            self.assertIn("parent_ids", entry)
            self.assertIn("volume",     entry)

    def test_metadata_final_thought_volume_positive(self):
        """Final thought in default pipeline has ancestors → volume > 0."""
        r = self._run(branches=3, kept=2, refine=1)
        self.assertGreater(r.metadata["final_thought_volume"], 0)

    def test_metadata_final_thought_id_in_graph(self):
        r    = self._run(branches=2, kept=1, refine=0)
        fid  = r.metadata["final_thought_id"]
        snap = r.metadata["graph_snapshot"]
        self.assertIn(fid, snap)

    # ── instruction prepend ───────────────────────────────────────────────

    def test_instruction_prepended_to_question(self):
        resps = _build_responses(branches=1, kept=1, refine_rounds=0)
        llm   = SequentialMockLLM(resps)
        got   = GoT(llm, num_branches=1, keep_best=1, refine_rounds=0)
        got.run(self.Q, instruction="Think carefully.")
        # The first prompt sent should contain both instruction and question
        self.assertIn("Think carefully.", llm.prompts_seen[0])
        self.assertIn(self.Q,             llm.prompts_seen[0])

    # ── temperature override ──────────────────────────────────────────────

    def test_temperature_override(self):
        # Passing temperature= should affect the run but NOT permanently mutate
        # the instance's gen_temperature (Issue 4 fix).
        resps = _build_responses(branches=1, kept=1, refine_rounds=0)
        got   = GoT(SequentialMockLLM(resps), num_branches=1, keep_best=1,
                    refine_rounds=0, gen_temperature=0.7)
        got.run(self.Q, temperature=0.99)
        # gen_temperature must be restored to original value after the run
        self.assertAlmostEqual(got.gen_temperature, 0.7)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Counter reset between runs
# ─────────────────────────────────────────────────────────────────────────────

class TestGoTCounterReset(unittest.TestCase):
    """Verify that token / call counters are reset at the start of every run()."""

    Q = "problem statement"

    def test_counters_reset_between_runs(self):
        """Running twice must NOT accumulate tokens across runs."""
        # Two complete independent response sets
        r1 = _build_responses(branches=2, kept=1, refine_rounds=0)
        r2 = _build_responses(branches=2, kept=1, refine_rounds=0)
        got = GoT(
            SequentialMockLLM(r1 + r2),
            num_branches=2, keep_best=1, refine_rounds=0,
        )

        resp1 = got.run(self.Q)
        resp2 = got.run(self.Q)

        calls = _expected_calls(2, 1, 0)
        # Each run should report the same number of calls independently
        self.assertEqual(resp1.num_llm_calls, calls)
        self.assertEqual(resp2.num_llm_calls, calls)

    def test_instance_counters_reset(self):
        """got.num_llm_calls reflects only the most recent run."""
        resps = _build_responses(branches=2, kept=1, refine_rounds=0) * 2
        got   = GoT(
            SequentialMockLLM(resps),
            num_branches=2, keep_best=1, refine_rounds=0,
        )
        got.run(self.Q)
        got.run(self.Q)
        # After the second run the instance counter should equal one run's worth
        self.assertEqual(got.num_llm_calls, _expected_calls(2, 1, 0))


# ─────────────────────────────────────────────────────────────────────────────
# 9. Edge cases and robustness
# ─────────────────────────────────────────────────────────────────────────────

class TestGoTEdgeCases(unittest.TestCase):
    """Boundary conditions and malformed LLM output handling."""

    Q = "What is 1 + 1?"

    def test_single_branch_no_aggregate_called(self):
        """
        When only one thought survives KeepBest the Aggregate step must be
        skipped entirely, saving one LLM call.
        """
        resps = _build_responses(branches=1, kept=1, refine_rounds=0)
        got   = GoT(SequentialMockLLM(resps), num_branches=1, keep_best=1, refine_rounds=0)
        r     = got.run(self.Q)
        # 1 generate + 1 score = 2 calls; no aggregate
        self.assertEqual(r.num_llm_calls, 2)

    def test_zero_refine_rounds(self):
        resps = _build_responses(branches=2, kept=2, refine_rounds=0)
        got   = GoT(SequentialMockLLM(resps), num_branches=2, keep_best=2, refine_rounds=0)
        r     = got.run(self.Q)
        self.assertEqual(r.num_llm_calls, _expected_calls(2, 2, 0))

    def test_malformed_score_falls_back_to_zero(self):
        """An unparseable score response should produce 0.0, not crash."""
        resps = [
            _generate_response(),           # generate
            "This is great!",               # score → no number → 0.0
        ]
        got = GoT(SequentialMockLLM(resps), num_branches=1, keep_best=1, refine_rounds=0)
        r   = got.run(self.Q)
        # Should complete without raising; score of sole thought is 0.0
        snap = r.metadata["graph_snapshot"]
        first_thought = next(iter(snap.values()))
        self.assertAlmostEqual(first_thought["score"], 0.0)

    def test_malformed_answer_falls_back_to_last_line(self):
        """When there is no 'Answer:' tag the last line is used as answer."""
        raw_without_format = "I think the answer is 2"
        resps = [raw_without_format, _score_response()]
        got   = GoT(SequentialMockLLM(resps), num_branches=1, keep_best=1, refine_rounds=0)
        r     = got.run(self.Q)
        self.assertEqual(r.final_answer, "I think the answer is 2")

    def test_high_branches_low_keep(self):
        """branches=5, kept=1 → no aggregate; 1 refine round."""
        resps = _build_responses(branches=5, kept=1, refine_rounds=1)
        got   = GoT(SequentialMockLLM(resps), num_branches=5, keep_best=1, refine_rounds=1)
        r     = got.run(self.Q)
        self.assertEqual(r.num_llm_calls, _expected_calls(5, 1, 1))

    def test_repr_contains_class_and_baseline_name(self):
        got = GoT(SequentialMockLLM([]))
        self.assertIn("GoT",       repr(got))
        self.assertIn("GoT",       repr(got))


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
