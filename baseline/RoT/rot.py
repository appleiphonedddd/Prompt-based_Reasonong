"""
Reversal of Thought (RoT) Prompting Implementation.

RoT enhances LLM reasoning through a three-stage pipeline:
1. Reverse Reasoning Warm-up: Generate multiple candidate task definitions
   by asking the LLM to reverse-engineer the task from demonstrations.
2. Pairwise Preference Selection: Use the LLM as a judge to compare
   candidates pairwise, then apply transitive closure to strengthen
   preference scores and select the optimal candidate.
3. Instantiation: Apply the selected optimal prompt ("LLM taste") to
   solve the actual input question with structured reasoning.

Reference:
- Yuan, J., Du, D., Zhang, H., Di, Z., & Naseem, U. (2025).
  "Reversal of Thought: Enhancing Large Language Models with
  Preference-Guided Reverse Reasoning Warm-up."
  Proceedings of ACL 2025 (Main), pp. 19442–19459.

Author: Egor Morozov
"""

import re
import math
from typing import Optional, List, Dict, Tuple

from baseline.basebaseline import BaseBaseline, BaselineResponse
from models.base import BaseLLM


# ─────────────────────────────────────────────────────────
# Prompt templates (adapted from the official RoT release)
# ─────────────────────────────────────────────────────────

REVERSAL_OF_THOUGHT_PROMPT = """###Instruction###
You are a highly distinguished expert in mathematics and information reasoning. \
Based on the given example, define the specific task, including the task definition, \
pseudocode, logical pseudocode, case examples, and input-output format.
 1. Understand Task Description:
 Meticulously study demonstrations to deeply understand generic task description.
 2. Plan Generic Pseudocode:
 Provide pseudocode in text form and plan an efficient algorithm to complete the task with your experiences.
 3. Formulate Logical Pseudocode:
 Convert the pseudocode into generic logical algorithm pseudocode using ONLY logical symbols:
 Logical Operators:
 Conjunction: A ∧ B ; Disjunction: A ∨ B
 equivalence: A ≡ B , Negation: ¬A
 Quantifiers:
 Universal quantifier: ∀x ; Existential quantifier: ∃x
 Inequalities:
 Less than: x < y ; Greater than: x > y 
 Less than or equal to: x ≤ y
 Greater than or equal to: x ≥ y
 Equals: x = y ; Not equals: x ≠ y
 Conditional Statements:
 If A then B: A ⊃ B
 If A ∧B then C: (A∧B) ⊃ C
 If A ∨B then C: (A∨B) ⊃ C
 If ∀x(P(x)) then Q: ∀x(P(x)) ⊃ Q
 If ∃x(P(x)) then Q: ∃x(P(x)) ⊃ Q etc.
 Input: [Demonstration] Output: [Output]"""

PAIRWISE_PREFERENCE_PROMPT = (
    "Please choose your more preferred instruction: A or B ?\nInput:\n"
)

INSTANTIATION_PROMPT = """You are an expert-level LLM specialized in structured problem solving across \
domains including mathematics, programming, logic, and reasoning.

Your reasoning and response style should follow your internal preference:
\"{llm_taste}\"

Your task is to generate a complete and precise solution that strictly adheres to the \
provided thought template, while adapting it to the specifics of the task.

Follow these output rules:
- If the solution involves Python code, output exactly one code block, with no explanations, headers, or extra text.
- All Python code must be self-contained, correct, and ready to run.
- For non-code solutions, return a clean, extractable final answer in plain text.
- Do not output multiple code blocks or add commentary.
- Always align input parameters, variable names, and data formats with the user task.

Begin generating your solution only after fully reading both the task and thought template. \
Think step by step, but reflect your reasoning only through the final output.
### Output Format ###
** Thinking **: {{your internal reasoning trace here, if needed}}
** Answer **: {{final answer}}"""


class RoT(BaseBaseline):
    """Reversal of Thought (RoT) prompting baseline.

    Implements the full PGRR pipeline from Algorithm 1 of the paper:
      1. Generate K candidate reverse-reasoned task definitions.
      2. Evaluate all pairwise preferences via LLM-as-judge.
      3. Apply transitive closure to the preference matrix.
      4. Select the optimal candidate (highest combined score).
      5. Instantiate the optimal prompt to solve the target question.

    Attributes:
        warmup: Number of reverse reasoning candidates (K in paper).
        candidate_temperature: Sampling temperature for candidate generation.
        instantiation_temperature: Sampling temperature for final reasoning.
        demos: Few-shot demonstrations for reverse reasoning.

    Example:
        >>> llm = GeminiClient()
        >>> baseline = RoT(llm, warmup=5, demos="Input:1,5,5,5; Output:5*(5-1/5)=24")
        >>> response = baseline.run("2, 5, 8, 11")
        >>> print(response.final_answer)
    """

    def __init__(
        self,
        llm: BaseLLM,
        warmup: int = 5,
        candidate_temperature: float = 0.7,
        instantiation_temperature: float = 0.1,
        demos: Optional[str] = None,
    ):
        """Initialize RoT baseline.

        Args:
            llm: An instance of a BaseLLM subclass.
            warmup: Number of reverse reasoning candidates to generate (K).
            candidate_temperature: Temperature for candidate generation
                                   (higher → more diverse candidates).
            instantiation_temperature: Temperature for final answer generation
                                       (lower → more deterministic reasoning).
            demos: Few-shot demonstration string used for reverse reasoning.
                   Format: "Input:... Output:..."
        """
        super().__init__(llm, baseline_name="RoT")
        self.warmup = warmup
        self.candidate_temperature = candidate_temperature
        self.instantiation_temperature = instantiation_temperature
        self.demos = demos or "Input:1, 5, 5, 5; Output:5× (5 − 1 ÷ 5) = 24"

    # ──────────────────────────────────────────
    # Stage 1: Reverse Reasoning Warm-up
    # ──────────────────────────────────────────

    def _generate_candidates(self) -> List[str]:
        """Generate K candidate task definitions via reverse reasoning.

        Each candidate is produced by prompting the LLM with the
        reversal-of-thought system prompt and the few-shot demos,
        using a higher temperature to encourage diversity.

        Returns:
            List of K candidate reverse-reasoned prompts.
        """
        candidates = []
        prompt = f"{REVERSAL_OF_THOUGHT_PROMPT}\n\n{self.demos}"

        for _ in range(self.warmup):
            response = self.call_llm(prompt, temperature=self.candidate_temperature)
            candidates.append(response.content.strip())

        return candidates

    # ──────────────────────────────────────────
    # Stage 2: Pairwise Preference Evaluation
    # ──────────────────────────────────────────

    def _evaluate_preference(self, candidate_a: str, candidate_b: str) -> float:
        """Evaluate pairwise preference between two candidates.

        Uses the LLM as a judge to decide which candidate instruction
        is preferred ("A" or "B").

        Args:
            candidate_a: First candidate instruction.
            candidate_b: Second candidate instruction.

        Returns:
            Preference score for candidate A (1.0 if A preferred,
            0.0 if B preferred, 0.5 on ambiguity).
        """
        prompt = f"{PAIRWISE_PREFERENCE_PROMPT}A:{candidate_a}\nB:{candidate_b}"
        response = self.call_llm(prompt, temperature=0.0)
        choice = response.content.strip().upper()

        # Parse the LLM's preference decision
        if choice.startswith("A"):
            return 1.0
        elif choice.startswith("B"):
            return 0.0
        else:
            # Fallback: search for A or B anywhere in the response
            a_count = choice.count("A")
            b_count = choice.count("B")
            if a_count > b_count:
                return 1.0
            elif b_count > a_count:
                return 0.0
            return 0.5

    def _build_preference_matrix(
        self, candidates: List[str]
    ) -> Dict[Tuple[int, int], float]:
        """Build the full pairwise preference matrix with transitive closure.

        Steps (following Algorithm 1 from the paper):
        1. Evaluate all (K choose 2) pairwise comparisons.
        2. Apply transitive closure:
           P_pre[(i,k)] = max(P_pre[(i,k)], P_pre[(i,j)] * P_pre[(j,k)])

        Args:
            candidates: List of K candidate strings.

        Returns:
            Preference matrix as a dict mapping (i, j) → score.
        """
        k = len(candidates)
        p_pre: Dict[Tuple[int, int], float] = {}

        # Step 1: Direct pairwise comparisons
        for i in range(k - 1):
            for j in range(i + 1, k):
                score = self._evaluate_preference(candidates[i], candidates[j])
                p_pre[(i, j)] = score
                p_pre[(j, i)] = 1.0 - score

        # Step 2: Transitive closure (Floyd–Warshall–style propagation)
        for i in range(k):
            for j in range(k):
                if i == j:
                    continue
                for m in range(k):
                    if m == i or m == j:
                        continue
                    transitive_score = p_pre.get((i, j), 0.0) * p_pre.get((j, m), 0.0)
                    p_pre[(i, m)] = max(p_pre.get((i, m), 0.0), transitive_score)

        return p_pre

    def _select_optimal(
        self,
        candidates: List[str],
        p_pre: Dict[Tuple[int, int], float],
    ) -> Tuple[int, str]:
        """Select the optimal candidate based on preference scores.

        Computes P_pre_avg[i] = mean preference of candidate i over all others,
        then selects argmax.  (The paper also uses P_res[i] from logprobs;
        since the BaseLLM interface does not expose logprobs, we use
        preference scores alone, which is equivalent to setting P_res uniform.)

        Args:
            candidates: List of K candidate strings.
            p_pre: Preference matrix from _build_preference_matrix.

        Returns:
            Tuple of (optimal_index, optimal_candidate_string).
        """
        k = len(candidates)
        scores = []

        for i in range(k):
            if k > 1:
                avg_pref = sum(
                    p_pre.get((i, j), 0.0) for j in range(k) if j != i
                ) / (k - 1)
            else:
                avg_pref = 1.0
            scores.append(avg_pref)

        optimal_idx = max(range(k), key=lambda i: scores[i])
        return optimal_idx, candidates[optimal_idx]

    # ──────────────────────────────────────────
    # Stage 3: Instantiation
    # ──────────────────────────────────────────

    def _build_instantiation_prompt(self, llm_taste: str) -> str:
        """Build the instantiation system prompt with the optimal LLM taste.

        Args:
            llm_taste: The optimal reverse-reasoned task definition.

        Returns:
            Formatted instantiation prompt string.
        """
        return INSTANTIATION_PROMPT.format(llm_taste=llm_taste)

    def _parse_instantiation_response(self, response_text: str) -> Tuple[str, str]:
        """Parse the instantiation response to extract thinking and answer.

        Args:
            response_text: Raw LLM response following the RoT output format.

        Returns:
            Tuple of (final_answer, thinking_trace).
        """
        # Extract thinking trace
        thinking_pattern = re.compile(
            r"\*\*\s*Thinking\s*\*\*\s*:\s*(.*?)(?=\*\*\s*Answer|\Z)",
            re.DOTALL | re.IGNORECASE,
        )
        thinking_match = thinking_pattern.search(response_text)
        thinking = thinking_match.group(1).strip() if thinking_match else ""

        # Extract final answer
        answer_pattern = re.compile(
            r"\*\*\s*Answer\s*\*\*\s*:\s*(.*?)(?=\n{2,}|\Z)",
            re.DOTALL | re.IGNORECASE,
        )
        answer_match = answer_pattern.search(response_text)
        if answer_match:
            final_answer = answer_match.group(1).strip()
        else:
            # Fallback: use the last non-empty line
            lines = [l.strip() for l in response_text.strip().split("\n") if l.strip()]
            final_answer = lines[-1] if lines else response_text.strip()

        return final_answer, thinking

    # ──────────────────────────────────────────
    # Main entry point
    # ──────────────────────────────────────────

    def run(
        self,
        question: str,
        system_prompt: Optional[str] = None,
        instruction: Optional[str] = None,
        temperature: float = 0.0,
        **kwargs,
    ) -> BaselineResponse:
        """Execute the full RoT pipeline on the given question.

        Pipeline stages:
        1. Generate K candidate task definitions (Reverse Reasoning Warm-up)
        2. Evaluate pairwise preferences and select optimal candidate
        3. Instantiate the optimal prompt to solve the question

        Args:
            question: The input question or problem to solve.
            system_prompt: Optional system-level instruction (unused in RoT,
                          kept for interface compatibility).
            instruction: Optional task-specific instruction (unused in RoT).
            temperature: Overrides instantiation_temperature if provided
                        and non-zero.
            **kwargs: Additional arguments for interface compatibility.

        Returns:
            BaselineResponse containing the answer, full reasoning trace,
            and detailed metadata about the PGRR process.
        """
        self.reset_counters()
        intermediate_steps = []
        inst_temp = temperature if temperature > 0 else self.instantiation_temperature

        # ── Stage 1: Reverse Reasoning Warm-up ──
        candidates = self._generate_candidates()
        intermediate_steps.append(
            f"[Stage 1: Reverse Reasoning] Generated {len(candidates)} candidates"
        )
        for idx, cand in enumerate(candidates):
            intermediate_steps.append(f"[Candidate {idx + 1}]\n{cand}")

        # ── Stage 2: Pairwise Preference Selection ──
        p_pre = self._build_preference_matrix(candidates)
        optimal_idx, llm_taste = self._select_optimal(candidates, p_pre)
        intermediate_steps.append(
            f"[Stage 2: Preference Selection] Selected candidate {optimal_idx + 1}"
        )

        # ── Stage 3: Instantiation ──
        instantiation_sys = self._build_instantiation_prompt(llm_taste)
        question_prompt = f"{instantiation_sys}\n\n{question}"
        response = self.call_llm(question_prompt, temperature=inst_temp)
        final_answer, thinking = self._parse_instantiation_response(
            response.content
        )

        intermediate_steps.append(
            f"[Stage 3: Instantiation]\n{response.content.strip()}"
        )

        # Build reasoning trace
        reasoning_trace = (
            f"[RoT Warm-up] Selected candidate {optimal_idx + 1}/{self.warmup}\n"
            f"[LLM Taste]\n{llm_taste}\n\n"
            f"[Thinking]\n{thinking}"
        )

        # Compute preference score summary
        k = len(candidates)
        pref_scores = {}
        for i in range(k):
            if k > 1:
                avg = sum(p_pre.get((i, j), 0.0) for j in range(k) if j != i) / (k - 1)
            else:
                avg = 1.0
            pref_scores[f"candidate_{i + 1}"] = round(avg, 4)

        return self.create_response(
            final_answer=final_answer,
            reasoning_trace=reasoning_trace,
            intermediate_steps=intermediate_steps,
            metadata={
                "model": response.model_name,
                "warmup": self.warmup,
                "candidate_temperature": self.candidate_temperature,
                "instantiation_temperature": inst_temp,
                "optimal_candidate_index": optimal_idx,
                "preference_scores": pref_scores,
                "demos": self.demos,
            },
        )
