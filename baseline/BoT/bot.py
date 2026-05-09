"""
Buffer of Thoughts (BoT) Implementation.

BoT is a thought-augmented reasoning framework with three core components:

    1. Problem Distiller  — extracts key variables, constraints, and a
                            high-level task description from the raw question.

    2. Meta-Buffer        — a lightweight in-memory (or file-backed) library of
                            generalised *thought-templates* organised into six
                            categories: Text Comprehension, Creative Language
                            Generation, Common Sense Reasoning, Mathematical
                            Reasoning, Code Programming, Application Scheduling.

    3. Buffer Manager     — after every solve, distils a new thought-template
                            from the (distilled problem, solution) pair and
                            conditionally inserts it into the meta-buffer if it
                            is sufficiently novel (similarity < threshold δ).

Bugs fixed from original document version
──────────────────────────────────────────
    A. BufferManager: removed @property decorators that shadowed the plain
       instance attributes total_input_tokens / total_output_tokens / num_calls,
       causing infinite RecursionError on every counter access.
    B. distil_and_update: corrected `_DISTILLATION_SYSTEM` → `DISTILLATION_SYSTEM`.
    C. distil_and_update: corrected `self._parse_template` → `self.parse_template`.
    D. BoT.run() / __repr__: corrected `self._buffer_manager` → `self.buffer_manager`
       and `self._meta_buffer` → `self.meta_buffer`.
    E. BoT: removed the @property stubs for meta_buffer / buffer_manager that
       returned non-existent private names (_meta_buffer / _buffer_manager).
    F. MetaBuffer.load(): now handles empty JSON files gracefully instead of
       raising JSONDecodeError.

Reference:
    Yang et al., "Buffer of Thoughts: Thought-Augmented Reasoning with Large
    Language Models", NeurIPS 2024.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from baseline.basebaseline import BaseBaseline, BaselineResponse
from models.base import BaseLLM

# Lazy-loaded singleton — model is downloaded once and reused across all MetaBuffer instances.
_EMBED_MODEL: Optional[SentenceTransformer] = None

def _get_embed_model() -> SentenceTransformer:
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        _EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBED_MODEL

def _embed(text: str) -> np.ndarray:
    return _get_embed_model().encode(text, normalize_embeddings=True)

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    # Vectors are already L2-normalised, so dot product == cosine similarity.
    return float(np.dot(a, b))


THOUGHT_CATEGORIES = [
    "Text Comprehension",
    "Creative Language Generation",
    "Common Sense Reasoning",
    "Mathematical Reasoning",
    "Code Programming",
    "Application Scheduling",
]

# Seed templates from BoT paper Appendix B.1 — used when meta-buffer is empty.
SEED_TEMPLATES: list = [
    {
        "index": 0,
        "category": "Text Comprehension",
        "description": (
            "Parse a structured data table with named attributes, integrate "
            "natural-language updates, and answer attribute-based lookup or "
            "comparison questions using logical reasoning."
        ),
        "template": (
            "Step 1: Parse the initial table, extracting header names and each "
            "entry's attributes into a structured format (e.g., list of dicts).\n"
            "Step 2: Read and integrate any additional natural-language information "
            "that updates or adds rows, keeping the data consistent.\n"
            "Step 3: Identify the target attribute (e.g., oldest, heaviest) and its "
            "corresponding column.\n"
            "Step 4: Apply logical comparison across all entries to find the answer "
            "(e.g., maximum age for the oldest entry).\n"
            "Step 5: Select the option that matches the result of the comparison."
        ),
    },
    {
        "index": 1,
        "category": "Creative Language Generation",
        "description": (
            "Write a 14-line English sonnet following the ABAB CDCD EFEF GG rhyme "
            "scheme in iambic pentameter, incorporating given words verbatim."
        ),
        "template": (
            "Step 1: Identify the words that must appear verbatim in the sonnet.\n"
            "Step 2: Map the rhyme scheme ABAB CDCD EFEF GG and brainstorm end-words "
            "for each position.\n"
            "Step 3: Develop a theme that naturally incorporates the required words.\n"
            "Step 4: Draft the first quatrain (lines 1-4) following ABAB, placing at "
            "least one required word.\n"
            "Step 5: Continue with the second quatrain CDCD (lines 5-8) and third "
            "quatrain EFEF (lines 9-12), weaving in remaining required words.\n"
            "Step 6: Write the closing couplet GG (lines 13-14).\n"
            "Step 7: Review for coherence, metre, rhyme adherence, and verbatim "
            "inclusion of all required words."
        ),
    },
    {
        "index": 2,
        "category": "Common Sense Reasoning",
        "description": (
            "Infer or calculate a calendar date by reasoning about month lengths, "
            "leap years, and arithmetic offsets from a given reference date or event."
        ),
        "template": (
            "Step 1: Identify the reference date's year, month, and day.\n"
            "Step 2: Apply the required date offset (days/months/years) using standard "
            "calendar rules.\n"
            "Step 3: Handle month-end rollovers — when a day exceeds the month's "
            "length, carry over to the next month.\n"
            "Step 4: Handle year-end rollover — December 31 + 1 day = January 1 of "
            "the next year.\n"
            "Step 5: Account for leap years when February is involved "
            "(leap year: divisible by 4, except centuries unless divisible by 400).\n"
            "Step 6: Return the final date in the requested format."
        ),
    },
    {
        "index": 3,
        "category": "Mathematical Reasoning",
        "description": (
            "Solve a quadratic equation ax²+bx+c=0 by computing the discriminant "
            "and applying the quadratic formula, handling real and complex roots."
        ),
        "template": (
            "Step 1: Identify coefficients a, b, c from the equation ax²+bx+c=0.\n"
            "Step 2: Compute the discriminant D = b² - 4ac.\n"
            "Step 3: Determine the nature of the roots:\n"
            "  - D > 0: two distinct real roots.\n"
            "  - D = 0: one repeated real root.\n"
            "  - D < 0: two complex conjugate roots.\n"
            "Step 4: Compute the roots:\n"
            "  - D >= 0: x = (-b ± sqrt(D)) / (2a).\n"
            "  - D < 0:  x = -b/(2a) ± sqrt(-D)/(2a) * i.\n"
            "Step 5: Verify by substituting roots back into the original equation."
        ),
    },
    {
        "index": 4,
        "category": "Code Programming",
        "description": (
            "Find a sequence of arithmetic operations on a given list of numbers "
            "that evaluates to a target value, using exhaustive permutation search."
        ),
        "template": (
            "from itertools import permutations, product\n\n"
            "def perform_operation(a, b, op):\n"
            "    if op == '+': return a + b\n"
            "    if op == '-': return a - b\n"
            "    if op == '*': return a * b\n"
            "    if op == '/' and b != 0: return a / b\n"
            "    raise ValueError\n\n"
            "def evaluate_sequence(seq, ops):\n"
            "    result = seq[0]\n"
            "    for i, op in enumerate(ops):\n"
            "        result = perform_operation(result, seq[i + 1], op)\n"
            "    return result\n\n"
            "def find_solution(input_elements, target_result):\n"
            "    ops = ['+', '-', '*', '/']\n"
            "    for seq in permutations(input_elements):\n"
            "        for op_combo in product(ops, repeat=len(seq)-1):\n"
            "            try:\n"
            "                if abs(evaluate_sequence(seq, op_combo) - target_result) < 1e-9:\n"
            "                    return seq, op_combo\n"
            "            except (ValueError, ZeroDivisionError):\n"
            "                continue\n"
            "    return None"
        ),
    },
    {
        "index": 5,
        "category": "Application Scheduling",
        "description": (
            "Apply a sequence of chess moves given in SAN notation to a board and "
            "find the single legal move that delivers checkmate."
        ),
        "template": (
            "import chess\n\n"
            "def find_checkmate_move(moves_san):\n"
            "    board = chess.Board()\n"
            "    for move_san in moves_san:\n"
            "        parts = move_san.split('. ')\n"
            "        if len(parts) > 1:\n"
            "            move_san = parts[1].strip()\n"
            "        if move_san:\n"
            "            board.push(board.parse_san(move_san))\n"
            "    for move in board.legal_moves:\n"
            "        board_copy = board.copy()\n"
            "        board_copy.push(move)\n"
            "        if board_copy.is_checkmate():\n"
            "            return board.san(move)\n"
            "    return None"
        ),
    },
    {
        "index": 6,
        "category": "Code Programming",
        "description": (
            "Predict the output of a Python function by tracing its execution "
            "step by step given specific input values, tracking variable states "
            "through each line of code."
        ),
        "template": (
            "Step 1: Read the function signature and identify all parameters. "
            "Bind each parameter to the provided input value.\n"
            "Step 2: Execute the function body line by line, maintaining a variable "
            "state table that records the current value of every variable after "
            "each assignment or mutation.\n"
            "Step 3: For control flow (if/else, for, while), evaluate the condition "
            "with the current variable values and follow only the branch that is taken.\n"
            "Step 4: For nested function calls or built-in operations, resolve them "
            "using their standard Python semantics (e.g., list.append mutates in place "
            "and returns None; sorted() returns a new list).\n"
            "Step 5: When a return statement is reached, record the returned value as "
            "the function output.\n"
            "Step 6: State the final output clearly, paying attention to the exact "
            "type and structure (int, str, list, tuple, None, etc.)."
        ),
    },
]


@dataclass
class ThoughtTemplate:
    index: int
    category: str
    description: str
    template: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ThoughtTemplate":
        return cls(**d)



class MetaBuffer:
    def __init__(
        self,
        buffer_path: Optional[str] = None,
        init_templates: Optional[List[ThoughtTemplate]] = None,
    ) -> None:
        self.buffer: List[ThoughtTemplate] = []
        self._embeddings: List[np.ndarray] = []  # parallel to self.buffer
        self.buffer_path = buffer_path
        if buffer_path and os.path.isfile(buffer_path):
            self.load(buffer_path)
        elif init_templates:
            self.buffer = list(init_templates)
            self._embeddings = [_embed(t.description) for t in self.buffer]

    def load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as fh:
            content = fh.read().strip()
        if not content:
            return  # FIX-F: empty file handled gracefully
        raw: List[Dict[str, Any]] = json.loads(content)
        self.buffer = [ThoughtTemplate.from_dict(d) for d in raw]
        self._embeddings = [_embed(t.description) for t in self.buffer]

    def save(self) -> None:
        if self.buffer_path:
            with open(self.buffer_path, "w", encoding="utf-8") as fh:
                json.dump([t.to_dict() for t in self.buffer], fh, indent=2, ensure_ascii=False)

    @property
    def size(self) -> int:
        return len(self.buffer)

    def retrieve(self, distilled_info: str, threshold: float = 0.6) -> Optional[ThoughtTemplate]:
        if not self.buffer:
            return None
        query_emb = _embed(distilled_info)
        scores = [_cosine_sim(query_emb, e) for e in self._embeddings]
        best_idx = int(np.argmax(scores))
        return self.buffer[best_idx] if scores[best_idx] >= threshold else None

    def add(self, template: ThoughtTemplate, threshold: float = 0.6) -> bool:
        new_emb = _embed(template.description)
        if self.buffer:
            max_sim = max(_cosine_sim(new_emb, e) for e in self._embeddings)
            if max_sim >= threshold:
                return False
        template.index = len(self.buffer)
        self.buffer.append(template)
        self._embeddings.append(new_emb)
        self.save()
        return True

    def all_templates(self) -> List[ThoughtTemplate]:
        return list(self.buffer)


PROBLEM_DISTILLER_SYSTEM = """\
You are a highly professional expert in information distillation.
Your goal is to extract the essential information needed to solve a problem \
and translate it into a structured, high-level description.

Given the user's question, produce a distilled task description that includes:

1. Key information
   - Identify all essential variables, parameters, and given values.
   - Clarify what is known and what must be found.

2. Restrictions / Constraints
   - State the objective clearly.
   - List all explicit and implicit constraints.

3. Distilled task
   - Generalise the specific problem into a *meta-problem* description.
   - Keep this section concise (≤ 5 sentences).

Return your response in the following format:
### Key Information
<bullet list>

### Constraints
<bullet list>

### Distilled Task
<short paragraph>
"""

DISTILLATION_SYSTEM = """\
You are an expert in extracting high-level reasoning paradigms from problem solutions.
Given a problem description and its solution, extract a generalised thought-template \
that can be reused for similar problems.

Return your response in the following JSON format (and ONLY this JSON):
{
  "category": "<one of the six categories>",
  "description": "<one sentence description for retrieval>",
  "template": "<the full thought-template text>"
}
"""

INSTANTIATION_SYSTEM = """\
You are a Meta Reasoner with deep expertise across Computer Science, Mathematics, \
Physics, Literature, History, Chemistry, Logic, and Language.

You will be given:
  (a) a distilled problem description, and
  (b) a thought-template that provides a high-level reasoning structure.

Format your final answer as:
### Reasoning
<step-by-step reasoning following the template>

### Answer
<concise final answer>
"""

NEW_TASK_SYSTEM = """\
You are a Meta Reasoner with deep expertise across Computer Science, Mathematics, \
Physics, Literature, History, Chemistry, Logic, and Language.

You will be given a distilled problem description for a task that has no prior \
reasoning template.

Format your final answer as:
### Reasoning Structure Chosen
<one of the three above>

### Reasoning
<step-by-step reasoning>

### Answer
<concise final answer>
"""


class BufferManager:
    def __init__(
        self,
        meta_buffer: MetaBuffer,
        llm: BaseLLM,
        similarity_threshold: float = 0.6,
        distill_temperature: float = 0.2,
    ) -> None:
        self.meta_buffer = meta_buffer
        self.llm = llm
        self.threshold = similarity_threshold
        self.distill_temperature = distill_temperature
        # Plain instance attributes — NOT @property (FIX-A: avoids RecursionError)
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.num_calls: int = 0

    def reset_counters(self) -> None:
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.num_calls = 0

    def distil_and_update(self, distilled_info: str, solution: str) -> Optional[ThoughtTemplate]:
        prompt = (
            f"{DISTILLATION_SYSTEM}\n\n"   # FIX-B: was _DISTILLATION_SYSTEM
            f"[Problem Description]\n{distilled_info}\n\n"
            f"[Solution Steps or Code]\n{solution}"
        )
        response = self.llm.generate(prompt, temperature=self.distill_temperature)
        self.total_input_tokens += response.input_tokens
        self.total_output_tokens += response.output_tokens
        self.num_calls += 1
        template = self.parse_template(response.content.strip())  # FIX-C
        if template is None:
            return None
        return template if self.meta_buffer.add(template, threshold=self.threshold) else None

    @staticmethod
    def parse_template(raw: str) -> Optional[ThoughtTemplate]:
        raw = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r"\{[\s\S]+\}", raw)
            if not m:
                return None
            try:
                data = json.loads(m.group())
            except json.JSONDecodeError:
                return None
        description = data.get("description", "")
        template_text = data.get("template", "")
        if not description or not template_text:
            return None
        return ThoughtTemplate(
            index=-1,
            category=data.get("category", "Common Sense Reasoning"),
            description=description,
            template=template_text,
        )


class BoT(BaseBaseline):
    """Buffer of Thoughts (BoT) prompting baseline."""

    def __init__(
        self,
        llm: BaseLLM,
        similarity_threshold: float = 0.6,
        buffer_path: Optional[str] = None,
        distill_temperature: float = 0.2,
        instantiation_temperature: float = 0.1,
        update_buffer: bool = True,
        init_templates: Optional[List[ThoughtTemplate]] = None,
    ) -> None:
        super().__init__(llm, baseline_name="BoT")
        self.similarity_threshold = similarity_threshold
        self.distill_temperature = distill_temperature
        self.instantiation_temperature = instantiation_temperature
        self.update_buffer = update_buffer
        # Fall back to paper's seed templates when the buffer would otherwise be empty.
        effective_init = init_templates
        if effective_init is None and not (buffer_path and os.path.isfile(buffer_path)):
            effective_init = [ThoughtTemplate.from_dict(d) for d in SEED_TEMPLATES]
        # Public attributes — no @property aliases needed (FIX-E)
        self.meta_buffer = MetaBuffer(buffer_path=buffer_path, init_templates=effective_init)
        self.buffer_manager = BufferManager(
            meta_buffer=self.meta_buffer,
            llm=llm,
            similarity_threshold=similarity_threshold,
            distill_temperature=distill_temperature,
        )

    # ── Stage 1 ───────────────────────────────────────────────────────────────

    def distil_problem(self, question: str, temperature: float = 0.0) -> str:
        prompt = f"{PROBLEM_DISTILLER_SYSTEM}\n\nQuestion:\n{question}"
        return self.call_llm(prompt, temperature=temperature).content.strip()

    # ── Stage 2 ───────────────────────────────────────────────────────────────

    def retrieve_template(self, distilled_info: str) -> Optional[ThoughtTemplate]:
        return self.meta_buffer.retrieve(distilled_info, threshold=self.similarity_threshold)

    # ── Stage 3 ───────────────────────────────────────────────────────────────

    def instantiate_with_template(
        self, distilled_info: str, template: ThoughtTemplate, system_prompt: Optional[str] = None
    ) -> str:
        role_line = f"[Role]\n{system_prompt}\n\n" if system_prompt else ""
        prompt = (
            f"{INSTANTIATION_SYSTEM}\n\n"
            f"{role_line}"
            f"[Distilled Problem]\n{distilled_info}\n\n"
            f"[Thought Template — {template.category}]\n{template.template}"
        )
        return self.call_llm(prompt, temperature=self.instantiation_temperature).content.strip()

    def instantiate_new_task(self, distilled_info: str, system_prompt: Optional[str] = None) -> str:
        role_line = f"[Role]\n{system_prompt}\n\n" if system_prompt else ""
        prompt = f"{NEW_TASK_SYSTEM}\n\n{role_line}[Distilled Problem]\n{distilled_info}"
        return self.call_llm(prompt, temperature=self.instantiation_temperature).content.strip()

    # ── Answer extraction ─────────────────────────────────────────────────────

    @staticmethod
    def extract_answer(raw_response: str) -> str:
        m = re.compile(r"###\s*Answer\s*\n(.*?)(?=###|\Z)", re.DOTALL | re.IGNORECASE).search(raw_response)
        if m:
            return m.group(1).strip()
        lines = [ln.strip() for ln in raw_response.split("\n") if ln.strip()]
        return lines[-1] if lines else raw_response.strip()

    # ── run() ─────────────────────────────────────────────────────────────────

    def run(
        self,
        question: str,
        system_prompt: Optional[str] = None,
        instruction: Optional[str] = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> BaselineResponse:
        self.reset_counters()
        self.buffer_manager.reset_counters()
        intermediate_steps: List[str] = []

        # Combine instruction with the raw question so the distiller has full context.
        full_question = f"{instruction}\n\n{question}" if instruction else question
        distilled_info = self.distil_problem(full_question, temperature=temperature)
        intermediate_steps.append(f"[Stage 1: Problem Distiller]\n{distilled_info}")

        template = self.retrieve_template(distilled_info)
        is_new_task = template is None
        if template is not None:
            intermediate_steps.append(
                f"[Stage 2: Template Retrieved]\nCategory: {template.category}\n"
                f"Description: {template.description}\nIndex: {template.index}"
            )
        else:
            intermediate_steps.append(
                "[Stage 2: Template Retrieval] No suitable template found — new task."
            )

        raw_solution = (
            self.instantiate_with_template(distilled_info, template, system_prompt=system_prompt)
            if template is not None
            else self.instantiate_new_task(distilled_info, system_prompt=system_prompt)
        )
        intermediate_steps.append(f"[Stage 3: Instantiated Reasoning]\n{raw_solution}")
        final_answer = self.extract_answer(raw_solution)

        new_template: Optional[ThoughtTemplate] = None
        if self.update_buffer:
            new_template = self.buffer_manager.distil_and_update(
                distilled_info=distilled_info, solution=raw_solution
            )
            self.total_input_tokens += self.buffer_manager.total_input_tokens
            self.total_output_tokens += self.buffer_manager.total_output_tokens
            self.num_llm_calls += self.buffer_manager.num_calls
            intermediate_steps.append(
                f"[Stage 4: Buffer Manager] New template added "
                f"(index={new_template.index}, category='{new_template.category}')."
                if new_template is not None
                else "[Stage 4: Buffer Manager] Template not added "
                     "(too similar to an existing one or distillation failed)."
            )

        reasoning_trace = (
            f"[Distilled Problem]\n{distilled_info}\n\n[Retrieved Template]\n"
            + (f"Category: {template.category}\nTemplate:\n{template.template}"
               if template else "(none — new task, used general reasoning structure)")
            + f"\n\n[Instantiated Reasoning]\n{raw_solution}"
        )

        return self.create_response(
            final_answer=final_answer,
            reasoning_trace=reasoning_trace,
            intermediate_steps=intermediate_steps,
            metadata={
                "is_new_task": is_new_task,
                "retrieved_template_index": template.index if template else None,
                "retrieved_template_category": template.category if template else None,
                "new_template_added": new_template is not None,
                "meta_buffer_size": self.meta_buffer.size,
                "similarity_threshold": self.similarity_threshold,
                "instantiation_temperature": self.instantiation_temperature,
                "distill_temperature": self.distill_temperature,
                "update_buffer": self.update_buffer,
            },
        )

    def __repr__(self) -> str:
        return (
            f"BoT(baseline_name='{self.baseline_name}', "
            f"llm={self.llm.__class__.__name__}, "
            f"similarity_threshold={self.similarity_threshold}, "
            f"meta_buffer_size={self.meta_buffer.size}, "
            f"update_buffer={self.update_buffer})"
        )