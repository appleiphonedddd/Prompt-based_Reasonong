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
import math
import os
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from baseline.basebaseline import BaseBaseline, BaselineResponse
from models.base import BaseLLM


THOUGHT_CATEGORIES = [
    "Text Comprehension",
    "Creative Language Generation",
    "Common Sense Reasoning",
    "Mathematical Reasoning",
    "Code Programming",
    "Application Scheduling",
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


def tokenise(text: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for tok in re.findall(r"[a-z]+", text.lower()):
        counts[tok] = counts.get(tok, 0) + 1
    return counts


def cosine_similarity(a: Dict[str, int], b: Dict[str, int]) -> float:
    shared = set(a) & set(b)
    dot = sum(a[k] * b[k] for k in shared)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class MetaBuffer:
    def __init__(
        self,
        buffer_path: Optional[str] = None,
        init_templates: Optional[List[ThoughtTemplate]] = None,
    ) -> None:
        self.buffer: List[ThoughtTemplate] = []
        self.buffer_path = buffer_path
        if buffer_path and os.path.isfile(buffer_path):
            self.load(buffer_path)
        elif init_templates:
            self.buffer = list(init_templates)

    def load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as fh:
            content = fh.read().strip()
        if not content:
            return  # FIX-F: empty file handled gracefully
        raw: List[Dict[str, Any]] = json.loads(content)
        self.buffer = [ThoughtTemplate.from_dict(d) for d in raw]

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
        query_bow = tokenise(distilled_info)
        best_score = -1.0
        best_tmpl: Optional[ThoughtTemplate] = None
        for tmpl in self.buffer:
            score = cosine_similarity(query_bow, tokenise(tmpl.description))
            if score > best_score:
                best_score = score
                best_tmpl = tmpl
        return best_tmpl if best_score >= threshold else None

    def add(self, template: ThoughtTemplate, threshold: float = 0.6) -> bool:
        if not self.buffer:
            template.index = 0
            self.buffer.append(template)
            self.save()
            return True
        new_bow = tokenise(template.description)
        max_sim = max(cosine_similarity(new_bow, tokenise(t.description)) for t in self.buffer)
        if max_sim < threshold:
            template.index = len(self.buffer)
            self.buffer.append(template)
            self.save()
            return True
        return False

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
        # Public attributes — no @property aliases needed (FIX-E)
        self.meta_buffer = MetaBuffer(buffer_path=buffer_path, init_templates=init_templates)
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

    def instantiate_with_template(self, distilled_info: str, template: ThoughtTemplate) -> str:
        prompt = (
            f"{INSTANTIATION_SYSTEM}\n\n"
            f"[Distilled Problem]\n{distilled_info}\n\n"
            f"[Thought Template — {template.category}]\n{template.template}"
        )
        return self.call_llm(prompt, temperature=self.instantiation_temperature).content.strip()

    def instantiate_new_task(self, distilled_info: str) -> str:
        prompt = f"{NEW_TASK_SYSTEM}\n\n[Distilled Problem]\n{distilled_info}"
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

        distilled_info = self.distil_problem(question, temperature=temperature)
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
            self.instantiate_with_template(distilled_info, template)
            if template is not None
            else self.instantiate_new_task(distilled_info)
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