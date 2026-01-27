"""
Zero-Shot Chain-of-Thought (CoT) Prompting Implementation.

Zero-Shot CoT elicits step-by-step reasoning from LLMs by simply appending
"Let's think step by step" to the prompt. This technique significantly
improves performance on reasoning tasks without requiring any examples.

Reference:
- Kojima, T., Gu, S. S., Reid, M., Matsuo, Y., & Iwasawa, Y. (2022). Large language models are zero-shot reasoners. Advances in neural information processing systems,
35, 22199-22213.

Author: Egor Morozov
"""

import re
from typing import Optional

from baseline.basebaseline import BaseBaseline, BaselineResponse
from models.base import BaseLLM


class ZeroShotCoT(BaseBaseline):
    """Zero-Shot Chain-of-Thought prompting baseline.
    
    This baseline uses the magic phrase "Let's think step by step" to
    elicit reasoning from the LLM, then extracts the final answer.
    
    The process involves two LLM calls:
    1. Reasoning: Generate step-by-step reasoning
    2. Extraction: Extract the final answer from the reasoning
    
    Example:
        >>> llm = GPTClient()
        >>> baseline = ZeroShotCoT(llm)
        >>> response = baseline.run("If John has 5 apples and gives 2 to Mary, how many does he have?")
        >>> print(response.reasoning_trace)
        "Let's think step by step. John starts with 5 apples..."
        >>> print(response.final_answer)
        "3"
    """

    # The magic trigger phrase from Kojima et al. (2022)
    COT_TRIGGER = "Let's think step by step."

    # Default prompt for extracting the final answer
    DEFAULT_EXTRACTION_PROMPT = "Therefore, the answer is"

    def __init__(
        self,
        llm: BaseLLM,
        cot_trigger: Optional[str] = None,
        extraction_prompt: Optional[str] = None,
    ):
        """Initialize Zero-Shot CoT baseline.
        
        Args:
            llm: An instance of a BaseLLM subclass.
            cot_trigger: Custom trigger phrase for eliciting reasoning.
                        Defaults to "Let's think step by step."
            extraction_prompt: Custom prompt for answer extraction.
                              Defaults to "Therefore, the answer is"
        """
        super().__init__(llm, baseline_name="ZeroShotCoT")
        self.cot_trigger = cot_trigger or self.COT_TRIGGER
        self.extraction_prompt = extraction_prompt or self.DEFAULT_EXTRACTION_PROMPT

    def build_reasoning_prompt(
        self,
        question: str,
        system_prompt: Optional[str] = None,
        instruction: Optional[str] = None,
    ) -> str:
        """Build the prompt for eliciting step-by-step reasoning.
        
        Args:
            question: The input question to answer.
            system_prompt: Optional system-level instruction.
            instruction: Optional task-specific instruction.
            
        Returns:
            The formatted reasoning prompt.
        """
        parts = []

        if system_prompt:
            parts.append(system_prompt)

        if instruction:
            parts.append(instruction)

        parts.append(f"Question: {question}")
        parts.append(f"Answer: {self.cot_trigger}")

        return "\n\n".join(parts)

    def build_extraction_prompt(
        self,
        question: str,
        reasoning: str,
    ) -> str:
        """Build the prompt for extracting the final answer.
        
        Args:
            question: The original question.
            reasoning: The generated reasoning trace.
            
        Returns:
            The formatted extraction prompt.
        """
        return (
            f"Question: {question}\n\n"
            f"Answer: {self.cot_trigger} {reasoning}\n\n"
            f"{self.extraction_prompt}"
        )

    def extract_answer_simple(self, extraction_response: str) -> str:
        """Simple extraction: take the first line/sentence."""
        answer = extraction_response.strip()
        
        if "\n" in answer:
            answer = answer.split("\n")[0].strip()
        
        pattern = r"(?:therefore,?)?\s*(?:the)?\s*answer\s*(?:is)?\s*[:=]?\s*"
        match = re.match(pattern, answer, re.IGNORECASE)
        if match:

            answer = answer[match.end():].strip()
        
        prefixes_to_remove = [":", "is", "is:", "="]
        for prefix in prefixes_to_remove:
            if answer.lower().startswith(prefix):
                answer = answer[len(prefix):].strip()

        answer = answer.rstrip(".")
        
        return answer

    def run(
        self,
        question: str,
        system_prompt: Optional[str] = None,
        instruction: Optional[str] = None,
        temperature: float = 0.0,
        extract_answer: bool = True,
        **kwargs,
    ) -> BaselineResponse:
        """Execute Zero-Shot CoT prompting on the given question.
        
        This method performs two-stage prompting:
        1. Generate reasoning with "Let's think step by step"
        2. Extract final answer from the reasoning (optional)
        
        Args:
            question: The input question to answer.
            system_prompt: Optional system-level instruction.
            instruction: Optional task-specific instruction.
            temperature: Sampling temperature for generation.
            extract_answer: If True, perform second LLM call to extract
                          the final answer. If False, return full reasoning
                          as the answer.
            **kwargs: Additional arguments (unused, for interface compatibility).
            
        Returns:
            BaselineResponse containing the answer, reasoning trace, and metrics.
        """
        self.reset_counters()
        intermediate_steps = []

        # Stage 1: Generate reasoning
        reasoning_prompt = self.build_reasoning_prompt(
            question, system_prompt, instruction
        )
        reasoning_response = self.call_llm(reasoning_prompt, temperature=temperature)
        reasoning_trace = reasoning_response.content.strip()
        
        intermediate_steps.append(f"[Reasoning]\n{reasoning_trace}")

        # Stage 2: Extract final answer (optional)
        if extract_answer:
            extraction_prompt = self.build_extraction_prompt(question, reasoning_trace)
            extraction_response = self.call_llm(extraction_prompt, temperature=0.0)
            final_answer = self.extract_answer_simple(extraction_response.content)
            
            intermediate_steps.append(f"[Extraction]\n{extraction_response.content.strip()}")
        else:
            final_answer = reasoning_trace

        return self.create_response(
            final_answer=final_answer,
            reasoning_trace=f"{self.cot_trigger} {reasoning_trace}",
            intermediate_steps=intermediate_steps,
            metadata={
                "reasoning_prompt": reasoning_prompt,
                "model": reasoning_response.model_name,
                "cot_trigger": self.cot_trigger,
                "extract_answer": extract_answer,
            },
        )


class ZeroShotCoTSinglePass(BaseBaseline):
    """Single-pass variant of Zero-Shot CoT.
    
    This variant uses only ONE LLM call by including the extraction
    prompt directly in the initial prompt. More efficient but may
    be slightly less accurate for complex reasoning.
    
    Example:
        >>> llm = GPTClient()
        >>> baseline = ZeroShotCoTSinglePass(llm)
        >>> response = baseline.run("What is 15% of 80?")
    """

    COT_TRIGGER = "Let's think step by step."

    def __init__(self, llm: BaseLLM):
        """Initialize Single-Pass Zero-Shot CoT baseline.
        
        Args:
            llm: An instance of a BaseLLM subclass.
        """
        super().__init__(llm, baseline_name="ZeroShotCoT-SinglePass")

    def build_prompt(
        self,
        question: str,
        system_prompt: Optional[str] = None,
        instruction: Optional[str] = None,
    ) -> str:
        """Build the single-pass CoT prompt.
        
        Args:
            question: The input question to answer.
            system_prompt: Optional system-level instruction.
            instruction: Optional task-specific instruction.
            
        Returns:
            The formatted prompt.
        """
        parts = []

        if system_prompt:
            parts.append(system_prompt)

        if instruction:
            parts.append(instruction)

        parts.append(f"Question: {question}")
        parts.append(
            f"{self.COT_TRIGGER}\n"
            "After your reasoning, provide your final answer in the format:\n"
            "**Final Answer:** [your answer]"
        )

        return "\n\n".join(parts)

    def parse_response(self, response_text: str) -> tuple[str, str]:
        """Parse the response to extract reasoning and final answer.
        
        Args:
            response_text: The full LLM response.
            
        Returns:
            Tuple of (final_answer, reasoning_trace).
        """
        # Try to find "Final Answer:" pattern
        pattern = r"\*?\*?Final Answer:?\*?\*?\s*(.+?)(?:\n|$)"
        match = re.search(pattern, response_text, re.IGNORECASE)
        
        if match:
            final_answer = match.group(1).strip()
            reasoning_trace = response_text[:match.start()].strip()
        else:
            # Fallback: use last line as answer
            lines = response_text.strip().split("\n")
            final_answer = lines[-1].strip() if lines else response_text
            reasoning_trace = "\n".join(lines[:-1]) if len(lines) > 1 else ""

        return final_answer, reasoning_trace

    def run(
        self,
        question: str,
        system_prompt: Optional[str] = None,
        instruction: Optional[str] = None,
        temperature: float = 0.0,
        **kwargs,
    ) -> BaselineResponse:
        """Execute single-pass Zero-Shot CoT prompting.
        
        Args:
            question: The input question to answer.
            system_prompt: Optional system-level instruction.
            instruction: Optional task-specific instruction.
            temperature: Sampling temperature for generation.
            **kwargs: Additional arguments (unused).
            
        Returns:
            BaselineResponse containing the answer, reasoning, and metrics.
        """
        self.reset_counters()

        prompt = self.build_prompt(question, system_prompt, instruction)
        response = self.call_llm(prompt, temperature=temperature)
        
        final_answer, reasoning_trace = self.parse_response(response.content)

        return self.create_response(
            final_answer=final_answer,
            reasoning_trace=reasoning_trace,
            intermediate_steps=[response.content.strip()],
            metadata={
                "prompt": prompt,
                "model": response.model_name,
                "single_pass": True,
            },
        )