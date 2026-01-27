"""
Input Prompting Implementation.

Input prompting is the simplest form of prompting where the model
is given a question directly without any examples or demonstrations.
The model must rely solely on its pre-trained knowledge.

Reference:
- Brown et al., "Language Models are Few-Shot Learners" (GPT-3 Paper)

Author: Egor Morozov
"""

from typing import Optional

from baseline.basebaseline import BaseBaseline, BaselineResponse
from models.base import BaseLLM

class Input(BaseBaseline):
    """Zero-Shot prompting baseline.
    
    This baseline directly prompts the LLM with the question without
    providing any examples or demonstrations.
    
    Example:
        >>> llm = GPTClient()
        >>> baseline = ZeroShot(llm)
        >>> response = baseline.run("What is the capital of France?")
        >>> print(response.final_answer)
        "Paris"
    """

    def __init__(self, llm: BaseLLM):
        """Initialize Zero-Shot baseline.
        
        Args:
            llm: An instance of a BaseLLM subclass.
        """
        super().__init__(llm, baseline_name="ZeroShot")

    def build_prompt(
        self,
        question: str,
        system_prompt: Optional[str] = None,
        instruction: Optional[str] = None,
    ) -> str:
        """Build the zero-shot prompt.
        
        Args:
            question: The input question to answer.
            system_prompt: Optional system-level instruction.
            instruction: Optional task-specific instruction.
            
        Returns:
            The formatted prompt string.
        """
        parts = []

        if system_prompt:
            parts.append(system_prompt)

        if instruction:
            parts.append(instruction)

        parts.append(f"Question: {question}")
        parts.append("Answer:")

        return "\n\n".join(parts)

    def run(
        self,
        question: str,
        system_prompt: Optional[str] = None,
        instruction: Optional[str] = None,
        temperature: float = 0.0,
        **kwargs,
    ) -> BaselineResponse:
        """Execute zero-shot prompting on the given question.
        
        Args:
            question: The input question to answer.
            system_prompt: Optional system-level instruction 
                          (e.g., "You are a helpful assistant.").
            instruction: Optional task-specific instruction
                        (e.g., "Solve the following math problem.").
            temperature: Sampling temperature for generation.
            **kwargs: Additional arguments (unused, for interface compatibility).
            
        Returns:
            BaselineResponse containing the answer and metrics.
        """
        self.reset_counters()

        prompt = self.build_prompt(question, system_prompt, instruction)
        response = self.call_llm(prompt, temperature=temperature)

        return self.create_response(
            final_answer=response.content.strip(),
            reasoning_trace="",  # No explicit reasoning in zero-shot
            intermediate_steps=[],
            metadata={
                "prompt": prompt,
                "model": response.model_name,
            },
        )