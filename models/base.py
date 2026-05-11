"""
Base definitions for Large Language Model (LLM) interfaces.

This module defines:
- LLMResponse: a standardized dataclass for LLM outputs
- BaseLLM: an abstract base class that enforces a common interface
  for LLM implementations via the `generate` method

The goal of this module is to provide a consistent, extensible
foundation for integrating different LLM providers.

Author: Egor Morozov
Created: 2026-01-26
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, field
from typing import Any, Optional, Dict

@dataclass
class LLMResponse:
    """Container for a standardized LLM generation response."""

    content: str
    model_name: str
    input_tokens: int = 0
    output_tokens: int = 0
    avg_logprob: Optional[float] = None  # average token probability (P_res); None if unsupported
    raw_response: Dict[str, Any] = field(default_factory=dict)

class BaseLLM(ABC):
    """Abstract base class for all LLM implementations."""

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0, logprobs: bool = False) -> LLMResponse:
        """Generate a response for the given prompt.

        Args:
            prompt: The input prompt.
            temperature: Sampling temperature.
            logprobs: If True, compute and populate avg_logprob in the response
                      (average token probability per Eq. 2 of the RoT paper).
                      Falls back to None if the backend does not support it.
        """
        pass
