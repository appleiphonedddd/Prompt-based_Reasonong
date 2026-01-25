from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, field
from typing import Any, Optional, Dict

@dataclass

class LLMResponse:
    content: str
    model_name: str
    input_tokens: int = 0
    output_tokens: int = 0
    raw_response: Dict[str, Any] = field(default_factory=dict)

class BaseLLM(ABC):

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
    
    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0) -> LLMResponse:
        pass