"""
Llama LLM client implementation.

This module provides a concrete implementation of the BaseLLM interface
using the Llama API (OpenAI-compatible endpoint). It handles API key
resolution, request execution, response parsing, and error handling.

Classes:
- LlamaClient: An LLM client that communicates with Llama models via a
  compatible OpenAI-style API and returns standardized LLMResponse objects.

Dependencies:
- Requires a valid Llama API key, provided either directly or via the
  API_KEY environment variable.

Author: Egor Morozov
"""

import math
import os
from openai import OpenAI
from .base import BaseLLM, LLMResponse
from utils.config import get_config

class LlamaClient(BaseLLM):

    def __init__(self, api_key: str = None, model: str ="llama3.1:8b"):
        config = get_config()
        key = api_key or os.getenv("API_KEY") or "local"
        model = model or config["models"]["llama"]
        super().__init__(key, model)
        base_url = config["llm"]["local"]["base_url"]
        self.client = OpenAI(api_key=self.api_key, base_url=base_url, timeout=60.0)

    def generate(self, prompt: str, temperature: float = 0, logprobs: bool = False) -> LLMResponse:
        try:
            kwargs = dict(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=2048,
            )
            if logprobs:
                kwargs["logprobs"] = True

            response = self.client.chat.completions.create(**kwargs)
            message_content = response.choices[0].message.content
            usage = response.usage

            # Ollama's OpenAI-compatible endpoint supports logprobs.
            avg_logprob = None
            if logprobs:
                try:
                    lp_content = response.choices[0].logprobs and response.choices[0].logprobs.content
                    if lp_content:
                        avg_logprob = sum(math.exp(t.logprob) for t in lp_content) / len(lp_content)
                except Exception:
                    pass

            return LLMResponse(
                content=message_content,
                model_name=self.model,
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
                avg_logprob=avg_logprob,
                raw_response=response.model_dump()
            )
        except Exception as e:
            raise RuntimeError(f"Llama API Error: {e}")
