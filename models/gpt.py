"""
OpenAI GPT client implementation.

This module provides a concrete implementation of the BaseLLM interface
using OpenAI's Chat Completions API. It handles API key resolution,
request execution, response parsing, and error handling.

Classes:
- GPTClient: An LLM client that communicates with OpenAI models and
  returns standardized LLMResponse objects.

Dependencies:
- Requires a valid OpenAI API key, provided either directly or via the
  OPENAI_API_KEY environment variable.

Author: Egor Morozov
"""

import math
import os
from openai import OpenAI
from .base import BaseLLM, LLMResponse

class GPTClient(BaseLLM):

    def __init__(self, api_key: str = None, model: str = "gpt-4o-mini"):
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OpenAI API Key is required.")
        super().__init__(key, model)
        self.client = OpenAI(api_key=self.api_key)

    def generate(self, prompt: str, temperature: float = 0, logprobs: bool = False) -> LLMResponse:
        try:
            kwargs = dict(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            if logprobs:
                kwargs["logprobs"] = True

            response = self.client.chat.completions.create(**kwargs)
            message_content = response.choices[0].message.content
            usage = response.usage

            avg_logprob = None
            if logprobs:
                lp_content = (response.choices[0].logprobs or {}) and response.choices[0].logprobs.content
                if lp_content:
                    avg_logprob = sum(math.exp(t.logprob) for t in lp_content) / len(lp_content)

            return LLMResponse(
                content=message_content,
                model_name=self.model,
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
                avg_logprob=avg_logprob,
                raw_response=response.model_dump()
            )
        except Exception as e:
            raise RuntimeError(f"GPT API Error: {e}")
