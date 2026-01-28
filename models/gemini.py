"""
Gemini LLM client implementation.

This module provides a concrete implementation of the BaseLLM interface
using Google's Gemini API. It manages API key resolution, request
configuration, response handling, and error propagation.

Classes:
- GeminiClient: An LLM client that sends prompts to the Gemini API and
  returns standardized LLMResponse objects.

Dependencies:
- Requires a valid Gemini API key, provided either directly or via the
  GEMINI_API_KEY environment variable.

Author: Egor Morozov
"""

import os
from openai import OpenAI
from .base import BaseLLM, LLMResponse

class GeminiClient(BaseLLM):

    def __init__(self, api_key: str = None, model: str ="gemini-2.0-flash-lite"):
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError("Gemini API Key is required.")
        super().__init__(key, model)
        self.client = OpenAI(api_key=self.api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
    
    def generate(self, prompt: str, temperature: float = 0) -> LLMResponse:

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature
            )

            message_content = response.choices[0].message.content
            usage = response.usage

            return LLMResponse(
                content=message_content,
                model_name=self.model,
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
                raw_response=response.model_dump()
            )
        except Exception as e:
            raise RuntimeError(f"Gemini API Error: {e}")