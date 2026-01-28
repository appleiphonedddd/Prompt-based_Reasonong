"""
DeepSeek LLM client implementation.

This module provides a concrete implementation of the BaseLLM interface
using the DeepSeek Chat API. It handles API key resolution, request
construction, response parsing, and error handling.

Classes:
- DeepSeekClient: An LLM client that communicates with the Ollama API
  and returns standardized LLMResponse objects.

Dependencies:
- Requires a valid DeepSeek API key, provided either directly or via the
  Ollama API environment variable.

Author: Egor Morozov
"""

import os
from openai import OpenAI
from .base import BaseLLM, LLMResponse
from utils.config import get_config

class DeepSeekClient(BaseLLM):

    def __init__(self, api_key: str = None, model: str ="deepseek-chat"):
        config = get_config()
        key = api_key or os.getenv("API_KEY")
        if not key:
            raise ValueError("API Key is required.")
        model = model or config["models"]["deepseek"]
        super().__init__(key, model)
        base_url = config["llm"]["local"]["base_url"]
        
        self.client = OpenAI(api_key=self.api_key, base_url=base_url)
    
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
            raise RuntimeError(f"API Error: {e}")