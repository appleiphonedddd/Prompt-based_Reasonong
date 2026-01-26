import os
from openai import OpenAI
from .base import BaseLLM, LLMResponse

class LlamaClient(BaseLLM):

    def __init__(self, api_key: str = None, model: str ="Llama-3.3-8B-Instruct"):
        key = api_key or os.getenv("LLAMA_API_KEY")
        if not key:
            raise ValueError("Llama API Key is required.")
        super().__init__(key, model)
        self.client = OpenAI(api_key=self.api_key, base_url="https://api.llama.com/compat/v1/")
    
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
            raise RuntimeError(f"Llama API Error: {e}")