import unittest
from unittest.mock import MagicMock, patch
import os

from models.base import LLMResponse
from models.deepseek import DeepSeekClient
from models.gpt import GPTClient
from models.gemini import GeminiClient
from models.llama import LlamaClient

class TestLLMClasses(unittest.TestCase):

    def setUp(self):
        self.fake_api_key = "sk-fake-key-123"

    @patch('models.deepseek.OpenAI')
    def test_deepseek_initialization(self, MockOpenAI):
        client = DeepSeekClient(api_key=self.fake_api_key)
        
        MockOpenAI.assert_called_with(api_key=self.fake_api_key, base_url="https://api.deepseek.com")
        self.assertEqual(client.model, "deepseek-chat")

    @patch('models.deepseek.OpenAI')
    def test_deepseek_generate_success(self, MockOpenAI):
        mock_instance = MockOpenAI.return_value
        
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "DeepSeek response content"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.model_dump.return_value = {"id": "123"}
        
        mock_instance.chat.completions.create.return_value = mock_response

        client = DeepSeekClient(api_key=self.fake_api_key)
        result = client.generate("Hello")

        self.assertIsInstance(result, LLMResponse)
        self.assertEqual(result.content, "DeepSeek response content")
        self.assertEqual(result.input_tokens, 10)
        self.assertEqual(result.output_tokens, 20)
        self.assertEqual(result.model_name, "deepseek-chat")

    @patch('models.llama.OpenAI')
    def test_llama_initialization(self, MockOpenAI):
        client = LlamaClient(api_key=self.fake_api_key)
        
        MockOpenAI.assert_called_with(api_key=self.fake_api_key, base_url="https://api.llama.com/compat/v1/")
        self.assertEqual(client.model, "Llama-3.3-8B-Instruct")

    @patch('models.llama.OpenAI')
    def test_llama_generate_success(self, MockOpenAI):
        mock_instance = MockOpenAI.return_value
        
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Llama response content"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.model_dump.return_value = {"id": "123"}
        
        mock_instance.chat.completions.create.return_value = mock_response

        client = LlamaClient(api_key=self.fake_api_key)
        result = client.generate("Hello")

        self.assertIsInstance(result, LLMResponse)
        self.assertEqual(result.content, "Llama response content")
        self.assertEqual(result.input_tokens, 10)
        self.assertEqual(result.output_tokens, 20)
        self.assertEqual(result.model_name, "Llama-3.3-8B-Instruct")

    @patch('models.gpt.OpenAI')
    def test_gpt_generate_success(self, MockOpenAI):
        mock_instance = MockOpenAI.return_value
        
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "GPT response content"
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.completion_tokens = 5
        mock_response.model_dump.return_value = {}
        
        mock_instance.chat.completions.create.return_value = mock_response

        client = GPTClient(api_key=self.fake_api_key)
        result = client.generate("Hello GPT")

        self.assertEqual(result.content, "GPT response content")
        self.assertEqual(result.model_name, "gpt-4o-mini")

    @patch('models.gemini.genai.Client')
    def test_gemini_generate_success(self, MockGenAIClient):
        mock_instance = MockGenAIClient.return_value

        mock_response = MagicMock()
        mock_response.text = "Gemini response content"
        mock_response.usage_metadata.prompt_token_count = 100
        mock_response.usage_metadata.candidates_token_count = 200
        
        mock_instance.models.generate_content.return_value = mock_response

        client = GeminiClient(api_key=self.fake_api_key)
        result = client.generate("Hello Gemini")

        self.assertEqual(result.content, "Gemini response content")
        self.assertEqual(result.input_tokens, 100)
        self.assertEqual(result.output_tokens, 200)
        self.assertEqual(result.model_name, "gemini-2.0-flash-lite")
    
    def test_missing_api_key_error(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                DeepSeekClient(api_key=None)
    
    @patch('models.gpt.OpenAI')
    def test_api_runtime_error(self, MockOpenAI):
        mock_instance = MockOpenAI.return_value
        mock_instance.chat.completions.create.side_effect = Exception("Network connection error")

        client = GPTClient(api_key=self.fake_api_key)
        
        with self.assertRaises(RuntimeError) as cm:
            client.generate("Test error")
        
        self.assertIn("GPT API Error", str(cm.exception))

if __name__ == '__main__':
    unittest.main()