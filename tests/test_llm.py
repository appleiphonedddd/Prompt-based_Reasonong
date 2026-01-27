import unittest
from unittest.mock import MagicMock, patch
import os

from models.base import LLMResponse, BaseLLM
from models.deepseek import DeepSeekClient
from models.gpt import GPTClient
from models.gemini import GeminiClient
from models.llama import LlamaClient


class TestLLMClasses(unittest.TestCase):

    def setUp(self):
        self.fake_api_key = "sk-fake-key-123"

    # ---------- Base / Interface tests ----------

    def test_base_llm_is_abstract(self):
        with self.assertRaises(TypeError):
            BaseLLM(api_key="x", model="y")

    # ---------- Helper Methods for OpenAI-compatible Clients ----------

    def assert_openai_style_client(self, client_cls, patch_target, expected_model, expected_base_url=None):
        """
        Generic helper to test the full lifecycle of an OpenAI-compatible client.
        Tests: Initialization, Generate Success, Temperature Forwarding, and Error Handling.
        """
        with patch(patch_target) as MockOpenAI:
            # 1. Setup Mock Response
            mock_instance = MockOpenAI.return_value
            mock_response = MagicMock()
            mock_response.choices[0].message.content = "Mocked Response Content"
            mock_response.usage.prompt_tokens = 42
            mock_response.usage.completion_tokens = 84
            mock_response.model_dump.return_value = {"raw_info": "test"}
            mock_instance.chat.completions.create.return_value = mock_response

            # 2. Test Initialization
            client = client_cls(api_key=self.fake_api_key)
            
            # Verify OpenAI client was init with correct args
            init_kwargs = {"api_key": self.fake_api_key}
            if expected_base_url:
                init_kwargs["base_url"] = expected_base_url
            MockOpenAI.assert_called_with(**init_kwargs)
            
            self.assertEqual(client.model, expected_model)

            # 3. Test Generate Success
            result = client.generate("Test Prompt")
            
            self.assertIsInstance(result, LLMResponse)
            self.assertEqual(result.content, "Mocked Response Content")
            self.assertEqual(result.input_tokens, 42)
            self.assertEqual(result.output_tokens, 84)
            self.assertEqual(result.model_name, expected_model)
            self.assertEqual(result.raw_response, {"raw_info": "test"})

            # 4. Test Temperature Forwarding
            client.generate("Test Temp", temperature=0.9)
            call_args = mock_instance.chat.completions.create.call_args_list[-1]
            self.assertEqual(call_args.kwargs["temperature"], 0.9)
            self.assertEqual(call_args.kwargs["model"], expected_model)

            # 5. Test API Runtime Error
            mock_instance.chat.completions.create.side_effect = Exception("API Connection Failed")
            with self.assertRaises(RuntimeError) as cm:
                client.generate("Should Fail")
            # Verify the class name is likely part of the error message or context
            self.assertTrue(any(x in str(cm.exception) for x in ["Error", "API"]))

    def assert_env_fallback(self, client_cls, patch_target, env_var_name, expected_base_url=None):
        """Helper to test API Key fallback from environment variables."""
        test_env_key = f"env-key-for-{env_var_name}"
        
        # Ensure no explicit key is passed, so it looks for ENV
        with patch.dict(os.environ, {env_var_name: test_env_key}, clear=True):
            with patch(patch_target) as MockOpenAI:
                client = client_cls(api_key=None)
                
                init_kwargs = {"api_key": test_env_key}
                if expected_base_url:
                    init_kwargs["base_url"] = expected_base_url
                MockOpenAI.assert_called_with(**init_kwargs)
                
                self.assertEqual(client.api_key, test_env_key)

    def assert_missing_key_error(self, client_cls):
        """Helper to test ValueError when no key is provided."""
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                client_cls(api_key=None)

    # ---------- Concrete Implementation Tests (DRY Version) ----------

    def test_deepseek_full_flow(self):
        self.assert_openai_style_client(
            DeepSeekClient,
            patch_target='models.deepseek.OpenAI',
            expected_model="deepseek-chat",
            expected_base_url="https://api.deepseek.com"
        )

    def test_deepseek_env_fallback(self):
        self.assert_env_fallback(
            DeepSeekClient,
            patch_target='models.deepseek.OpenAI',
            env_var_name="DEEPSEEK_API_KEY",
            expected_base_url="https://api.deepseek.com"
        )
    
    def test_deepseek_missing_key(self):
        self.assert_missing_key_error(DeepSeekClient)

    def test_gpt_full_flow(self):
        self.assert_openai_style_client(
            GPTClient,
            patch_target='models.gpt.OpenAI',
            expected_model="gpt-4o-mini",
            expected_base_url=None # GPT doesn't set a custom base_url by default
        )

    def test_gpt_env_fallback(self):
        self.assert_env_fallback(
            GPTClient,
            patch_target='models.gpt.OpenAI',
            env_var_name="OPENAI_API_KEY",
            expected_base_url=None
        )

    def test_gpt_missing_key(self):
        self.assert_missing_key_error(GPTClient)

    def test_llama_full_flow(self):
        self.assert_openai_style_client(
            LlamaClient,
            patch_target='models.llama.OpenAI',
            expected_model="Llama-3.3-8B-Instruct",
            expected_base_url="https://api.llama.com/compat/v1/"
        )

    def test_llama_env_fallback(self):
        self.assert_env_fallback(
            LlamaClient,
            patch_target='models.llama.OpenAI',
            env_var_name="LLAMA_API_KEY",
            expected_base_url="https://api.llama.com/compat/v1/"
        )

    def test_llama_missing_key(self):
        self.assert_missing_key_error(LlamaClient)

    def test_gemini_full_flow(self):
        self.assert_openai_style_client(
            GeminiClient,
            patch_target='models.gemini.OpenAI',
            expected_model="gemini-2.0-flash-lite",
            expected_base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )

    def test_gemini_env_fallback(self):
        self.assert_env_fallback(
            GeminiClient,
            patch_target='models.gemini.OpenAI',
            env_var_name="GEMINI_API_KEY",
            expected_base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )

    def test_gemini_missing_key(self):
        self.assert_missing_key_error(GeminiClient)


if __name__ == '__main__':
    unittest.main()
    