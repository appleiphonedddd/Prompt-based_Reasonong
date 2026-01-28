import argparse
import time
import os

from models.gpt import GPTClient
from models.deepseek import DeepSeekClient
from models.llama import LlamaClient
from models.gemini import GeminiClient
from models.qwen import QwenClient
from utils.metrics import Efficiency

def main():

    parser = argparse.ArgumentParser(description="Prompt-Based Reasoning Evaluation Starting!!!")
    
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model name")

    parser.add_argument("--benchmark", type=str, default="gameof24", help="Dataset")
    
    parser.add_argument("--baseline", type=str, default="ZeroCoT", help="Baseline")
    
    args = parser.parse_args()

    correct_count = 0

    print(f"Experiment Report")

    print(f"Model: {args.model}")
    print(f"Dataset: {args.benchmark}")
    print(f"Baseline: {args.baseline}")


if __name__ == "__main__":
    main()