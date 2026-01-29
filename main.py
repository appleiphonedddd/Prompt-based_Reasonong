import argparse
import time
import os
import logging
from models.gpt import GPTClient
from models.deepseek import DeepSeekClient
from models.llama import LlamaClient
from models.gemini import GeminiClient
from models.qwen import QwenClient
from utils.metrics import Efficiency, Accuracy

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

def run(args):

    model_str = args.model

    efficiency = Efficiency()
    accuracy = Accuracy()

    print("All done!")


if __name__ == "__main__":
    
    # general
    parser = argparse.ArgumentParser(description="Prompt-Based Reasoning Evaluation Starting!!!")
    
    parser.add_argument("--model", type=str, default="qwen2.5:14b", help="Model name")

    parser.add_argument("--benchmark", type=str, default="gameof24", help="Dataset")
    
    parser.add_argument("--baseline", type=str, default="ZeroCoT", help="Baseline")
    
    args = parser.parse_args()

    for arg in vars(args):
        print(arg, '=',getattr(args, arg))

    run(args)