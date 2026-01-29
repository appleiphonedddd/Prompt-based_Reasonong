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
from utils.get_mean_std import AccuracyStatistics

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

MODEL_MAP = {
    "gpt": GPTClient,
    "deepseek": DeepSeekClient,
    "llama": LlamaClient,
    "gemini": GeminiClient,
    "qwen": QwenClient
}

def run(args):
    model_family = args.model.split(':')[0].lower()
    
    if model_family not in MODEL_MAP:
        logger.error(f"Model {model_family} not supported.")
        return

    print(f"\n--- Starting Evaluation ---")
    print(f"Model: {args.model}")
    print(f"Benchmark: {args.benchmark}")
    print(f"Baseline: {args.baseline}")
    print(f"Number of Runs: {args.num_runs}\n")

    stats = AccuracyStatistics()

    for i in range(1, args.num_runs + 1):
        print(f"[Run {i}/{args.num_runs}] ", end="")
        client = MODEL_MAP[model_family](model_name=args.model)
        
        accuracy = Accuracy()
        acc = accuracy.get_accuracy()
        stats.add_result(acc)
        
        print(f"Accuracy: {acc:.2f}%")
        efficiency = Efficiency()
    
    stats.print_summary(baseline_name=args.baseline)
    
    print("\nAll done!")

if __name__ == "__main__":
    
    # general
    parser = argparse.ArgumentParser(description="Prompt-Based Reasoning Evaluation Starting!!!")
    
    parser.add_argument("--model", type=str, default="qwen2.5:14b", help="Model name")

    parser.add_argument("--benchmark", type=str, default="gameof24", help="Dataset")
    
    parser.add_argument("--baseline", type=str, default="ZeroCoT", help="Baseline")
    
    parser.add_argument("--num_runs", type=int, default=1, help="Number of experiment runs")
    
    args = parser.parse_args()

    for arg in vars(args):
        print(arg, '=',getattr(args, arg))

    run(args)