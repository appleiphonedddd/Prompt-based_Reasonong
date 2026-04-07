import argparse
import time
import os
import logging
from models.gpt import GPTClient
from models.deepseek import DeepSeekClient
from models.llama import LlamaClient
from models.gemini import GeminiClient
from models.qwen import QwenClient
from baseline.RoT import RoT
from baseline.ToT import ToT
from baseline.BoT import BoT
from baseline.GoT import GoT
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

def build_baseline(args, client):
    """Instantiate the requested baseline with its hyper-parameters.

    Args:
        args:   Parsed CLI arguments.
        client: An already-constructed BaseLLM instance.

    Returns:
        A BaseBaseline subclass instance ready to call .run().
    """
    name = args.baseline.lower()

    if name == "rot":
        return RoT(
            llm=client,
            warmup=args.warmup,
            candidate_temperature=args.candidate_temperature,
            instantiation_temperature=args.instantiation_temperature,
        )

    elif name == "tot":
        return ToT(
            llm=client,
            n_generate_sample=args.tot_n_generate,
            n_evaluate_sample=args.tot_n_evaluate,
            breadth_limit=args.tot_breadth,
            max_steps=args.tot_max_steps,
            search_algorithm=args.tot_algorithm,
            value_threshold=args.tot_value_threshold,
            propose_temperature=args.tot_propose_temperature,
            value_temperature=args.tot_value_temperature,
        )  # Reuse ToT's generation params for BoT
    elif name == "bot":
        return BoT(
            llm=client,
            similarity_threshold=args.bot_threshold,
            buffer_path=args.buffer_path,
            distill_temperature=args.bot_distill_temp,
            instantiation_temperature=args.bot_instantiate_temp,
            update_buffer=args.update_buffer,
        )
    elif name == "got":
        return GoT(
            client,
            num_branches      = args.got_branches,
            keep_best         = args.got_keep,
            refine_rounds     = args.got_refine,
            gen_temperature   = args.got_gen_temp,
            score_temperature = args.got_score_temp,
            agg_temperature   = args.got_agg_temp,
        )

    else:
        raise ValueError(f"Unknown baseline: '{args.baseline}'. "
                         "Supported: rot, tot, bot, got")

def run(args):
    model_family = args.model.split(':')[0].lower()

    if model_family not in MODEL_MAP:
        logger.error(f"Model {model_family} not supported.")
        return

    print(f"\n--- Starting Evaluation ---")
    print(f"Model:        {args.model}")
    print(f"Benchmark:    {args.benchmark}")
    print(f"Baseline:     {args.baseline}")
    print(f"Number of Runs: {args.num_runs}")

    stats = AccuracyStatistics()

    for i in range(1, args.num_runs + 1):
        print(f"[Run {i}/{args.num_runs}] ", end="")
        client = MODEL_MAP[model_family](model_name=args.model)

        baseline = build_baseline(args, client)

        accuracy = Accuracy()
        acc = accuracy.get_accuracy()
        stats.add_result(acc)

        print(f"Accuracy: {acc:.2f}%")
        efficiency = Efficiency()

    stats.print_summary(baseline_name=args.baseline)

    print("\nAll done!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Prompt-Based Reasoning Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── General ────────────────────────────────────────────────────────────
    parser.add_argument("--model",     type=str, default="qwen2.5:14b",
                        help="Model name (prefix determines provider, e.g. 'gemini:...')")
    parser.add_argument("--benchmark", type=str, default="gameof24",
                        help="Benchmark / dataset name")
    parser.add_argument("--baseline",  type=str, default="ZeroCoT",
                        help="Baseline to run: rot | tot | ...")
    parser.add_argument("--num_runs",  type=int, default=1,
                        help="Number of independent experiment runs")

    # ── RoT ────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--warmup", type=int, default=5,
        help="[RoT] Number of reverse-reasoning candidates K",
    )
    parser.add_argument(
        "--candidate_temperature", type=float, default=0.7,
        help="[RoT] Sampling temperature for candidate generation",
    )
    parser.add_argument(
        "--instantiation_temperature", type=float, default=0.1,
        help="[RoT] Sampling temperature for instantiation reasoning",
    )

    # ── ToT ────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--tot_algorithm", type=str, default="bfs", choices=["bfs", "dfs"],
        help="[ToT] Search algorithm: breadth-first (bfs) or depth-first (dfs)",
    )
    parser.add_argument(
        "--tot_n_generate", type=int, default=5,
        help="[ToT] Candidate thoughts to propose at each node (k in paper)",
    )
    parser.add_argument(
        "--tot_n_evaluate", type=int, default=3,
        help="[ToT] Value-prompt samples per thought for majority-vote scoring",
    )
    parser.add_argument(
        "--tot_breadth", type=int, default=5,
        help="[ToT][BFS] Frontier width b — states kept per BFS level "
             "(b=5 reproduces the 74%% Game-of-24 result from the paper)",
    )
    parser.add_argument(
        "--tot_max_steps", type=int, default=3,
        help="[ToT] Maximum tree depth T (3 for Game of 24)",
    )
    parser.add_argument(
        "--tot_value_threshold", type=float, default=1.0,
        help="[ToT][DFS] Prune states whose value score falls below this threshold",
    )
    parser.add_argument(
        "--tot_propose_temperature", type=float, default=0.7,
        help="[ToT] Sampling temperature for thought generation (higher → more diverse)",
    )
    parser.add_argument(
        "--tot_value_temperature", type=float, default=0.0,
        help="[ToT] Sampling temperature for state evaluation (0 = deterministic)",
    )

    # ── BoT ──────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--bot_threshold", type=float, default=0.6,
        help="[BoT] Similarity threshold (δ) for template retrieval and updates",
    )
    parser.add_argument(
        "--buffer_path", type=str, default="meta_buffer.json",
        help="[BoT] Path to the JSON file for storing/loading thought-templates",
    )
    parser.add_argument(
        "--bot_distill_temp", type=float, default=0.2,
        help="[BoT] Temperature for problem distillation and template extraction",
    )
    parser.add_argument(
        "--bot_instantiate_temp", type=float, default=0.1,
        help="[BoT] Temperature for final reasoning instantiation",
    )
    parser.add_argument(
        "--no_update_buffer", action="store_false", dest="update_buffer",
        help="[BoT] Disable automatic buffer updating after solving",
    )

    # ── GoT ──────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--got_branches", type=int, default=3,
        help="[GoT] Number of branches to explore at each step",
    )
    parser.add_argument(
        "--got_keep", type=int, default=1,
        help="[GoT] Number of branches to keep after each refinement round",
    )
    parser.add_argument(
        "--got_refine", type=int, default=2,
        help="[GoT] Number of refinement rounds to perform",
    )
    parser.add_argument(
        "--got_gen_temp", type=float, default=0.7,
        help="[GoT] Sampling temperature for branch generation",
    )
    parser.add_argument(
        "--got_score_temp", type=float, default=0.0,
        help="[GoT] Sampling temperature for branch scoring (0 = deterministic)",
    )
    parser.add_argument(
        "--got_agg_temp", type=float, default=0.0,
        help="[GoT] Sampling temperature for final answer aggregation (0 = deterministic)",
    )

    args = parser.parse_args()

    for arg in vars(args):
        print(arg, '=', getattr(args, arg))

    run(args)