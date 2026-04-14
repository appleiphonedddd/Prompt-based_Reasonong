import argparse
import logging
import time
from models.gpt import GPTClient
from models.deepseek import DeepSeekClient
from models.llama import LlamaClient
from models.gemini import GeminiClient
from models.qwen import QwenClient
from baseline.CoT import ZeroShotCoT, ZeroShotCoTSinglePass
from baseline.RoT import RoT
from baseline.ToT import ToT
from baseline.BoT import BoT
from baseline.GoT import GoT
from baseline.Standard import Input
from utils.metrics import Efficiency, Accuracy
from utils.get_mean_std import AccuracyStatistics
from benchmark import DATASET_REGISTRY

logging.getLogger().setLevel(logging.ERROR)

# ── Registries ────────────────────────────────────────────────────────────────
# To add a new model: insert one entry here.
MODEL_REGISTRY: dict[str, type] = {
    "gpt":      GPTClient,
    "deepseek": DeepSeekClient,
    "llama":    LlamaClient,
    "llama3.1":    LlamaClient,
    "llama3.3":    LlamaClient,
    "llama2":    LlamaClient,
    "gemini":   GeminiClient,
    "qwen":     QwenClient,
    "qwen2":    QwenClient,
    "qwen2.5":  QwenClient,
    "qwen3":    QwenClient,
}

# To add a new baseline: insert one entry here (class, kwargs-extractor).
BASELINE_REGISTRY: dict[str, tuple] = {
    "standard":       (Input,                  lambda _: {}),
    "zerocot":        (ZeroShotCoT,            lambda _: {}),
    "zerocot_single": (ZeroShotCoTSinglePass,  lambda _: {}),
    "rot": (RoT, lambda a: dict(
        warmup=a.warmup,
        candidate_temperature=a.candidate_temperature,
        instantiation_temperature=a.instantiation_temperature,
    )),
    "tot": (ToT, lambda a: dict(
        n_generate_sample=a.tot_n_generate,
        n_evaluate_sample=a.tot_n_evaluate,
        breadth_limit=a.tot_breadth,
        max_steps=a.tot_max_steps,
        search_algorithm=a.tot_algorithm,
        value_threshold=a.tot_value_threshold,
        propose_temperature=a.tot_propose_temperature,
        value_temperature=a.tot_value_temperature,
    )),
    "bot": (BoT, lambda a: dict(
        similarity_threshold=a.bot_threshold,
        buffer_path=a.buffer_path,
        distill_temperature=a.bot_distill_temp,
        instantiation_temperature=a.bot_instantiate_temp,
        update_buffer=a.update_buffer,
    )),
    "got": (GoT, lambda a: dict(
        num_branches=a.got_branches,
        keep_best=a.got_keep,
        refine_rounds=a.got_refine,
        gen_temperature=a.got_gen_temp,
        score_temperature=a.got_score_temp,
        agg_temperature=a.got_agg_temp,
    )),
}


# ── Evaluator ─────────────────────────────────────────────────────────────────
class Evaluator:
    """Orchestrates a single benchmark evaluation session."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.model_family = args.model.split(":")[0].lower()
        self.validate()

    def validate(self) -> None:
        if self.model_family not in MODEL_REGISTRY:
            raise ValueError(
                f"Model '{self.model_family}' not supported. "
                f"Supported providers: {list(MODEL_REGISTRY)}"
            )
        if self.args.baseline.lower() not in BASELINE_REGISTRY:
            raise ValueError(
                f"Baseline '{self.args.baseline}' not supported. "
                f"Supported: {list(BASELINE_REGISTRY)}"
            )
        if self.args.benchmark.lower() not in DATASET_REGISTRY:
            raise ValueError(
                f"Benchmark '{self.args.benchmark}' not supported. "
                f"Supported: {list(DATASET_REGISTRY)}"
            )

    def build_client(self):
        return MODEL_REGISTRY[self.model_family](model=self.args.model)

    def build_baseline(self, client):
        cls, extract_kwargs = BASELINE_REGISTRY[self.args.baseline.lower()]
        return cls(llm=client, **extract_kwargs(self.args))

    def build_dataset(self):
        benchmark_key = self.args.benchmark.lower()
        cls, extract_kwargs = DATASET_REGISTRY[benchmark_key]
        dataset = cls(**extract_kwargs(self.args))
        dataset.load_dataset()
        return dataset

    def run_once(self, run_index: int, dataset, efficiency: Efficiency) -> float:
        print(f"\n[Run {run_index}/{self.args.num_runs}]")
        client   = self.build_client()
        baseline = self.build_baseline(client)
        accuracy = Accuracy()

        n = len(dataset)

        task_times: list[float] = []
        for i in range(n):
            problem = dataset.get_problem(i)
            t0 = time.perf_counter()
            try:
                response = baseline.run(
                    problem.question,
                    system_prompt=dataset.get_system_prompt(),
                    instruction=dataset.get_instruction(),
                )
            except Exception as exc:
                elapsed = time.perf_counter() - t0
                task_times.append(elapsed)
                accuracy.record(False)
                print(f"  [{i + 1}/{n}] ✗  ERROR ({elapsed:.1f}s): {exc!r}")
                continue
            elapsed = time.perf_counter() - t0
            task_times.append(elapsed)

            result = dataset.evaluate_answer(response.final_answer, problem.ground_truth)
            accuracy.record(result.is_correct)
            mark = "✓" if result.is_correct else "✗"
            print(f"  [{i + 1}/{n}] {mark}  ({elapsed:.1f}s)  answer={response.final_answer!r}")

        efficiency.record_sample(task_times)
        acc = accuracy.get_accuracy()
        print(f"  Accuracy: {acc:.2f}%")
        return acc

    def run(self) -> None:
        args = self.args
        print(f"\n--- Starting Evaluation ---")
        print(f"Model:          {args.model}")
        print(f"Benchmark:      {args.benchmark}")
        print(f"Baseline:       {args.baseline}")
        print(f"Number of Runs: {args.num_runs}")

        dataset = self.build_dataset()
        n = len(dataset)
        print(f"Questions:      {n}")
        efficiency = Efficiency(num_tasks=n)

        stats = AccuracyStatistics()
        for i in range(1, args.num_runs + 1):
            stats.add_result(self.run_once(i, dataset, efficiency))

        stats.print_summary(baseline_name=args.baseline)
        print(f"Avg time/question: {efficiency.get_T():.2f}s  (over {efficiency.get_M()} run(s))")
        print("\nAll done!")


# ── CLI argument groups ───────────────────────────────────────────────────────
# To add a new baseline: add one _add_<name>_args function and call it below.

def general_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model",        default="qwen2.5:14b",
                        help="Model name (prefix = provider, e.g. 'gemini:...')")
    parser.add_argument("--benchmark",    default="gameof24",
                        help="Benchmark / dataset name")
    parser.add_argument("--baseline",     default="zerocot",
                        help="Baseline: standard | zerocot | zerocot_single | rot | tot | bot | got")
    parser.add_argument("--num_runs",     type=int, default=1,
                        help="Independent experiment runs")
    parser.add_argument("--language",     default="en",
                        help="Language for MGSM benchmark (en, de, fr, es, ru, zh, ja, th, sw, bn)")

def rot_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("RoT")
    g.add_argument("--warmup", type=int, default=5,
                   help="Number of reverse-reasoning candidates K")
    g.add_argument("--candidate_temperature", type=float, default=0.7,
                   help="Sampling temperature for candidate generation")
    g.add_argument("--instantiation_temperature", type=float, default=0.1,
                   help="Sampling temperature for instantiation reasoning")


def tot_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("ToT")
    g.add_argument("--tot_algorithm", default="bfs", choices=["bfs", "dfs"],
                   help="Search algorithm: breadth-first (bfs) or depth-first (dfs)")
    g.add_argument("--tot_n_generate", type=int, default=5,
                   help="Candidate thoughts to propose at each node (k in paper)")
    g.add_argument("--tot_n_evaluate", type=int, default=3,
                   help="Value-prompt samples per thought for majority-vote scoring")
    g.add_argument("--tot_breadth", type=int, default=5,
                   help="[BFS] Frontier width b — states kept per BFS level")
    g.add_argument("--tot_max_steps", type=int, default=3,
                   help="Maximum tree depth T (3 for Game of 24)")
    g.add_argument("--tot_value_threshold", type=float, default=1.0,
                   help="[DFS] Prune states whose value score falls below this threshold")
    g.add_argument("--tot_propose_temperature", type=float, default=0.7,
                   help="Sampling temperature for thought generation (higher → more diverse)")
    g.add_argument("--tot_value_temperature", type=float, default=0.0,
                   help="Sampling temperature for state evaluation (0 = deterministic)")


def bot_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("BoT")
    g.add_argument("--bot_threshold", type=float, default=0.6,
                   help="Similarity threshold (δ) for template retrieval and updates")
    g.add_argument("--buffer_path", default="meta_buffer.json",
                   help="Path to the JSON file for storing/loading thought-templates")
    g.add_argument("--bot_distill_temp", type=float, default=0.2,
                   help="Temperature for problem distillation and template extraction")
    g.add_argument("--bot_instantiate_temp", type=float, default=0.1,
                   help="Temperature for final reasoning instantiation")
    g.add_argument("--no_update_buffer", action="store_false", dest="update_buffer",
                   help="Disable automatic buffer updating after solving")


def got_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("GoT")
    g.add_argument("--got_branches", type=int, default=3,
                   help="Number of branches to explore at each step")
    g.add_argument("--got_keep", type=int, default=1,
                   help="Number of branches to keep after each refinement round")
    g.add_argument("--got_refine", type=int, default=2,
                   help="Number of refinement rounds to perform")
    g.add_argument("--got_gen_temp", type=float, default=0.7,
                   help="Sampling temperature for branch generation")
    g.add_argument("--got_score_temp", type=float, default=0.0,
                   help="Sampling temperature for branch scoring (0 = deterministic)")
    g.add_argument("--got_agg_temp", type=float, default=0.0,
                   help="Sampling temperature for final answer aggregation (0 = deterministic)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prompt-Based Reasoning Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    general_args(parser)
    rot_args(parser)
    tot_args(parser)
    bot_args(parser)
    got_args(parser)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()

    for key, val in vars(args).items():
        print(f"{key} = {val}")

    Evaluator(args).run()
