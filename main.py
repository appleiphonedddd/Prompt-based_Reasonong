import argparse
import logging
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

logging.getLogger().setLevel(logging.ERROR)

# ── Registries ────────────────────────────────────────────────────────────────
# To add a new model: insert one entry here.
MODEL_REGISTRY: dict[str, type] = {
    "gpt":      GPTClient,
    "deepseek": DeepSeekClient,
    "llama":    LlamaClient,
    "gemini":   GeminiClient,
    "qwen":     QwenClient,
}

# To add a new baseline: insert one entry here (class, kwargs-extractor).
BASELINE_REGISTRY: dict[str, tuple] = {
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
        self._validate()

    def _validate(self) -> None:
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

    def _build_client(self):
        return MODEL_REGISTRY[self.model_family](model_name=self.args.model)

    def _build_baseline(self, client):
        cls, extract_kwargs = BASELINE_REGISTRY[self.args.baseline.lower()]
        return cls(llm=client, **extract_kwargs(self.args))

    def _run_once(self, run_index: int) -> float:
        print(f"[Run {run_index}/{self.args.num_runs}] ", end="")
        client   = self._build_client()
        baseline = self._build_baseline(client)   # noqa: F841

        accuracy = Accuracy()
        acc = accuracy.get_accuracy()
        print(f"Accuracy: {acc:.2f}%")

        efficiency = Efficiency()                  # noqa: F841
        return acc

    def run(self) -> None:
        args = self.args
        print(f"\n--- Starting Evaluation ---")
        print(f"Model:          {args.model}")
        print(f"Benchmark:      {args.benchmark}")
        print(f"Baseline:       {args.baseline}")
        print(f"Number of Runs: {args.num_runs}")

        stats = AccuracyStatistics()
        for i in range(1, args.num_runs + 1):
            stats.add_result(self._run_once(i))

        stats.print_summary(baseline_name=args.baseline)
        print("\nAll done!")


# ── CLI argument groups ───────────────────────────────────────────────────────
# To add a new baseline: add one _add_<name>_args function and call it below.

def _add_general_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model",     default="qwen2.5:14b",
                        help="Model name (prefix = provider, e.g. 'gemini:...')")
    parser.add_argument("--benchmark", default="gameof24",
                        help="Benchmark / dataset name")
    parser.add_argument("--baseline",  default="ZeroCoT",
                        help="Baseline: rot | tot | bot | got")
    parser.add_argument("--num_runs",  type=int, default=1,
                        help="Independent experiment runs")


def _add_rot_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("RoT")
    g.add_argument("--warmup", type=int, default=5,
                   help="Number of reverse-reasoning candidates K")
    g.add_argument("--candidate_temperature", type=float, default=0.7,
                   help="Sampling temperature for candidate generation")
    g.add_argument("--instantiation_temperature", type=float, default=0.1,
                   help="Sampling temperature for instantiation reasoning")


def _add_tot_args(parser: argparse.ArgumentParser) -> None:
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


def _add_bot_args(parser: argparse.ArgumentParser) -> None:
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


def _add_got_args(parser: argparse.ArgumentParser) -> None:
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
    _add_general_args(parser)
    _add_rot_args(parser)
    _add_tot_args(parser)
    _add_bot_args(parser)
    _add_got_args(parser)
    return parser


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = build_parser().parse_args()

    for key, val in vars(args).items():
        print(f"{key} = {val}")

    Evaluator(args).run()
