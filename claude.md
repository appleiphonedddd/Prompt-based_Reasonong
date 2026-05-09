# Claude.md: Prompt-based Reasoning Project Guide

## Project Overview

**Prompt-based Reasoning** is a comprehensive framework for evaluating and benchmarking Large Language Models (LLMs) on complex reasoning tasks using various prompting strategies. The project enables systematic comparison of:

- **Multiple LLM Providers**: OpenAI (GPT), DeepSeek, Meta (Llama), Google (Gemini), Alibaba (Qwen)
- **Prompting Baselines**: Standard input, Zero-Shot CoT (two-step & single-pass), Reversal-of-Thought (RoT), Tree-of-Thought (ToT), Buffer-of-Thought (BoT), and Graph-of-Thought (GoT)
- **Reasoning Benchmarks**: Game of 24, MGSM, Sonnet Writing, BigBenchHard (27 tasks), Programming Puzzles, HumanEval, MBPP, APPS, ClassEval, CRUXEval

The framework provides standardized evaluation metrics (accuracy, efficiency) and supports both local models (via Ollama) and cloud-based APIs.

---

## Technology Stack & Versions

### Core Dependencies
- **Python**: 3.11.14 (via Conda)
- **PyTorch**: 2.11.0 (with CUDA 13.0 support for GPU acceleration)
- **NumPy**: 2.4.1
- **Pandas**: 3.0.0
- **CUDA Toolkit**: 13.0.2 (for GPU support)

### LLM & API Integration
- **OpenAI SDK**: 2.15.0 (for GPT models)
- **Requests/HTTPX**: 2.32.5 / 0.28.1 (HTTP client libraries)
- **YAML Configuration**: PyYAML 6.0.3

### Data & ML Libraries
- **Pydantic**: 2.12.5 (data validation and settings management)
- **Datasets**: 4.5.0 (HuggingFace benchmark datasets)
- **Huggingface Hub**: 1.3.4
- **PyArrow**: 23.0.0 (columnar data format)
- **scikit-learn**: similarity computations (used in BoT for template retrieval)
- **scipy**: statistical analysis
- **sentence-transformers**: semantic embedding for BoT similarity threshold

### Environment & Deployment
- **Conda**: Environment management (virtual environment in `/home/infor/miniconda3/envs/Prompt`)
- **Ollama**: Local LLM inference engine for GPU-accelerated model serving
- **NVIDIA Container Toolkit**: For GPU support (if using Docker containers)

### Development Tools
- **Typer / Click**: CLI framework (argparse used directly in `main.py`)
- **Logging**: Python's built-in logging module

---

## Directory Structure & Architecture

```
Prompt-based-Reasoning/
├── main.py                    # Entry point with Model/Baseline/Dataset registries
├── config.yaml               # LLM API endpoints & model configurations
├── env.yaml                  # Conda environment specification
├── setup_ollama_gpu.sh       # GPU/Docker environment setup script
│
├── models/                   # LLM Provider Implementations
│   ├── base.py              # Abstract BaseLLM class & LLMResponse dataclass
│   ├── gpt.py               # OpenAI GPT implementation
│   ├── gemini.py            # Google Gemini implementation
│   ├── deepseek.py          # DeepSeek LLM implementation
│   ├── llama.py             # Meta Llama (via Ollama) implementation
│   └── qwen.py              # Alibaba Qwen (via Ollama) implementation
│
├── baseline/                # Prompting Strategy Implementations
│   ├── basebaseline.py      # Abstract BaseBaseline class & BaselineResponse
│   ├── Standard/
│   │   └── io.py            # Standard input-output baseline
│   ├── CoT/
│   │   └── zero_shot_cot.py  # ZeroShotCoT (2-step) & ZeroShotCoTSinglePass
│   ├── RoT/
│   │   └── rot.py            # Reversal-of-Thought (with caching & parallelism)
│   ├── ToT/
│   │   └── tot.py            # Tree-of-Thought (BFS/DFS)
│   ├── BoT/
│   │   └── bot.py            # Buffer-of-Thought (meta-buffer + similarity retrieval)
│   └── GoT/
│       └── got.py            # Graph-of-Thought
│
├── benchmark/               # Reasoning Task Datasets
│   ├── __init__.py          # DATASET_REGISTRY (10 benchmarks)
│   ├── datasetbase.py       # Abstract DatasetBase, Problem, EvaluationResult
│   ├── GameOf24/            # Arithmetic puzzle: combine 4 numbers to reach 24
│   ├── MGSM/                # Multilingual Grade School Math (10 languages)
│   ├── SonnetWriting/       # Shakespearean sonnet generation
│   ├── BigBenchHard/        # All 27 BIG-Bench Hard tasks
│   ├── ProgrammingPuzzles/  # Python programming puzzles (sat-function verification)
│   ├── HumanEval/           # 164 Python function completion tasks (OpenAI)
│   ├── MBPP/                # 974 Python function generation tasks (Google)
│   ├── APPS/                # 5000 competitive-programming problems
│   ├── ClassEval/           # 100 class-level Python implementation tasks
│   └── CRUXEval/            # 799 code output-prediction tasks (CRUXEval-O)
│
├── utils/                   # Utility Modules
│   ├── config.py            # Configuration loading & validation
│   ├── metrics.py           # Accuracy & Efficiency metric classes
│   └── get_mean_std.py      # Statistical analysis (mean, std dev)
│
└── tests/                   # Test Suite
```

### Key Architectural Patterns

1. **Registry Pattern** (`main.py`)
   - `MODEL_REGISTRY`: Maps model name prefixes to LLM client classes
   - `BASELINE_REGISTRY`: Maps baseline names to strategy classes + argument extractors
   - `DATASET_REGISTRY` (`benchmark/__init__.py`): Maps benchmark names to dataset classes

2. **Abstract Base Classes**
   - `BaseLLM` in `models/base.py`: Enforces `generate()` method interface
   - `BaseBaseline` in `baseline/basebaseline.py`: Enforces `run()` method interface
   - `DatasetBase` in `benchmark/datasetbase.py`: Enforces `load_dataset()`, `get_problem()`, `evaluate_answer()`

3. **Dataclass-based Response Objects**
   - `LLMResponse`: Standardized LLM generation output (content, model_name, input_tokens, output_tokens)
   - `BaselineResponse`: Unified benchmark evaluation result (final_answer, reasoning_trace)
   - `Problem`: Benchmark problem (index, question, ground_truth, metadata)
   - `EvaluationResult`: Evaluation outcome (is_correct, score, prediction, ground_truth, details)

---

## Baselines

| Key | Class | Description |
|-----|-------|-------------|
| `standard` | `Input` | Direct prompt → answer |
| `zerocot` | `ZeroShotCoT` | Two-step: reasoning + answer extraction |
| `zerocot_single` | `ZeroShotCoTSinglePass` | Single-pass: "Let's think step by step" inline |
| `rot` | `RoT` | Reversal-of-Thought: generate K reverse-reasoning candidates, cache Stage 1+2, parallelize LLM calls |
| `tot` | `ToT` | Tree-of-Thought: BFS or DFS over thought tree |
| `bot` | `BoT` | Buffer-of-Thought: meta-buffer with semantic template retrieval |
| `got` | `GoT` | Graph-of-Thought: branch-score-aggregate refinement loops |

### ZeroShotCoT vs ZeroShotCoTSinglePass
- **ZeroShotCoT** (`zerocot`): Two LLM calls — (1) elicit reasoning with "Let's think step by step", (2) extract final answer
- **ZeroShotCoTSinglePass** (`zerocot_single`): One LLM call — append the phrase and parse the answer directly from the response
- For generative tasks (code generation, sonnets), both detect `is_generative_task` and skip the second extraction pass

### BoT Buffer
- Templates stored in `meta_buffer.json` (configurable via `--buffer_path`)
- Similarity matching via sentence-transformers embeddings + cosine similarity
- `--no_update_buffer` disables automatic buffer updates after each solve

---

## Benchmarks

### Math & Reasoning
| Key | Class | Size | Description |
|-----|-------|------|-------------|
| `gameof24` | `GameOf24` | ~1362 | Arithmetic puzzle: combine 4 numbers to reach 24 using +−×÷ |
| `mgsm` | `MGSM` | 250×lang | Multilingual Grade School Math (10 languages: en, de, fr, es, ru, zh, ja, th, sw, bn) |
| `bigbenchhard` | `BigBenchHard` | 250/task | All 27 BIG-Bench Hard tasks (use `--bigbenchhard_task`) |
| `sonnetwriting` | `SonnetWriting` | 20 | Shakespearean sonnet generation with constraints |

### Programming
| Key | Class | Size | Description |
|-----|-------|------|-------------|
| `humaneval` | `HumanEval` | 164 | Python function completion from docstring (OpenAI HumanEval) |
| `mbpp` | `MBPP` | 974 | Python function generation from problem description (Google MBPP) |
| `apps` | `APPS` | 5000 | Competitive programming (stdin/stdout + fn_name formats) |
| `classeval` | `ClassEval` | 100 | Class-level Python implementation (methods + class structure) |
| `cruxeval` | `CRUXEval` | 799 | Code output prediction: given `f` and input, predict return value |
| `programmingpuzzles` | `ProgrammingPuzzles` | 1715 | Python puzzles with sat-function verification |

### APPS Evaluation Details
APPS has two problem formats:
- **stdin/stdout** (~99%): Model writes a program reading from `input()`/`sys.stdin`, mocked at eval time
- **fn_name** (~1%): LeetCode-style — model implements a named function or `Solution` class method

Answer extraction always takes the **last** fenced code block so CoT baselines (which may emit wrong code before the corrected solution) are handled correctly.

### CRUXEval Evaluation Details
Model must predict the exact Python literal returned by a given function `f(*args)`. Evaluation uses `ast.literal_eval` when possible, falling back to normalized string comparison.

### BigBenchHard Task Categories
- **Boolean/Yes-No** (6): `boolean_expressions`, `causal_judgement`, `formal_fallacies`, `navigate`, `sports_understanding`, `web_of_lies`
- **Multiple-Choice letter** (17): `date_understanding`, `disambiguation_qa`, `geometric_shapes`, `hyperbaton`, `logical_deduction_{3,5,7}_objects`, `movie_recommendation`, `penguins_in_a_table`, `reasoning_about_colored_objects`, `ruin_names`, `salient_translation_error_detection`, `snarks`, `temporal_sequences`, `tracking_shuffled_objects_{3,5,7}_objects`
- **Numeric** (2): `multistep_arithmetic_two`, `object_counting`
- **Free-form** (2): `dyck_languages`, `word_sorting`

---

## CLI Interface

```bash
python main.py --model <model> --baseline <baseline> --benchmark <benchmark> [options]
```

### General Options
| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `qwen2.5:14b` | Model name (prefix = provider, e.g. `gpt:gpt-4o`) |
| `--baseline` | `zerocot` | `standard \| zerocot \| zerocot_single \| rot \| tot \| bot \| got` |
| `--benchmark` | `gameof24` | See benchmark keys in table above |
| `--num_runs` | `1` | Independent experiment runs |
| `--language` | `all` | MGSM language (`en, de, fr, es, ru, zh, ja, th, sw, bn` or `all`) |
| `--languages` | None | MGSM: space-separated list, overrides `--language` |
| `--bigbenchhard_task` | `boolean_expressions` | BigBenchHard task name |
| `--split` | `test` | Dataset split (used by BigBenchHard) |

### RoT Options
| Flag | Default | Description |
|------|---------|-------------|
| `--warmup` | `5` | Number of reverse-reasoning candidates K |
| `--candidate_temperature` | `0.7` | Temperature for candidate generation |
| `--instantiation_temperature` | `0.1` | Temperature for instantiation reasoning |

### ToT Options
| Flag | Default | Description |
|------|---------|-------------|
| `--tot_algorithm` | `bfs` | Search: `bfs` or `dfs` |
| `--tot_n_generate` | `5` | Candidate thoughts per node |
| `--tot_n_evaluate` | `3` | Value-prompt samples per thought |
| `--tot_breadth` | `5` | BFS frontier width |
| `--tot_max_steps` | `3` | Max tree depth |
| `--tot_value_threshold` | `1.0` | DFS pruning threshold |
| `--tot_propose_temperature` | `0.7` | Temperature for thought generation |
| `--tot_value_temperature` | `0.0` | Temperature for state evaluation |

### BoT Options
| Flag | Default | Description |
|------|---------|-------------|
| `--bot_threshold` | `0.6` | Similarity threshold δ for template retrieval |
| `--buffer_path` | `meta_buffer.json` | Path to JSON file for thought-templates |
| `--bot_distill_temp` | `0.2` | Temperature for problem distillation |
| `--bot_instantiate_temp` | `0.1` | Temperature for final reasoning instantiation |
| `--no_update_buffer` | — | Disable automatic buffer updates |

### GoT Options
| Flag | Default | Description |
|------|---------|-------------|
| `--got_branches` | `3` | Number of branches to explore |
| `--got_keep` | `1` | Branches kept after each refinement |
| `--got_refine` | `2` | Refinement rounds |
| `--got_gen_temp` | `0.7` | Temperature for branch generation |
| `--got_score_temp` | `0.0` | Temperature for branch scoring |
| `--got_agg_temp` | `0.0` | Temperature for final aggregation |

### ProgrammingPuzzles Options
| Flag | Default | Description |
|------|---------|-------------|
| `--pp_num_samples` | None (all) | Number of puzzles to evaluate |
| `--pp_module` | None (all) | Filter by module (e.g. `study.py`, `basic.py`, `IMO.py`) |

---

## Usage Examples

```bash
# Activate environment
conda activate Prompt

# Local model via Ollama
python main.py --model qwen2.5:14b --baseline zerocot --benchmark gameof24

# BoT with custom buffer
python main.py --model qwen2.5:14b --baseline bot --benchmark gameof24 \
  --buffer_path my_buffer.json --bot_threshold 0.7

# MGSM: all 10 languages
python main.py --model gemini:gemini-2.0-flash --baseline zerocot_single --benchmark mgsm

# MGSM: specific languages
python main.py --model gpt:gpt-4o --baseline zerocot --benchmark mgsm \
  --languages en zh ja

# BigBenchHard: specific task
python main.py --benchmark bigbenchhard --bigbenchhard_task boolean_expressions \
  --baseline zerocot --model qwen2.5:14b --num_runs 5

# Programming benchmarks
python main.py --model gpt:gpt-4o --baseline zerocot_single --benchmark humaneval
python main.py --model gpt:gpt-4o --baseline zerocot_single --benchmark mbpp
python main.py --model gpt:gpt-4o --baseline zerocot_single --benchmark apps
python main.py --model qwen2.5:14b --baseline zerocot_single --benchmark cruxeval

# ToT with DFS
python main.py --model qwen2.5:14b --baseline tot --benchmark gameof24 \
  --tot_algorithm dfs --tot_max_steps 4

# Setup GPU/Ollama (first time)
./setup_ollama_gpu.sh
docker exec -it ollama ollama run qwen2.5:14b
```

---

## Code Style & Conventions

### Python Style
- **Version**: Python 3.11+ (modern syntax with type hints)
- **Type Hints**: Full type annotations on all function signatures
- **Formatting**: 4-space indentation (PEP 8 compliant)
- **Docstrings**: Google-style with Args, Returns, and Example sections

### Registry Pattern Usage
- **New model**: Create `models/my_model.py` extending `BaseLLM`, register in `MODEL_REGISTRY` in `main.py`
- **New baseline**: Create `baseline/MyMethod/my_method.py` extending `BaseBaseline`, register in `BASELINE_REGISTRY` in `main.py`
- **New benchmark**: Create `benchmark/MyBench/mybench.py` extending `DatasetBase`, register in `DATASET_REGISTRY` in `benchmark/__init__.py` — no `main.py` changes needed

### Example: Adding a New Model

```python
# models/my_model.py
from models.base import BaseLLM, LLMResponse

class MyModelClient(BaseLLM):
    def __init__(self, api_key: str, model: str):
        super().__init__(api_key, model)

    def generate(self, prompt: str, temperature: float = 0) -> LLMResponse:
        return LLMResponse(content="...", model_name=self.model,
                           input_tokens=0, output_tokens=0)

# main.py MODEL_REGISTRY:
"mymodel": MyModelClient,
```

### Example: Adding a New Baseline

```python
# baseline/MyMethod/my_method.py
from baseline.basebaseline import BaseBaseline, BaselineResponse

class MyMethodBaseline(BaseBaseline):
    def __init__(self, llm: BaseLLM, param1: int = 10):
        super().__init__(llm, baseline_name="MyMethod")
        self.param1 = param1

    def run(self, question: str, **kwargs) -> BaselineResponse:
        return self.create_response(final_answer="...", reasoning_trace="...")

# main.py BASELINE_REGISTRY:
"mymethod": (MyMethodBaseline, lambda a: dict(param1=a.my_param)),
```

### Example: Adding a New Benchmark

```python
# benchmark/MyBench/mybench.py
from benchmark.datasetbase import DatasetBase, Problem, EvaluationResult

class MyBenchmark(DatasetBase):
    def load_dataset(self) -> None:
        self._data = [...]  # list or dict

    def get_problem(self, index: int) -> Problem:
        self._ensure_loaded()
        return Problem(index=index, question="...", ground_truth=..., metadata={})

    def evaluate_answer(self, prediction: str, ground_truth: Any) -> EvaluationResult:
        is_correct = ...
        return EvaluationResult(is_correct=is_correct, score=float(is_correct),
                                prediction=prediction, ground_truth=ground_truth, details={})

    def get_instruction(self) -> Optional[str]:
        return "..."

    def get_system_prompt(self) -> Optional[str]:
        return "..."

# benchmark/__init__.py DATASET_REGISTRY:
"mybench": (MyBenchmark, lambda _: {}),
```

---

## Development Notes

### Configuration Management
- API keys in environment variables (never hardcoded), exported before running
- `config.yaml`: centralized LLM endpoints and default model selections
- `env.yaml`: frozen dependencies for reproducibility

### Local Development Setup

```bash
conda env create -f env.yaml
conda activate Prompt
./setup_ollama_gpu.sh          # first time only
export API_KEY="ollama"
docker exec -it ollama ollama run qwen2.5:14b
python main.py --model qwen2.5:14b --baseline standard --benchmark mgsm
```

### Logging
- Level set to `ERROR` in `main.py` line 21 by default
- Set to `logging.DEBUG` for verbose development output

### Performance Considerations
1. **GPU Memory**: Monitor with `nvidia-smi` for large models
2. **API Rate Limiting**: Space out requests to cloud APIs
3. **RoT**: Stage 1+2 warmup is cached across questions; LLM calls within a stage are parallelized
4. **BoT**: `--no_update_buffer` skips buffer writes to speed up pure evaluation runs
5. **APPS**: Execution timeout applies per test case to prevent infinite loops

### Token Tracking & Metrics
- `LLMResponse`: tracks `input_tokens` / `output_tokens` per call
- `BaselineResponse`: aggregates totals across all LLM calls
- `Efficiency` metric: average time per question over M runs
- `Accuracy` metric: correct/incorrect evaluation per benchmark

### Git Workflow
- Main branch: `main`
- Commit message convention: `type(scope): description` (e.g. `feat(bot): ...`, `fix(rot): ...`)

---

## Quick Reference

### Files to Modify for Common Tasks
| Task | File(s) |
|------|---------|
| Add LLM provider | `models/new_provider.py`, `main.py` MODEL_REGISTRY |
| Add prompting method | `baseline/NewMethod/`, `main.py` BASELINE_REGISTRY |
| Add benchmark | `benchmark/NewBench/`, `benchmark/__init__.py` |
| Add CLI flag | `main.py` argument group functions |
| Update metrics | `utils/metrics.py`, `utils/get_mean_std.py` |

### Supported Models (MODEL_REGISTRY prefixes)
`gpt`, `deepseek`, `llama` / `llama3.1` / `llama3.3` / `llama2`, `gemini`, `qwen` / `qwen2` / `qwen2.5` / `qwen3`

All Llama variants map to `LlamaClient` (Ollama); all Qwen variants map to `QwenClient` (Ollama).
