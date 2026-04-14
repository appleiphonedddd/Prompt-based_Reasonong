# Claude.md: Prompt-based Reasoning Project Guide

## Project Overview

**Prompt-based Reasoning** is a comprehensive framework for evaluating and benchmarking Large Language Models (LLMs) on complex reasoning tasks using various prompting strategies. The project enables systematic comparison of:

- **Multiple LLM Providers**: OpenAI (GPT), DeepSeek, Meta (Llama), Google (Gemini), Alibaba (Qwen), Mistral, and Google (Gemma)
- **Prompting Baselines**: Standard input, Zero-Shot CoT, Reflection-of-Thought (RoT), Tree-of-Thought (ToT), Buffer-of-Thought (BoT), and Graph-of-Thought (GoT)
- **Reasoning Benchmarks**: Game of 24, MGSM (Multilingual Grade School Math), Sonnet Writing, and Programming Puzzles

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
- **Anthropic SDK**: (check if used for Claude integration)
- **Requests/HTTPX**: 2.32.5 / 0.28.1 (HTTP client libraries)
- **YAML Configuration**: PyYAML 6.0.3

### Data & ML Libraries
- **Pydantic**: 2.12.5 (data validation and settings management)
- **Datasets**: 4.5.0 (HuggingFace benchmark datasets)
- **Huggingface Hub**: 1.3.4
- **PyArrow**: 23.0.0 (columnar data format)

### Environment & Deployment
- **Conda**: Environment management (virtual environment in `/home/infor/miniconda3/envs/Prompt`)
- **Ollama**: Local LLM inference engine for GPU-accelerated model serving
- **NVIDIA Container Toolkit**: For GPU support (if using Docker containers)

### Development Tools
- **Typer**: 0.21.1 (CLI framework)
- **Click**: 8.3.1 (command-line interface creation)
- **Logging**: Python's built-in logging module

---

## Directory Structure & Architecture

```
Prompt-based-Reasoning/
├── main.py                    # Entry point with Model/Baseline registries
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
│   ├── qwen.py              # Alibaba Qwen (via Ollama) implementation
│   ├── ministral.py         # Mistral Ministral implementation
│   └── gemma.py             # Google Gemma implementation
│
├── baseline/                # Prompting Strategy Implementations
│   ├── basebaseline.py      # Abstract BaseBaseline class & BaselineResponse
│   ├── Standard/
│   │   └── io.py            # Standard input-output baseline
│   ├── CoT/                 # Chain-of-Thought
│   │   ├── __init__.py
│   │   └── cot.py
│   ├── RoT/                 # Reflection-of-Thought
│   │   ├── __init__.py
│   │   └── rot.py
│   ├── ToT/                 # Tree-of-Thought
│   │   ├── __init__.py
│   │   └── tot.py
│   ├── BoT/                 # Buffer-of-Thought
│   │   ├── __init__.py
│   │   └── bot.py
│   └── GoT/                 # Graph-of-Thought
│       ├── __init__.py
│       └── got.py
│
├── benchmark/               # Reasoning Task Datasets
│   ├── __init__.py          # DATASET_REGISTRY
│   ├── GameOf24/            # Game of 24 benchmark
│   ├── MGSM/                # Multilingual Grade School Math
│   ├── SonnetWriting/       # Shakespearean sonnet generation
│   └── ProgrammingPuzzles/  # Programming challenge dataset
│
├── utils/                   # Utility Modules
│   ├── config.py            # Configuration loading & validation
│   ├── metrics.py           # Accuracy & Efficiency metric classes
│   └── get_mean_std.py      # Statistical analysis (mean, std dev)
│
├── tests/                   # Test Suite
│   └── (test files)
│
├── .vscode/                 # VSCode configuration
└── .git/                    # Git repository metadata
```

### Key Architectural Patterns

1. **Registry Pattern** (`main.py`)
   - `MODEL_REGISTRY`: Maps model names to LLM client classes
   - `BASELINE_REGISTRY`: Maps baseline names to strategy classes + argument extractors
   - `DATASET_REGISTRY`: Maps benchmark names to dataset implementations

2. **Abstract Base Classes**
   - `BaseLLM` in `models/base.py`: Enforces `generate()` method interface
   - `BaseBaseline` in `baseline/basebaseline.py`: Enforces `run()` method interface

3. **Dataclass-based Response Objects**
   - `LLMResponse`: Standardized LLM generation output
   - `BaselineResponse`: Unified benchmark evaluation result

---

## Code Style & Conventions

### Python Style
- **Version**: Python 3.11+ (modern syntax with type hints)
- **Type Hints**: Full type annotations on all function signatures
- **Formatting**: 4-space indentation (PEP 8 compliant)
- **Docstrings**: Google-style docstrings with Args, Returns, and Example sections

### Code Organization Principles

1. **Module Structure**
   - One responsibility per module
   - Clear separation: models vs. baselines vs. benchmarks
   - Base classes define interfaces, subclasses implement strategies

2. **Class Design**
   - Use abstract base classes (`ABC`) for plugin architectures
   - Dataclasses for immutable response objects
   - Concrete implementations inherit from `BaseLLM` or `BaseBaseline`

3. **Registry Pattern Usage**
   - Add new models: Register in `MODEL_REGISTRY` in `main.py`
   - Add new baselines: Register in `BASELINE_REGISTRY` with argument extractor lambda
   - Add new benchmarks: Register in `DATASET_REGISTRY` in `benchmark/__init__.py`

### Example: Adding a New Model

```python
# 1. Create models/my_model.py
from models.base import BaseLLM, LLMResponse

class MyModelClient(BaseLLM):
    def __init__(self, api_key: str, model: str):
        super().__init__(api_key, model)
    
    def generate(self, prompt: str, temperature: float = 0) -> LLMResponse:
        # Implementation here
        return LLMResponse(
            content="...",
            model_name=self.model,
            input_tokens=0,
            output_tokens=0,
        )

# 2. Register in main.py
MODEL_REGISTRY = {
    # ... existing entries ...
    "mymodel": MyModelClient,
}
```

### Example: Adding a New Baseline

```python
# 1. Create baseline/MyMethod/my_method.py
from baseline.basebaseline import BaseBaseline, BaselineResponse

class MyMethodBaseline(BaseBaseline):
    def __init__(self, llm: BaseLLM, param1: int = 10):
        super().__init__(llm, baseline_name="MyMethod")
        self.param1 = param1
    
    def run(self, question: str, **kwargs) -> BaselineResponse:
        # Implementation here
        return self.create_response(final_answer="...", reasoning_trace="...")

# 2. Register in main.py
BASELINE_REGISTRY = {
    # ... existing entries ...
    "mymethod": (MyMethodBaseline, lambda a: dict(param1=a.my_param)),
}
```

---

## Development Notes & Best Practices

### Configuration Management

1. **API Keys & Credentials**
   - Store API keys in environment variables (avoid hardcoding)
   - Use `.env` files (not committed to Git)
   - Export before running: `export API_KEY="your_key"`

2. **Model Configuration** (`config.yaml`)
   - LLM endpoints are centralized
   - Base URLs for local (vLLM), Gemini, and other providers
   - Default model selections per provider

3. **Environment Setup** (`env.yaml`)
   - All dependencies frozen for reproducibility
   - CUDA/GPU libraries pre-specified
   - Activate with: `conda activate Prompt`

### Local Development Setup

```bash
# 1. Create Conda environment
conda env create -f env.yaml
conda activate Prompt

# 2. Setup GPU/Ollama environment (first time only)
./setup_ollama_gpu.sh

# 3. Set API keys (optional, only if using cloud models)
export API_KEY="your_openai_key"
export GEMINI_API_KEY="your_gemini_key"

# 4. Pull and run local model via Ollama
ollama pull qwen:7b
ollama serve

# 5. Run evaluation (no API key needed for local models)
python main.py --model "qwen:7b" --baseline standard --benchmark mgsm
```

### Token Tracking & Metrics

- **`LLMResponse`**: Tracks `input_tokens` and `output_tokens` per call
- **`BaselineResponse`**: Aggregates totals across all LLM calls in a baseline run
- **Efficiency Metrics**: Token counts used for cost and speed analysis
- **Accuracy Metrics**: Correct/incorrect evaluation per benchmark

### Error Handling & Validation

1. **Model/Baseline Validation** (in `Evaluator.validate()`)
   - Checks if model/baseline names exist in registries
   - Raises `ValueError` with helpful messages listing valid options

2. **API Error Handling**
   - Each model client should handle API timeouts, auth failures gracefully
   - Consider retry logic for transient failures

3. **Dataset Validation**
   - Ensure benchmark data is loaded correctly
   - Handle missing or corrupted dataset files

### Extending with New Benchmarks

The SonnetWriting benchmark provides a complete example. Follow this pattern:

```python
# 1. Create benchmark/MyBench/mybench.py
from benchmark.datasetbase import DatasetBase, Problem, EvaluationResult

class MyBenchmark(DatasetBase):
    def load_dataset(self) -> None:
        # Load data (JSON, TSV, or HuggingFace)
        # Set self._data to a list/dict
        pass
    
    def get_problem(self, index: int) -> Problem:
        self._ensure_loaded()
        return Problem(
            index=index,
            question="...",
            ground_truth=...,
            metadata={...}
        )
    
    def evaluate_answer(self, prediction: str, ground_truth: Any) -> EvaluationResult:
        # Implement your scoring logic
        score = ...  # float in [0, 1]
        is_correct = ...  # bool
        return EvaluationResult(
            is_correct=is_correct,
            score=score,
            prediction=prediction,
            ground_truth=ground_truth,
            details={...}
        )
    
    def get_instruction(self) -> Optional[str]:
        # Optional: task-specific guidance
        return "..."
    
    def get_system_prompt(self) -> Optional[str]:
        # Optional: system-level persona
        return "..."

# 2. Register in benchmark/__init__.py
from benchmark.MyBench.mybench import MyBenchmark
DATASET_REGISTRY["mybench"] = (MyBenchmark, lambda _: {})
# Add to __all__: "MyBenchmark"
```

**Real example**: See `benchmark/SonnetWriting/` for a complete implementation evaluating Shakespearean sonnets on three criteria (word inclusion, structure, rhyme scheme).

### Logging & Debugging

- Logging level set to `ERROR` by default in `main.py` (see line 21)
- Set to `logging.DEBUG` for development
- Each model client should log API calls and errors

### Testing

- Test files located in `tests/` directory
- Run tests before committing
- Write tests for new model clients and baseline methods

### Performance Considerations

1. **GPU Memory**: Monitor with `nvidia-smi` when running large models
2. **API Rate Limiting**: Space out requests to cloud APIs
3. **Token Costs**: Track token usage for expensive models (GPT-4, etc.)
4. **Caching**: Consider caching LLM responses for development iterations

### Git Workflow

- Main branch: `main` (current branch)
- Recent commits show additions of Gemma, Ministral models and MGSM fixes
- Commit messages follow pattern: `Add X model` or `Fix: Y issue`

---

## CLI Interface

The project uses a command-line interface (likely via Typer/Click):

```bash
python main.py --model <model_name> --baseline <baseline_name> --benchmark <benchmark_name> [options]
```

**Common Options** (inferred from `main.py`):
- `--model`: Model to evaluate (e.g., `gpt`, `llama3:8b`, `gemini`)
- `--baseline`: Prompting strategy (e.g., `standard`, `zerocot`, `tot`)
- `--benchmark`: Task dataset (e.g., `mgsm`, `game_of_24`)
- Additional baseline-specific flags for advanced methods

---

## Quick Reference for Future Work

### When Adding New Features
1. **New Model?** → Extend `BaseLLM`, register in `MODEL_REGISTRY`
2. **New Baseline?** → Extend `BaseBaseline`, register in `BASELINE_REGISTRY`
3. **New Dataset?** → Register in `DATASET_REGISTRY`
4. **New Metric?** → Add to `utils/metrics.py`

### Files to Modify for Common Tasks
| Task | File(s) | Example |
|------|---------|---------|
| Add LLM provider | `models/new_provider.py`, `main.py` | See: gemini.py, llama.py |
| Add prompting method | `baseline/NewMethod/`, `main.py` | See: BoT/, ToT/ |
| Add benchmark | `benchmark/NewBench/`, `benchmark/__init__.py` | See: SonnetWriting/ |
| Modify config | `config.yaml`, `env.yaml` | API endpoints, model settings |
| Update metrics | `utils/metrics.py`, `utils/get_mean_std.py` | Accuracy, efficiency stats |

---

## Recent Additions

### SonnetWriting Benchmark (Latest)
A generative creative task evaluating Shakespearean sonnet generation. Models are evaluated on:
- **Word inclusion** (1/3): All 3 provided words must appear verbatim
- **Structure** (1/3): Exactly 14 lines required
- **Rhyme scheme** (1/3): ABAB CDCD EFEF GG pattern (via suffix matching)

Dataset: 20 unique prompts. Score = average of three criteria. Model is "correct" only if all criteria fully satisfied.

Reference: BoT paper (Yang et al., NeurIPS 2024), meta-prompting task (Suzgun & Kalai, 2024)

---

## Summary

This is a well-structured Python research framework following solid software engineering practices:
- **Extensibility**: Registry patterns make adding new models/baselines/benchmarks simple (see SonnetWriting for reference)
- **Testability**: Abstract interfaces enable unit testing
- **Maintainability**: Clear module separation and type hints
- **Reproducibility**: Environment specs frozen in `env.yaml`

The codebase is ready for scalable LLM evaluation research and benchmarking.
