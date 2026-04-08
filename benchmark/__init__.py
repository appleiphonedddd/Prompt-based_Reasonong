"""
Benchmark Module.

Provides dataset wrappers for evaluating LLMs on reasoning tasks.

Available datasets:
    - GameOf24:  Arithmetic puzzle — combine 4 numbers to reach 24.
    - MGSM:      Multilingual Grade School Math word problems.

Usage::

    from benchmark import GameOf24, MGSM

    ds = GameOf24()
    ds.load_dataset()
    problem = ds.get_problem(0)

Author: Egor Morozov
"""

from benchmark.datasetbase import DatasetBase, Problem, EvaluationResult
from benchmark.GameOf24.gameof24 import GameOf24
from benchmark.MGSM.mgsm import MGSM

# Registry used by main.py for dynamic instantiation.
# Format: { key: (DatasetClass, kwargs_extractor) }
# kwargs_extractor receives the parsed argparse.Namespace and returns a dict
# of keyword arguments to pass to the dataset constructor.
# To add a new dataset: insert one entry here — no changes to main.py needed.
DATASET_REGISTRY: dict[str, tuple] = {
    "gameof24": (GameOf24, lambda _: {}),
    "mgsm":     (MGSM,     lambda a: {"language": a.mgsm_language}),
}

__all__ = [
    "DatasetBase",
    "Problem",
    "EvaluationResult",
    "GameOf24",
    "MGSM",
    "DATASET_REGISTRY",
]
