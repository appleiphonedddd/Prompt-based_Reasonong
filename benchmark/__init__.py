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
from benchmark.GameOf24 import GameOf24
from benchmark.MGSM import MGSM

# Registry used by main.py for dynamic instantiation
DATASET_REGISTRY: dict[str, type[DatasetBase]] = {
    "gameof24": GameOf24,
    "mgsm":     MGSM,
}

__all__ = [
    "DatasetBase",
    "Problem",
    "EvaluationResult",
    "GameOf24",
    "MGSM",
    "DATASET_REGISTRY",
]
