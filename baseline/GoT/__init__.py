"""
Graph of Thoughts (GoT) Baseline Package.

This package implements the Graph of Thoughts prompting framework,
which models LLM reasoning as an arbitrary directed graph where
thoughts are vertices and dependencies are edges.

Reference:
- Besta et al., "Graph of Thoughts: Solving Elaborate Problems with
  Large Language Models" (AAAI 2024).
"""

from baseline.GoT.got import GoT

__all__ = ["GoT"]