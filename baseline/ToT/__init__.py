"""
Tree of Thoughts (ToT) Prompting Baseline.

This module provides an implementation of the Tree of Thoughts framework,
which enables deliberate problem solving by maintaining a tree of intermediate
reasoning steps ("thoughts") and searching over them with BFS or DFS.

The key idea: instead of generating a single left-to-right reasoning chain,
ToT explores multiple candidate thoughts at each step, evaluates their
promise via LLM self-assessment, and uses tree search (BFS/DFS) with
lookahead and backtracking to find the best solution path.

Reference:
- Yao, S., Yu, D., Zhao, J., Shafran, I., Griffiths, T. L., Cao, Y., &
  Narasimhan, K. (2023). "Tree of Thoughts: Deliberate Problem Solving
  with Large Language Models." NeurIPS 2023.
  https://arxiv.org/abs/2305.10601

Author: Egor Morozov
"""

from .tot import ToT

__all__ = ["ToT"]