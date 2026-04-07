"""
Buffer of Thoughts (BoT) Prompting Baseline.

This module implements the Buffer of Thoughts framework, a thought-augmented
reasoning approach that enhances LLM accuracy, efficiency and robustness by:

1. Maintaining a meta-buffer of reusable high-level thought-templates
2. Distilling task information via a problem-distiller
3. Retrieving the most relevant thought-template for each problem
4. Instantiating the template into a concrete reasoning structure
5. Updating the meta-buffer via a buffer-manager after each solve

Reference:
- Yang, L., Yu, Z., Zhang, T., Cao, S., Xu, M., Zhang, W., Gonzalez, J. E., & Cui, B. (2024).
  "Buffer of Thoughts: Thought-Augmented Reasoning with Large Language Models."
  38th Conference on Neural Information Processing Systems (NeurIPS 2024).
  https://github.com/YangLing0818/buffer-of-thought-llm

Author: (your name)
"""

from .bot import BoT

__all__ = ["BoT"]
