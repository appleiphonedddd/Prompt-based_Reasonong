"""
Reversal of Thought (RoT) Prompting Baseline.

This module provides an implementation of the Reversal of Thought framework,
which enhances LLM reasoning via Preference-Guided Reverse Reasoning (PGRR)
and pairwise preference evaluation.

The key idea: instead of giving the LLM a fixed prompt, RoT reverse-engineers
an optimal task definition from demonstrations, then selects the best one
using pairwise preference comparisons.

Reference:
- Yuan, J., Du, D., Zhang, H., Di, Z., & Naseem, U. (2025).
  "Reversal of Thought: Enhancing Large Language Models with
  Preference-Guided Reverse Reasoning Warm-up."
  Proceedings of ACL 2025 (Main), pp. 19442–19459.

Author: Egor Morozov
"""

from .rot import RoT

__all__ = ["RoT"]
