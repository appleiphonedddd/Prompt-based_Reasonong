"""
Reversal of Thought (RoT) Prompting Baseline.

This module provides an implementation of the Reversal of Thought framework,
which enhances LLM reasoning via Preference-Guided Reverse Reasoning (PGRR),
Cognitive Preference Manager (CPM), and pairwise preference evaluation.

Reference:
- Yuan, J., Du, D., Zhang, H., Di, Z., & Naseem, U. (2025).
  "Reversal of Thought: Enhancing Large Language Models with
  Preference-Guided Reverse Reasoning Warm-up."
  Proceedings of ACL 2025 (Main), pp. 19442-19459.

Author: Egor Morozov
"""

from .rot import (
    RoT,
    BaseEmbeddingModel,
    SentenceTransformerEmbedding,
    LLMBasedSimilarity,
)

__all__ = [
    "RoT",
    "BaseEmbeddingModel",
    "SentenceTransformerEmbedding",
    "LLMBasedSimilarity",
]