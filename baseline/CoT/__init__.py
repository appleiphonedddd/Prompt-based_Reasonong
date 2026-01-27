"""
Chain-of-Thought (CoT) Prompting Baselines.

This module provides implementations for Chain-of-Thought prompting techniques:
- ZeroShotCoT: Elicit reasoning with "Let's think step by step"
- FewShotCoT: In-context learning with reasoning demonstrations (TODO)

References:
- Wei et al., "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" (2022)
- Kojima et al., "Large Language Models are Zero-Shot Reasoners" (2022)

Author: Egor Morozov
"""

from .zero_shot_cot import ZeroShotCoT, ZeroShotCoTSinglePass

__all__ = ["ZeroShotCoT", "ZeroShotCoTSinglePass"]