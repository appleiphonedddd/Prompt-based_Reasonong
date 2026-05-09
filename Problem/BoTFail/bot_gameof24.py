#!/usr/bin/env python3
"""
BoT accuracy experiment on Game of 24.

Usage:
    python Problem/BoTFail/bot_gameof24.py
    python Problem/BoTFail/bot_gameof24.py --model qwen2.5:3b --limit 20
    python Problem/BoTFail/bot_gameof24.py --no-update-buffer --output results.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from models.qwen import QwenClient
from models.llama import LlamaClient
from baseline.BoT import BoT

JSONL_PATH = Path(__file__).parent / "gameof24.jsonl"

INSTRUCTION = (
    "Using all four numbers exactly once and the basic arithmetic operations "
    "(+, -, *, /), make 24. "
    "State your final arithmetic expression clearly, e.g. (3 + 5) * (8 / 2)."
)

SYSTEM_PROMPT = "You are an expert puzzle solver specializing in the Game of 24."


def load_problems(path: Path, limit: int | None = None) -> list[dict]:
    problems = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    return problems[:limit] if limit else problems


def safe_eval(expr: str) -> float | None:
    expr = expr.strip()
    # strip trailing '= 24' annotation
    expr = re.sub(r"\s*=\s*24\s*$", "", expr).strip()
    if not expr:
        return None
    if re.search(r"[^\d\s\.\+\-\*\/\(\)]", expr):
        return None
    try:
        return float(eval(expr, {"__builtins__": {}}, {}))
    except Exception:
        return None


def uses_correct_numbers(expr: str, expected: list[float]) -> bool:
    found = sorted(float(n) for n in re.findall(r"\d+(?:\.\d+)?", expr))
    return found == sorted(expected)


def check_answer(answer: str, numbers: list[float]) -> tuple[bool, str]:
    """Return (correct, expression) where expression is the found expr or empty string."""
    candidates: list[str] = [answer]
    for line in answer.split("\n"):
        candidates.append(line)
        candidates.extend(re.split(r"[:=]", line))

    for cand in candidates:
        val = safe_eval(cand)
        if val is not None and abs(val - 24) < 1e-6:
            expr = cand.strip()
            if uses_correct_numbers(expr, numbers):
                return True, expr
    return False, ""


def build_llm(model_name: str):
    if "qwen" in model_name.lower():
        return QwenClient(model=model_name)
    if "llama" in model_name.lower():
        return LlamaClient(model=model_name)
    raise ValueError(f"Unknown model prefix: {model_name!r}. Use a qwen or llama model name.")


def main():
    parser = argparse.ArgumentParser(description="BoT Game of 24 accuracy experiment")
    parser.add_argument("--model", default="qwen2.5:32b", help="Model name (default: qwen2.5:32b)")
    parser.add_argument("--limit", type=int, default=None, help="Max problems to evaluate")
    parser.add_argument("--no-update-buffer", action="store_true", help="Freeze the meta-buffer")
    parser.add_argument("--threshold", type=float, default=0.6, help="Similarity threshold (default: 0.6)")
    parser.add_argument("--output", default=None, help="Save results to this JSON file")
    args = parser.parse_args()

    problems = load_problems(JSONL_PATH, args.limit)
    print(f"Loaded {len(problems)} problems")

    llm = build_llm(args.model)
    bot = BoT(
        llm=llm,
        similarity_threshold=args.threshold,
        update_buffer=not args.no_update_buffer,
    )
    print(f"BoT | model={args.model} | buffer_size={bot.meta_buffer.size} | update={not args.no_update_buffer}")
    print("-" * 72)

    results = []
    correct = 0

    for i, prob in enumerate(problems):
        numbers = [float(n) for n in prob["input"].split()]
        try:
            response = bot.run(
                question=prob["input"],
                system_prompt=SYSTEM_PROMPT,
                instruction=INSTRUCTION,
            )
            answer = response.final_answer
            ok, expr = check_answer(answer, numbers)
        except Exception as e:
            answer = f"ERROR: {e}"
            ok, expr = False, ""

        correct += int(ok)
        mark = "✓" if ok else "✗"
        display = f"{expr} = 24" if expr else f"(no valid expression) raw: {answer[:60]!r}"
        print(f"[{i+1:3d}/{len(problems)}] {mark} | {prob['input']} | {display}")

        results.append({
            "input": prob["input"],
            "answer": answer,
            "expression": expr,
            "correct": ok,
        })

    accuracy = correct / len(problems) * 100
    print("-" * 72)
    print(f"Accuracy: {correct}/{len(problems)} = {accuracy:.1f}%")

    if args.output:
        out = {
            "model": args.model,
            "accuracy": accuracy,
            "correct": correct,
            "total": len(problems),
            "results": results,
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
