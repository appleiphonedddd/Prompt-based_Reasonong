"""
Inference Stability (Stab) — WRONG answer case.

Corrected Stab computation (per LMS3 paper Eq. 22):
  1. Generate the answer with greedy decoding.
  2. Feed the COMPLETE (prompt + Q + A) through the model in one forward pass.
  3. At each layer, collect hidden states for the answer token positions only.
  4. Stab per layer = mean over answer tokens of  ‖WV · h / √d‖.

Expected result: Stab should be HIGHER for problems the model gets wrong
(higher gradient norm = less stable prediction).
"""

import os, sys, json
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from transformers import AutoTokenizer, AutoModelForCausalLM
from benchmark.GameOf24.gameof24 import GameOf24

MODEL_ID       = "Qwen/Qwen2.5-3B-Instruct"
MAX_NEW_TOKENS = 80
OUT_DIR        = os.path.dirname(os.path.abspath(__file__))


# ── Helpers ───────────────────────────────────────────────────────────────────

def greedy_generate(model, tokenizer, enc, device):
    """Simple greedy generation; returns decoded response string."""
    ids = enc["input_ids"].clone()
    mask = enc["attention_mask"].clone()
    generated = []
    past_kv = None

    with torch.no_grad():
        for step in range(MAX_NEW_TOKENS):
            out = model(
                input_ids=ids if step == 0 else ids[:, -1:],
                attention_mask=mask,
                past_key_values=past_kv,
                use_cache=True,
            )
            past_kv = out.past_key_values
            next_id = out.logits[0, -1, :].argmax().item()
            generated.append(next_id)
            ids  = torch.tensor([[next_id]], device=device)
            mask = torch.cat([mask, torch.ones((1, 1), device=device, dtype=mask.dtype)], dim=-1)
            if next_id == tokenizer.eos_token_id:
                break

    return tokenizer.decode(generated, skip_special_tokens=True)


def compute_stab_per_layer(model, full_enc, answer_start: int, answer_end: int, device):
    """
    One forward pass over (prompt + Q + A).
    For each layer: Stab = mean_over_answer_tokens( ‖WV · h / √d‖ ).
    """
    num_layers = len(model.model.layers)
    stab_per_layer = [None] * num_layers

    def make_hook(layer_idx):
        def hook(module, inp, _out):
            # inp[0]: (1, full_seq, hidden)
            h_ans = inp[0][0, answer_start:answer_end, :].detach().float()  # (ans_len, d)
            wv    = module.weight.detach().float()                           # (v_dim, d)
            d     = h_ans.shape[-1]
            if h_ans.shape[0] == 0:
                stab_per_layer[layer_idx] = 0.0
                return
            stabs = [(wv @ h / d ** 0.5).norm().item() for h in h_ans]
            stab_per_layer[layer_idx] = float(np.mean(stabs))
        return hook

    hooks = [
        model.model.layers[i].self_attn.v_proj.register_forward_hook(make_hook(i))
        for i in range(num_layers)
    ]
    with torch.no_grad():
        model(**full_enc)
    for h in hooks:
        h.remove()

    return stab_per_layer


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    print(f"Loading {MODEL_ID} …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, device_map="auto"
    )
    model.eval()
    device = next(model.parameters()).device

    print("Loading Game of 24 dataset …")
    ds = GameOf24()
    ds.load_dataset()
    problem = ds.get_problem(0)
    print(f"Problem 0  →  numbers: {problem.question}")

    sys_prompt  = ds.get_system_prompt()
    user_msg    = f"{ds.get_instruction()}\n\nNumbers: {problem.question}"
    messages    = [{"role": "system", "content": sys_prompt},
                   {"role": "user",   "content": user_msg}]

    # ── Step 1: Generate answer ───────────────────────────────────────────────
    enc = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True,
        tokenize=True, return_dict=True, return_tensors="pt",
    ).to(device)
    prompt_len = enc["input_ids"].shape[-1]

    response = greedy_generate(model, tokenizer, enc, device)
    result   = ds.evaluate_answer(response, problem.ground_truth)
    print(f"Response : {response!r}")
    print(f"Answer   : {'CORRECT ✓' if result.is_correct else 'WRONG ✗'}  "
          f"(evaluated to {result.details.get('evaluated_result', '?')})")

    # ── Step 2: Encode complete (prompt + Q + A) ──────────────────────────────
    full_enc = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": response}],
        add_generation_prompt=False,
        tokenize=True, return_dict=True, return_tensors="pt",
    ).to(device)
    full_len     = full_enc["input_ids"].shape[-1]
    answer_start = prompt_len          # answer tokens start here
    answer_end   = full_len            # include trailing <|im_end|>

    answer_preview = tokenizer.decode(
        full_enc["input_ids"][0, answer_start:answer_end]
    )
    print(f"Answer tokens ({answer_end - answer_start} toks): {answer_preview!r}")

    # ── Step 3: Compute Stab per layer ────────────────────────────────────────
    print(f"\nComputing Stab across {len(model.model.layers)} layers …")
    stab_per_layer = compute_stab_per_layer(
        model, full_enc, answer_start, answer_end, device
    )
    print(f"Stab range: [{min(stab_per_layer):.4f}, {max(stab_per_layer):.4f}]")

    # ── Save ──────────────────────────────────────────────────────────────────
    data = {
        "problem":       problem.question,
        "ground_truth":  problem.ground_truth,
        "response":      response,
        "is_correct":    result.is_correct,
        "stab_per_layer": stab_per_layer,
    }
    json_path = os.path.join(OUT_DIR, "stab_data_wrong.json")
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved → {json_path}")

    visualise(stab_per_layer, result.is_correct, problem.question, response)


# ── Visualisation ─────────────────────────────────────────────────────────────

def visualise(stab_per_layer, is_correct, problem_str, response):
    layers = np.arange(len(stab_per_layer))
    color  = "red" if not is_correct else "green"
    verdict = "WRONG ✗" if not is_correct else "CORRECT ✓"

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(layers, stab_per_layer, color=color, alpha=0.7, width=0.8)
    ax.plot(layers, stab_per_layer, "o-", color=color, markersize=3, linewidth=1)
    ax.set_xlabel("Transformer layer", fontsize=11)
    ax.set_ylabel("Stab(X)  =  mean‖WV · h / √d‖  over answer tokens", fontsize=10)
    ax.set_title(
        f"Inference Stability per Layer  ·  Game of 24: {problem_str}  ·  {verdict}\n"
        f"Response: {response!r}",
        fontsize=11, fontweight="bold", color=color,
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_xlim(-0.5, len(stab_per_layer) - 0.5)

    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, "stab_visualization_wrong.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved visualisation → {out_png}")
    plt.close()


if __name__ == "__main__":
    run()
