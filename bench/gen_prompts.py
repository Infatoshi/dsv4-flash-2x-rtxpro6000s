"""Build prompt sets for the DSV4 concurrency/category sweep.

Outputs opt/sweep_prompts.json:
  grid: {"512": [16 prompts], "2048": [...], "8192": [...]}  (ultrachat-derived,
        distinct per stream so prefix caching cannot share)
  categories: {"chat": [8], "math": [8], "code": [8], "baseline": [8]}

Token lengths measured with the checkpoint tokenizer.
"""

import os
import json
import random

from transformers import AutoTokenizer

MODEL_DIR = os.environ.get("MODEL", os.path.expanduser("~/kernels/DeepSeek-V4-Flash-0731-NVFP4"))
OUT = MODEL_DIR + "/opt/sweep_prompts.json"
random.seed(0)

tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)


def ntok(s):
    return len(tok.encode(s))


def truncate_to_tokens(s, target):
    ids = tok.encode(s)[:target]
    return tok.decode(ids, skip_special_tokens=True)


# ---- ultrachat for chat + grid material ----
from datasets import load_dataset

uc = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
chat_prompts = []
long_material = []
for i, row in enumerate(uc):
    if i >= 4000:
        break
    msgs = row["messages"]
    user0 = next((m["content"] for m in msgs if m["role"] == "user"), None)
    if user0 is None:
        continue
    if len(chat_prompts) < 64 and 200 < len(user0) < 2500:
        chat_prompts.append(user0)
    # full conversation text as long-context material
    convo = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in msgs)
    if len(convo) > 8000:
        long_material.append(convo)

# ---- gsm8k for math ----
gsm = load_dataset("openai/gsm8k", "main", split="test")
math_prompts = [
    row["question"] + "\nSolve step by step and give the final numeric answer."
    for row in gsm.select(range(64))
]

# ---- Elliot's traces for code ----
code_prompts = []
kw = ("def ", "class ", "python", "function", "bug", "compile", "cuda",
      "kernel", "rust", "script", "implement", "error", "traceback")
with open(
    os.environ.get("SHAREGPT", os.path.expanduser("~/.cache/huggingface/datasets/sharegpt.jsonl"))
) as f:
    for line in f:
        row = json.loads(line)
        human = next(
            (c["value"] for c in row["conversations"] if c["from"] == "human"), ""
        )
        if not (300 < len(human) < 3000):
            continue
        if "[Image" in human or human.count("\n") > 60:
            continue
        low = human.lower()
        if sum(k in low for k in kw) >= 2:
            code_prompts.append(human)
        if len(code_prompts) >= 64:
            break

baseline_prompt = (
    "Explain, step by step, how a paged KV cache works in an LLM inference server."
)

# ---- grid prompts at target input lengths ----
grid = {}
for target in (512, 2048, 8192):
    plist = []
    for i in range(16):
        if target <= 512:
            base = chat_prompts[i % len(chat_prompts)]
            text = f"[stream {i}] " + base
            if ntok(text) > target - 8:
                text = truncate_to_tokens(text, target - 8)
            else:
                # extend with material to approach the target
                extra = long_material[i % len(long_material)]
                text = (
                    f"[stream {i}] Context:\n"
                    + truncate_to_tokens(extra, target - ntok(base) - 40)
                    + "\n\nQuestion: "
                    + base
                )
        else:
            head = f"[stream {i}] Read the following transcripts carefully.\n\n"
            tail = (
                "\n\nNow summarize the main technical points discussed above and "
                "list any action items, step by step."
            )
            budget = target - ntok(head) - ntok(tail) - 8
            parts, j, used = [], i * 7 + 1, 0
            while used < budget + 2000:
                m = long_material[j % len(long_material)]
                parts.append(m)
                used += ntok(m)
                j += 1
            text = head + truncate_to_tokens("\n\n---\n\n".join(parts), budget) + tail
        plist.append(text)
    grid[str(target)] = plist
    print("grid", target, "->", [ntok(p) for p in plist[:4]], "...")

cats = {
    "chat": chat_prompts[:8],
    "math": math_prompts[:8],
    "code": code_prompts[:8],
    "baseline": [baseline_prompt] * 8,
}
for k, v in cats.items():
    print("cat", k, len(v), "tok:", [ntok(p) for p in v])

json.dump({"grid": grid, "categories": cats}, open(OUT, "w"))
print("wrote", OUT)
