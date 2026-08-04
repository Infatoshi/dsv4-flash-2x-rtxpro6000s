"""Batch-1 decode throughput bench for DSV4 on localhost:8000.

M = completion_tokens / elapsed (tok/s), 1 warmup + N reps, temperature 0.
Also captures the completion text for the correctness (C) check and a hw stamp.
"""

import json
import statistics
import subprocess
import sys
import time
import urllib.request

N_REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
TAG = sys.argv[2] if len(sys.argv) > 2 else "run"
URL = "http://localhost:8000/v1/chat/completions"
PROMPT = "Explain, step by step, how a paged KV cache works in an LLM inference server."
BODY = {
    "model": "dsv4-flash",
    "messages": [{"role": "user", "content": PROMPT}],
    "max_tokens": 256,
    "temperature": 0,
    "ignore_eos": True,
}


def one():
    req = urllib.request.Request(
        URL, json.dumps(BODY).encode(), {"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    dt = time.perf_counter() - t0
    toks = d["usage"]["completion_tokens"]
    return toks / dt, dt, d["choices"][0]["message"]["content"]


one()  # warmup
rates, text = [], None
for i in range(N_REPS):
    r, dt, text = one()
    rates.append(r)
    print(f"rep {i}: {r:.2f} tok/s ({dt:.2f}s)")

mu = statistics.mean(rates)
sd = statistics.stdev(rates) if len(rates) > 1 else 0.0
stamp = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,clocks.sm,clocks.mem,temperature.gpu",
     "--format=csv,noheader"],
    capture_output=True, text=True).stdout.strip().replace("\n", " | ")
row = {"tag": TAG, "mean": mu, "std": sd, "reps": rates, "hw": stamp,
       "ts": time.strftime("%F %T")}
print(f"MEAN {mu:.2f} tok/s  STD {sd:.3f}")
with open("scoreboard.jsonl", "a") as f:
    f.write(json.dumps(row) + "\n")
with open(f"output_{TAG}.txt", "w") as f:
    f.write(text)
