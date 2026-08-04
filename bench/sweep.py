"""Concurrency x seqlen x category sweep against the local DSV4 server.

usage: sweep.py <mode: grid|cats> <server_tag> [conc_list_csv]
Appends rows to opt/sweep_results.jsonl.

grid: for each input-length bucket and concurrency level, fire C concurrent
requests (distinct prompts), 256 output tokens, temp 0, ignore_eos.
  aggregate_tps = total completion tokens / wall time
  per_stream_tps = mean over streams of (tokens_i / duration_i)
cats: per category, run 8 prompts at concurrency 1, then a c=4 batch.
Spec-decode acceptance is diffed from /metrics before/after each cell.
"""

import concurrent.futures as futures
import os
import json
import re
import subprocess
import sys
import time
import urllib.request

URL = "http://localhost:8000/v1/chat/completions"
METRICS = "http://localhost:8000/metrics"
MODEL = os.environ.get("MODEL", os.path.expanduser("~/kernels/DeepSeek-V4-Flash-0731-NVFP4"))
OUT = MODEL + "/opt/sweep_results.jsonl"
PROMPTS = json.load(open(MODEL + "/opt/sweep_prompts.json"))

MODE = sys.argv[1]
TAG = sys.argv[2]
CONCS = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [1, 2, 4, 8, 16]
LENS = set(sys.argv[4].split(",")) if len(sys.argv) > 4 else None


def one_request(prompt, max_tokens=256):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": True,
    }
    req = urllib.request.Request(
        URL, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1800) as r:
        resp = json.load(r)
    dt = time.perf_counter() - t0
    u = resp["usage"]
    return {
        "prompt_tokens": u["prompt_tokens"],
        "completion_tokens": u["completion_tokens"],
        "elapsed_s": dt,
    }


def spec_counters():
    try:
        with urllib.request.urlopen(METRICS, timeout=10) as r:
            text = r.read().decode()
    except Exception:
        return None
    out = {}
    for key, pat in (
        ("accepted", r"spec_decode_num_accepted_tokens_total"),
        ("drafted", r"spec_decode_num_draft_tokens_total"),
    ):
        vals = re.findall(pat + r'(?:\{[^}]*\})?\s+([0-9.e+]+)', text)
        if vals:
            out[key] = sum(float(v) for v in vals)
    return out or None


def run_cell(prompts, conc, max_tokens=256):
    prompts = prompts[:conc]
    assert len(prompts) == conc
    s0 = spec_counters()
    wall0 = time.perf_counter()
    with futures.ThreadPoolExecutor(max_workers=conc) as ex:
        results = list(ex.map(lambda p: one_request(p, max_tokens), prompts))
    wall = time.perf_counter() - wall0
    s1 = spec_counters()
    total_tokens = sum(r["completion_tokens"] for r in results)
    agg = total_tokens / wall
    per_stream = sum(
        r["completion_tokens"] / r["elapsed_s"] for r in results
    ) / len(results)
    row = {
        "tag": TAG,
        "conc": conc,
        "n_prompts": len(prompts),
        "prompt_tokens_mean": sum(r["prompt_tokens"] for r in results) / len(results),
        "completion_tokens": total_tokens,
        "wall_s": round(wall, 3),
        "aggregate_tps": round(agg, 2),
        "per_stream_tps": round(per_stream, 2),
    }
    if s0 and s1 and "accepted" in s0 and "drafted" in s1:
        dd = s1["drafted"] - s0["drafted"]
        da = s1["accepted"] - s0["accepted"]
        if dd > 0:
            row["accept_rate"] = round(da / dd, 4)
            row["accept_len"] = round(1 + da / max(1e-9, dd / 3), 3)
    return row


def emit(row):
    row["hw"] = subprocess.run(
        ["nvidia-smi", "--query-gpu=clocks.sm,temperature.gpu",
         "--format=csv,noheader"], capture_output=True, text=True
    ).stdout.strip().replace("\n", " | ")
    row["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(OUT, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps({k: v for k, v in row.items() if k != "hw"}))


if MODE == "grid":
    # warmup
    one_request(PROMPTS["grid"]["512"][0], 32)
    for length, plist in sorted(PROMPTS["grid"].items(), key=lambda kv: int(kv[0])):
        if LENS and length not in LENS:
            continue
        for conc in CONCS:
            try:
                row = run_cell(plist, conc)
            except Exception as e:
                print(json.dumps({"cell": f"in{length}_c{conc}", "error": str(e)[:200]}))
                time.sleep(5)
                continue
            row["cell"] = f"in{length}_c{conc}"
            row["input_len"] = int(length)
            emit(row)
elif MODE == "cats":
    one_request(PROMPTS["categories"]["chat"][0], 32)
    for cat, plist in PROMPTS["categories"].items():
        # sequential c=1 over 8 prompts
        s0 = spec_counters()
        results = [one_request(p) for p in plist]
        s1 = spec_counters()
        tps = [r["completion_tokens"] / r["elapsed_s"] for r in results]
        row = {
            "tag": TAG,
            "cell": f"cat_{cat}_c1",
            "category": cat,
            "conc": 1,
            "n_prompts": len(plist),
            "prompt_tokens_mean": sum(r["prompt_tokens"] for r in results)
            / len(results),
            "per_stream_tps": round(sum(tps) / len(tps), 2),
            "per_stream_tps_min": round(min(tps), 2),
            "per_stream_tps_max": round(max(tps), 2),
            "aggregate_tps": round(
                sum(r["completion_tokens"] for r in results)
                / sum(r["elapsed_s"] for r in results),
                2,
            ),
        }
        if s0 and s1 and "drafted" in s1:
            dd = s1["drafted"] - s0["drafted"]
            da = s1["accepted"] - s0["accepted"]
            if dd > 0:
                row["accept_rate"] = round(da / dd, 4)
        emit(row)
        row4 = run_cell(plist[:4], 4)
        row4["cell"] = f"cat_{cat}_c4"
        row4["category"] = cat
        emit(row4)
else:
    raise SystemExit("mode must be grid or cats")
