"""Live decode-speed demo against a running dsv4-flash server.

Builds a synthetic ~36k-token context (fictional ops runbook) with one
retrieval needle buried in the middle, then streams an answer that must
(a) retrieve the needle and (b) keep decoding — showing a live tok/s
readout in a bottom status bar. Exact final rate comes from
server-reported usage, not event counting.

Usage:
    python demo/stream_demo.py [--url http://HOST:8000] [--ctx-chars 140000]
    python demo/stream_demo.py --prime-only   # warm the prefix cache, print nothing else

Run --prime-only once first so the recorded run shows warm-cache TTFT
instead of a ~12 s cold 36k prefill.
"""

import argparse
import json
import shutil
import sys
import time
import urllib.request

NEEDLE = "ZEBRA-471"

SECTION = (
    "## Runbook section {i}: {name}\n"
    "The {name} subsystem exposes three health probes. The liveness probe "
    "checks the event-loop lag and fails past 250 ms. The readiness probe "
    "verifies that the shard map has converged and that replica {i} holds a "
    "current lease. The startup probe waits for the write-ahead log replay "
    "to finish before admitting traffic. On-call procedure: if the pager "
    "fires for {name}, first check the lease table, then the replay lag "
    "gauge, then restart at most one replica at a time and wait for the "
    "shard map to reconverge before touching the next. Rollbacks use the "
    "blue/green pair and take roughly four minutes end to end.\n\n"
)
NAMES = [
    "ingest", "compactor", "router", "scheduler", "indexer", "billing",
    "archive", "metrics", "gateway", "replicator", "planner", "sessions",
]


def build_context(target_chars: int) -> str:
    parts, i = [], 0
    while sum(len(p) for p in parts) < target_chars:
        i += 1
        parts.append(SECTION.format(i=i, name=NAMES[i % len(NAMES)]))
        if i == 60:  # bury the needle mid-document
            parts.append(
                "NOTE: the emergency deployment key for the paging bypass "
                f"is {NEEDLE}. It rotates quarterly.\n\n"
            )
    return "".join(parts)


QUESTION = (
    "Two tasks. First: what is the emergency deployment key mentioned "
    "somewhere in the runbook above? State it in your first sentence. "
    "Second: explain step by step how speculative decoding speeds up LLM "
    "inference, ending with a short Python sketch of the accept/reject loop."
)


def sse_events(resp):
    for raw in resp:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            return
        yield json.loads(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--ctx-chars", type=int, default=140_000)
    ap.add_argument("--max-tokens", type=int, default=700)
    ap.add_argument("--prime-only", action="store_true")
    args = ap.parse_args()

    context = build_context(args.ctx_chars)
    body = {
        "model": "dsv4-flash",
        "temperature": 0.0,
        "max_tokens": args.max_tokens,
        "stream": True,
        # continuous_usage_stats: exact cumulative completion_tokens in every
        # chunk (spec decode batches tokens per SSE event, so event counting
        # or chars/4 estimates both mislead)
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
        "messages": [
            {"role": "user", "content": context + "\n\n" + QUESTION},
        ],
    }

    if args.prime_only:
        body.update(stream=False, max_tokens=1)
        body.pop("stream_options")
        req = urllib.request.Request(
            args.url + "/v1/chat/completions",
            json.dumps(body).encode(),
            {"Content-Type": "application/json"},
        )
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=600) as r:
            u = json.load(r).get("usage", {})
        print(f"primed: {u.get('prompt_tokens', '?')} prompt tokens "
              f"in {time.time() - t0:.1f}s")
        return

    cols, rows = shutil.get_terminal_size((100, 24))
    W = sys.stdout.write

    # reserve the last row as a status bar via a scroll region
    W(f"\x1b[1;{rows - 1}r\x1b[H\x1b[2J")

    def status(line):
        W(f"\x1b7\x1b[{rows};1H\x1b[2K\x1b[7m {line[: cols - 2]} \x1b[0m\x1b8")

    W("\x1b[1mdsv4-flash\x1b[0m on 2x RTX PRO 6000 Blackwell (TP=2, sm_120)\n")
    W(f"context: ~31k-token runbook with a buried needle | temperature 0\n\n")
    W(f"\x1b[2m> {QUESTION}\x1b[0m\n\n")
    sys.stdout.flush()

    req = urllib.request.Request(
        args.url + "/v1/chat/completions",
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    t0 = time.time()
    ttft = None
    usage = None
    last_status = 0.0
    with urllib.request.urlopen(req, timeout=600) as resp:
        for ev in sse_events(resp):
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices", []):
                delta = ch.get("delta", {}).get("content")
                if delta:
                    if ttft is None:
                        ttft = time.time() - t0
                    W(delta)
            now = time.time()
            if ttft is not None and usage and now - last_status > 0.1:
                last_status = now
                n = usage["completion_tokens"]
                rate = n / max(now - t0 - ttft, 1e-6)
                status(
                    f"streaming  {n:4d} tok   {now - t0:5.2f} s   "
                    f"{rate:5.0f} tok/s decode"
                )
            sys.stdout.flush()

    wall = time.time() - t0
    n = usage["completion_tokens"] if usage else 0
    decode_s = wall - (ttft or 0)
    # finale: reset scroll region, rewrite the bottom rows cleanly
    W("\x1b[r")
    sep = "-" * min(cols - 1, 74)
    lines = [
        "",
        f"\x1b[1m{sep}\x1b[0m",
        f"\x1b[1m{n} tokens in {decode_s:.2f} s -> "
        f"{n / decode_s:.1f} tok/s decode\x1b[0m (server-reported usage)",
        f"ttft {ttft:.2f} s (warm prefix cache) | "
        f"prompt {usage.get('prompt_tokens', 0):,} tok | "
        f"DSpark k=3 + CUDA graphs",
    ]
    for r, line in enumerate(lines):
        W(f"\x1b[{rows - len(lines) + 1 + r};1H\x1b[2K" + line)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
