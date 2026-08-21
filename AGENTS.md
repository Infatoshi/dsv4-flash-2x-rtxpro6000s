# Agent instructions

You are probably here to get DeepSeek-V4-Flash-0731-NVFP4 serving on a 2x
RTX PRO 6000 Blackwell box for a user. This file is the operational path;
README.md explains the why.

## First: what does the user actually want?

Ask before building, but don't interrogate — one question with a default is
enough. The knobs that matter: concurrent users, max context, speculative
decode. If the user just wants it running (demo, personal use, "set it up"),
use the defaults and say so:

**Default: `scripts/serve_256k_marlin.sh`** — 256k context, 1 stream,
DSpark spec decode k=3, marlin W4A16 MoE. Fastest single-user decode
(~200 tok/s) and the long-term-stable production config.

Map answers to scripts:

| user says | script |
|---|---|
| just me / demo / unspecified | `serve_256k_marlin.sh` (default) |
| 4x RTX PRO 6000 (server) | `serve_256k_tp4.sh` (same as marlin, TP=4; notes in docs/4X-RTX-PRO-6000.md) |
| 2-4 users, interactive | `serve_32k_dspark_full.sh` (32k) or drop context expectations |
| 5+ users | `serve_64k_multiuser.sh` (no spec decode — spec loses at high concurrency anyway) |
| >256k context (long documents) | `serve_512k_longctx.sh` (512k x 2, no spec; 1M x 1 fits eager only) |
| debugging the stack | `serve_eager_fallback.sh` |

Hard constraint, do not "improve" it: **never raise `--max-num-seqs` above 1
on a spec-decode config.** A short request decoding eagerly in a mixed batch
device-asserts the whole engine (docs/GOTCHAS.md #3). Multi-user means the
non-spec script.

## Setup sequence

1. Preflight: `nvidia-smi` — 2x or 4x RTX PRO 6000 Blackwell (sm_120, 96 GB),
   idle. 2x: weights ~87.8 GB/GPU. 4x TP=4: ~44 GB/GPU. A co-tenant process
   means failure at KV allocation, not a graceful degrade. 4x notes:
   `docs/4X-RTX-PRO-6000.md`.
2. Pinned venv (see README Quickstart). Versions are exact, not minimums —
   the patches are diffs against vLLM commit `74295e3bd` and flashinfer
   `0.6.15.post1`. If the pinned nightly wheel is gone, copy a known-good
   patched `site-packages` or build vLLM from that commit. Also install
   `torchvision` (this vLLM cut imports it).
3. `./patches/apply.sh <venv>/lib/python3.12/site-packages` — intended
   idempotent; a `.rej` file means version drift, stop and report. If a hunk
   **succeeds with offset** on an already-patched tree (seen on
   `sparse_attn_indexer.py`), restore that file from the known-good venv
   rather than leaving the double patch.
4. Download the checkpoint (README step 3), ~176 GB. A 50 GB cloud OS volume
   cannot hold it; attach extra NVMe (Verda: VM must be shutdown to attach).
5. Launch the chosen script inside tmux/screen (it `exec`s vLLM in the
   foreground). Boot takes ~5-7 min cold (weight load + CUDA graph capture);
   poll `GET /health` for 200 rather than watching the log.

## Verify before reporting done

A healthy endpoint is not a working model — the failure mode documented in
this repo (docs/GOTCHAS.md #1) was a server that chatted fluently while
long-range recall was silently destroyed. Minimum verification:

1. Smoke: the curl in README Quickstart returns the exact requested string.
2. Retrieval probe: put a unique fact (a made-up username or ID) early in a
   30k+ token prompt, ask for it back at the end, temperature 0. Run it 3x —
   the corruption class is probabilistic. 3/3 or investigate.
3. If tool calling matters: send a request with `tools` and check the
   response contains structured `tool_calls`, not DSML text in `content`
   (docs/TOOL-CALLING.md).

When benchmarking: count tokens via `usage.completion_tokens` with
`stream_options: {"include_usage": true}` — SSE event counting reads ~3x low
with spec decode. Generations of 1024+ tokens for steady-state rates.

## When something breaks

- `Triton Error [CUDA]: device-side assert` — the kernel in the traceback is
  the messenger, not the cause. Search the log *upward* for CUDA
  `Assertion ... failed` lines. Known instance: GOTCHAS #3.
- 0% spec-decode acceptance with coherent output — drafter NaN, patch 9
  missing or venv reinstalled without re-applying patches.
- Fluent but amnesiac at long context — GOTCHAS #1 class; verify patches
  applied (`python -c "import dsv4_sm120_ops"`).
- KV allocation failure at startup — something else on the GPUs, or context
  too large; drop `--max-model-len` before touching `gpu-memory-utilization`.
- Short-prompt tests (< 512 tokens of context) bypass the indexer entirely —
  they prove nothing about the indexer patches.

Do not combine `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` with CUDA
graphs, and don't add `--load-format instanttensor` unless `RLIMIT_MEMLOCK`
is raised past 8 MB.
