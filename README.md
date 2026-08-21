# DeepSeek-V4-Flash on 2x RTX PRO 6000 Blackwell

Serving [DeepSeek-V4-Flash-0731-NVFP4](https://huggingface.co/MJPansa/DeepSeek-V4-Flash-0731-NVFP4)
(~176 GB of weights) on a single workstation with two RTX PRO 6000 Blackwell 96 GB
GPUs (`sm_120`), TP=2, via a patched vLLM nightly.

![live decode demo](demo/decode-demo.gif)

*Live recording (`demo/stream_demo.py`): needle retrieval + 242 tok/s decode
at a 30,765-token prompt, server-reported usage, warm prefix cache. Regenerate
with `asciinema rec -c "python demo/stream_demo.py" demo.cast && agg demo.cast out.gif`
after a `--prime-only` warmup run.*

Headline numbers (batch-1 decode, temperature 0, measured with `bench/bench.py`):

| config | decode tok/s |
|---|---|
| eager, no patches beyond load fixes | 17.8 |
| piecewise CUDA graphs | 77.1 |
| + tuned GEMM configs + triton indexer kernel | 94.8 |
| + FULL_AND_PIECEWISE graphs | 109.3 |
| + DSpark speculative decode (k=3) | 193.2 |
| + Marlin W4A16 MoE backend | **202.7** |

Context up to 512k tokens with CUDA graphs (1M fits eager). A real 504,381-token
request served end-to-end: **1.9 min prefill (4,339 tok/s average)**, 66 tok/s
decode at 500k depth. Prefill was 18.3 min before patch 13 replaced the
indexer fallback with a fused triton kernel (9.5x end-to-end; the indexer op
itself is 61-76x — see `docs/BENCHMARKS.md` and `bench/prefill/`).

**None of this works out of the box.** `sm_120` (Blackwell workstation) is not
`sm_100` (B200/GB300): DeepGEMM has no sm_120 kernels, cutlass c3x fp8 block
scaled-mm rejects the arch, TRTLLM MXFP4 MoE kernels are SM100-family-only, and
flashinfer's sparse-MLA decode kernels only exist for specific top-k values.
This repo is the complete recipe: 13 patches, tuned kernel configs, custom
triton kernels for the sparse-attention indexer (decode and prefill), serve
scripts for every configuration we validated, and the benchmark data.

## Contents

```
patches/        unified diffs vs upstream (vLLM @ 74295e3bd, flashinfer 0.6.15.post1)
                + new files (triton indexer kernel, tuned GEMM configs) + apply.sh
scripts/        serve scripts, one per validated configuration
bench/          benchmark harness + raw results (scoreboard.jsonl, sweep_results.jsonl)
docs/           BENCHMARKS.md, GOTCHAS.md, TOOL-CALLING.md, 4X-RTX-PRO-6000.md
```

## Requirements

- 2x NVIDIA RTX PRO 6000 Blackwell Workstation Edition (96 GB, `sm_120`)
- CUDA 13.2 toolkit on PATH (flashinfer JIT-compiles sparse-MLA kernels, needs `nvcc`)
- Python 3.12
- Exact versions this recipe was validated against:
  - vLLM `0.26.1rc1.dev303+g74295e3bd` (nightly; build from source at commit
    [`74295e3bd`](https://github.com/vllm-project/vllm/commit/74295e3bd) if the
    wheel has rotated off the nightly index — the patches are diffed against
    exactly that commit)
  - `flashinfer-python==0.6.15.post1`
  - `torch==2.13.0`, `triton==3.7.1`, `xgrammar==0.2.3`, `transformers==5.14.1`

Weights are ~87.8 GB per GPU at TP=2, leaving ~8 GB for KV cache, graphs and
activations — every context/batch setting below is shaped by that budget.

## Quickstart

```bash
# 1. venv with the pinned stack
uv venv ~/venvs/vllm-dsv4 --python 3.12
uv pip install --python ~/venvs/vllm-dsv4/bin/python \
  "vllm==0.26.1rc1.dev303+g74295e3bd" --extra-index-url https://wheels.vllm.ai/nightly \
  "flashinfer-python==0.6.15.post1"

# 2. apply all patches (see patches/README.md for what each one does)
./patches/apply.sh ~/venvs/vllm-dsv4/lib/python3.12/site-packages

# 3. download the checkpoint
hf download MJPansa/DeepSeek-V4-Flash-0731-NVFP4 --local-dir ~/kernels/DeepSeek-V4-Flash-0731-NVFP4

# 4. serve (edit MODEL/VENV paths at the top of the script if yours differ)
./scripts/serve_256k_marlin.sh
```

Smoke test:

```bash
curl -s http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"dsv4-flash","messages":[{"role":"user","content":"Reply with exactly: pong"}],"max_tokens":16}'
```

## Configuration matrix

Every script was benchmarked and needle-tested (long-context retrieval probe)
on this exact hardware. Pick by workload:

| script | context | seqs | spec decode | MoE backend | decode tok/s | for |
|---|---|---|---|---|---|---|
| `serve_256k_marlin.sh` | 256k | 1 | dspark k=3 | marlin W4A16 | **202.7** | single-user agent/chat (recommended) |
| `serve_256k_tp4.sh` | 256k | 1 | dspark k=3 | marlin W4A16 | one-shot 235 on 4x (not scoreboard) | 4x Server Edition, same flags, TP=4 |
| `serve_256k_cutlass.sh` | 256k | 1 | dspark k=3 | cutlass W4A4 | 193 | same, fp4 activations (see below) |
| `serve_32k_dspark_full.sh` | 32k | 2 | dspark k=3 | cutlass | 193.2 | short-context interactive |
| `serve_64k_multiuser.sh` | 64k | 4 | none | cutlass | 109/stream | multi-user, no spec |
| `serve_512k_longctx.sh` | 512k | 2 | none | cutlass | 95 @100k, 66 @500k | very long documents |
| `serve_eager_fallback.sh` | 32k | 2 | none | cutlass | 17.8 | debugging only |

### How each knob trades off

**Context window (`--max-model-len`).** KV is cheap on this model: only a few
of the 43 layers are full-attention (the rest are SWA-128), so KV costs
~10 KB/token with fp8 KV cache. 256k x 1 seq leaves a 418k-token pool;
512k x 2 seqs fits with graphs (838k pool, no spec); 1M x 1 fits eager only.
The speculative drafter + its graphs cost ~4.6 GiB and cap max len at ~427k —
that's why the 512k script drops spec decode.

**Concurrency (`--max-num-seqs`).** The validated configs use 1-4.
Per-stream decode at 256k-class configs with spec decode: c=1: 247 (favorable
content), c=2: ~196, c=4: ~140, c=8: ~100-110 steady-state (~825 aggregate).
**Warning:** `--max-num-seqs 8` crashed in production — a short request
(KV < 512 tokens, i.e. shorter than the indexer top-k) decoding eagerly in a
mixed batch hits an out-of-bounds `index_select` in flashinfer's
`SparseMlaDecodeV3Runner` fallback and device-asserts the whole engine. At
`--max-num-seqs 1` decode always replays the captured CUDA graph and the path
is unreachable. Until the fallback is fixed (pad/clamp top-k indices when
`kv_len < index_topk`), keep concurrency at 1 for spec-decode configs, or use
the non-spec multiuser script. Details in [docs/GOTCHAS.md](docs/GOTCHAS.md).

**Speculative decode (`--speculative-config '{"method":"dspark",...}'`).**
The checkpoint ships its DSpark/MTP drafter. k=3 wins big at low concurrency
(c=1: 205 vs 109 base at 512-token inputs) and loses at high concurrency with
long inputs (c=8 @ 8k input: 173 vs 196 base) because draft compute steals
batch throughput. Crossover is around c=8. k=4 gives the same mean accept
length with more variance — not worth it. Acceptance at temp 0 by content:
math 69%, code 58%, chat 55% (per-position ~76/56/31%). Two patches are
mandatory for the drafter on sm_120 (patches 8 and 9 — without patch 9 the
drafter silently produces NaN drafts and acceptance is exactly 0%).

**MoE backend (`--kernel-config '{"moe_backend": "marlin"}'`).** The NVFP4
checkpoint's expert weights are a provably lossless remap of the source MXFP4
release (e4m3 scales are 100% exact powers of two, `weight_scale_2 = 2^-9`).
The default FLASHINFER_CUTLASS backend runs W4A4 — fp4 *activations* with a
calibrated static input scale, which is the real quality delta vs the API.
Marlin runs W4A16 (bf16 activations): mathematically identical to serving the
original MXFP4 checkpoint natively, and it measured *faster* (202.7 vs 193
decode) with a bigger KV pool (no flashinfer workspace). Marlin's large-M
weakness never engages because chunked prefill caps M at
`--max-num-batched-tokens` (2048).

**CUDA graphs (`--compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}'`).**
FULL_AND_PIECEWISE is worth +14 tok/s over piecewise-only (109 vs 95) and +28%
on top of the spec-decode config (193 vs 165). Anything that breaks capture
falls back silently — see gotchas (`.item()` calls,
`expandable_segments:True`).

**Chunked prefill (`--max-num-batched-tokens`).** 2048 for the 256k configs
(prefill ~4,600 tok/s at 31k depth with the patch-13 indexer kernel; the old
fallback managed ~2,964 tok/s at 44.5k), 4096 where headroom allows. Larger
single chunks OOM — weights leave no room.

**Quantized KV (`--kv-cache-dtype fp8 --block-size 256`).** Required to fit
any useful context.

## The two bugs that will eat your week

Documented fully in [docs/GOTCHAS.md](docs/GOTCHAS.md); short version:

1. **Strided paged-cache views.** vLLM's paged KV pool hands each layer a
   *strided view* of a shared allocation. A custom indexer kernel that computes
   page offsets as `page * page_payload_bytes` reads other layers' memory —
   the model stays locally fluent but long-range recall silently dies
   (needle at 36k: ~1/8 correct), with failure rate varying by pool geometry.
   Always take `tensor.stride(0)` from the live tensor.

2. **Preserved-MXFP4 drafter experts.** The NVFP4 conversion covers the main
   model's routed experts; the `mtp.*` drafter experts ship in original MXFP4,
   but vLLM's dspark loader renames `mtp.* -> model.layers.*` so the quant
   config exclusion never matches — e8m0 scales loaded into e4m3 params =>
   NaN drafts => exactly 0% acceptance with a perfectly coherent target model.

## Benchmarks

Full methodology and raw data in [docs/BENCHMARKS.md](docs/BENCHMARKS.md) —
perf ladder with mean/std per step (`bench/scoreboard.jsonl`), concurrency x
input-length x spec-vs-base sweep (`bench/sweep_results.jsonl`), long-context
prefill/decode curves, and drafter acceptance by content type. 4x Server
Edition bring-up (2026-08-21): [docs/4X-RTX-PRO-6000.md](docs/4X-RTX-PRO-6000.md).

## Tool calling / agent harnesses

Works, but needed four patches (11-12) to make the model's DSML tool-call
format reliable under long agentic prompts (structural-tag grammar wired into
the chat path, grammar triggered on the bare `<|DSML|>` token, format reminder
injected as the last system message): 0/8 -> 8/8 structured calls on a
17k-token agent prompt. Client config and caveats in
[docs/TOOL-CALLING.md](docs/TOOL-CALLING.md).

## License

Apache-2.0. The diffs under `patches/` modify [vLLM](https://github.com/vllm-project/vllm)
and [flashinfer](https://github.com/flashinfer-ai/flashinfer) (both Apache-2.0);
tuned GEMM config JSONs follow vLLM's format and license.
