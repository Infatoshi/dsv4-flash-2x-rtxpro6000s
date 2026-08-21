# Benchmarks

All numbers measured on 2x RTX PRO 6000 Blackwell Workstation Edition
(TP=2, sm_120), CUDA 13.2, the pinned software stack from the README,
2026-08-03/04. Raw data ships in `bench/`.

## Methodology

The measurement contract (`bench/CONTRACT.md`): metric is decode tokens/s
(maximize), measured by `bench/bench.py` — fixed prompt, `max_tokens=256`,
`temperature=0`, `ignore_eos`, `completion_tokens / elapsed` against
`localhost:8000`, 1 warmup + 5 reps, mean and std recorded with GPU clock
stamps per row (`bench/scoreboard.jsonl`). Correctness gate: temperature-0
output text identical to the baseline capture; kernel swaps verified
atol/rtol 1e-2 against a torch reference before any end-to-end measurement.
Keep threshold epsilon 2% with a 2-sigma noise floor.

Token counting with speculative decode on: each SSE event can carry a
multi-token accepted run — count via `stream_options: {"include_usage": true}`
and `usage.completion_tokens`, never by event count (event counting reads ~3x
low).

## Single-stream perf ladder

Each step includes everything above it (from `bench/scoreboard.jsonl`,
mean over 5 reps, std < 0.6 throughout):

| step | decode tok/s |
|---|---|
| eager | 17.8 |
| piecewise CUDA graphs | 77.1 |
| + tuned GEMM configs + triton decode-indexer kernel | 94.8 |
| + FULL_AND_PIECEWISE graphs | 109.3 |
| + dspark k=3 (piecewise graphs only) | 165 |
| + dspark k=3 with FULL_AND_PIECEWISE | 193.2 |
| + marlin W4A16 MoE backend | **202.7** |

## Speculative decode acceptance (temperature 0)

- Per-position acceptance, k=3: ~76% / 56% / 31%; mean accept length 2.6-2.73.
- k=4: same mean accept length, more variance — not worth it.
- By content type: math 69%, code 58%, chat 55%. Repetitive structure drafts
  best: math hit 229 tok/s single-stream vs 193-201 for chat/code.
- Favorable content peaks higher (measured 247 tok/s at accept length 3.63 on
  a code-navigation payload with warm prefix cache).

## Concurrency x input length x spec-vs-base

256-token generations, inputs 512/2048/8192 tokens, concurrency 1-16
(`bench/sweep_results.jsonl`, tables via `bench/report_sweep.py`).
Per-stream decode tok/s:

- **dspark k=3 wins at low concurrency:** c=1: 205 / 162 / 103 tok/s at
  512 / 2k / 8k input, vs base 109 / 97 / 70.
- **Base wins at high concurrency with long inputs:** 8k input, c=8:
  base 196 vs dspark 173; c=16: base 189 vs dspark 137. Draft compute steals
  batch throughput.
- Peak aggregate: dspark 685 tok/s (512-token input, c=16), base 578.
- Crossover: around c=8 for long inputs; below that, spec decode always wins.

At the 256k marlin config (36k-token shared-prefix payload, 1024-token gens):
c=1: 247, c=2: ~196 mean (392 aggregate), c=4: ~140 (559), c=8: 100-110
steady-state (~825 aggregate). Note the c=8 measurement predates the
short-KV crash discovery — that concurrency level is not currently safe with
spec decode (see GOTCHAS).

Measurement artifact worth knowing: in a simultaneous-start burst, the
earliest-arriving stream decodes during the 3-4 s batch ramp while the other
streams' onboarding preempts it, and shows 73-78 tok/s instead of 100-110.
Short runs (256 gens) sit entirely inside the ramp and read misleadingly low.
Measure steady-state with 1024+ token generations.

## Prefill indexer kernel (patch 13)

The Lightning Indexer scores every query chunk against the *entire* KV prefix
(that global scoring is what makes top-512 selection possible, and it is the
one quadratic term in prefill). The original sm_120 route was a chunked torch
fallback, bandwidth-bound on an unfused `[H, M, N]` fp32 intermediate —
~9-20 TFLOPS. Patch 13 replaces it with a fused triton kernel: fp8 `tl.dot`
into fp32 accumulators, `k_scale` folded through the relu
(`relu(x*s) = s*relu(x)`, `s >= 0`), head reduction and causal-window
masking in-kernel, out-of-window tiles skipped entirely.

Per-op times on one RTX PRO 6000 (CUDA events, 3+ warmup, 10+ reps or 200 ms
minimum per cell; M = query chunk, N = KV length, 32 heads/rank; raw data and
harness in `bench/prefill/`):

| M | N | fallback | triton kernel | speedup |
|---|---|---|---|---|
| 2048 | 8,192 | 6.8 ms | 0.18 ms | 37x |
| 2048 | 32,768 | 52.8 ms | 0.77 ms | 68x |
| 2048 | 102,400 | 167.1 ms | 2.44 ms | 68x |
| 2048 | 262,144 | 450.4 ms | 6.34 ms | 71x |
| 2048 | 500,000 | 929.9 ms | 12.30 ms | 76x |

Geomean over the full 20-cell grid (M 256-4096 x N 8k-500k): **61x**.
Accuracy: ~5e-7 relative error vs an fp32 reference (the kernel accumulates
in fp32; the old fallback itself ran bf16). Sustained-load numbers: the 600 W
power cap sags clocks ~3-5%, so a hand-written CUDA/mma kernel (also in
`bench/prefill/`, tied within ~1% on production shapes) and the triton kernel
are indistinguishable under load; triton ships because it needs no `nvcc` at
serve time. Compute rate: ~830 TFLOPS fp8 on the inner GEMM cold — about 94%
of what cuBLAS `_scaled_mm` reaches on the same shapes on this GPU.

## Long context

Only a few of the 43 layers are full-attention (the rest are SWA-128), so KV
costs ~10 KB/token at fp8. Model is native 1M context (YaRN x16 over a 64k
original window).

- 512k x 2 seqs fits **with** CUDA graphs, no spec decode: 838k-token KV pool.
  (The drafter + its graphs cost 4.6 GiB and cap max len at ~427k.)
- 1M x 1 seq fits eager (1.54x headroom).
- Real 504,381-token request served end-to-end (512k config, cold cache,
  needle retrieved): **prefill 116 s = 4,339 tok/s average** with the patch-13
  indexer kernel. Before the kernel, the same request took 18.3 min
  (458 tok/s average; 1,725 tok/s at 100k depth) — the chunked torch fallback
  made the indexer's quadratic term dominate everything past ~50k depth.
  Decode: 95 tok/s at 100k depth, 66 tok/s at 500k (unchanged; decode was
  never indexer-fallback-bound).
- Prefill at the 256k marlin config with the kernel: ~4,600 tok/s cold at 31k
  depth (needle-verified 3/3 after the swap; the fallback managed ~2,964 tok/s
  at 44.5k).

## KV pool sizes (fp8 KV, block 256)

| config | pool (tokens) |
|---|---|
| 256k marlin + dspark (2x TP=2) | 418,136 |
| 256k cutlass + dspark (2x TP=2) | ~389k (flashinfer workspace overhead) |
| 512k x 2, no spec (2x TP=2) | ~838k |
| 256k no spec (4x TP=4, 2026-08-21) | 3,008,608 |
| 256k marlin + dspark (4x TP=4, 2026-08-21) | 4,695,313 |

4x Server Edition bring-up, CUDA 13.0, same patched stack: see
`docs/4X-RTX-PRO-6000.md`. Non-spec 4x engine log sat at 72.9 tok/s (1 req).
DSpark k=3 one-shot warm decode (400 completion tokens / 1.703 s wall, not
the scoreboard harness) was 234.8 tok/s. Do not mix those into the 2x ladder
above.
