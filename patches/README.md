# Patches

Unified diffs against **vLLM commit
[`74295e3bd`](https://github.com/vllm-project/vllm/commit/74295e3bd)**
(= nightly `0.26.1rc1.dev303+g74295e3bd`) and **flashinfer-python 0.6.15.post1**.
Apply with:

```bash
./apply.sh /path/to/venv/lib/python3.12/site-packages
```

`apply.sh` runs `patch -p1` for every diff, drops `new-files/dsv4_sm120_ops.py`
into site-packages root, and installs the tuned GEMM configs. Re-run after any
venv reinstall. `--forward` skips reversed hunks; it does **not** stop offset
hunks from applying again on an already-patched `sparse_attn_indexer.py`
(2026-08-21: file grew 34,944 → 36,599 bytes). If you see `Hunk succeeded`
instead of `Reversed (or previously applied)`, restore that file from the
known-good venv. All patches are gated on
`current_platform.is_device_capability_family(120)` (or equivalent) where they
touch shared code paths, so they are no-ops on other architectures.

## Why patches at all

`sm_120` (RTX PRO 6000 Blackwell / GeForce Blackwell) is not `sm_100`
(B200/GB300). Three kernel families this model depends on simply don't exist
for it: vendored DeepGEMM (zero sm_120 kernels), cutlass c3x fp8 block
scaled-mm (rejects the arch), and TRTLLM MXFP4 MoE (capability-family-100
only). Patches 1-7 route around those; 8-9 fix the speculative drafter;
11-12 fix tool calling; 13 replaces the prefill-indexer fallback with a fused
triton kernel (~75x at 500k context); the stride fix inside patch 7's kernel
is a correctness bug that cost us the longest debugging session of the
project.

## The set

| # | file | what / why |
|---|---|---|
| 1 | `vllm/model_executor/kernels/mhc/tilelang.py` | Guard an unconditional DeepGEMM call in `mhc_pre_broadcast_tilelang` (first-layer mHC pre) — DeepGEMM has no sm_120 support. |
| 2 | `vllm/model_executor/kernels/linear/scaled_mm/triton.py` | Upcast e8m0 scales to fp32 (triton 3.7 `KeyError` on the e8m0 dtype). |
| 3 | `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py` | Route both `_o_proj` variants to `rocm_inv_rope_einsum` on sm_120. |
| 4 | `vllm/v1/attention/backends/mla/indexer.py` | Skip DeepGEMM schedule-metadata build on sm_120. |
| 5/6 | `vllm/model_executor/layers/sparse_attn_indexer.py` | Route prefill/decode indexer logits off DeepGEMM on sm_120 (kernels absent). Prefill now calls the fused triton kernel in `new-files/dsv4_sm120_prefill.py` (patch 13); decode uses the patch-7 kernel. |
| 7 | same file + `new-files/dsv4_sm120_ops.py` | Decode `next_n>=1` indexer logits -> custom capture-safe triton kernel (`fp8_paged_mqa_logits_triton`), verified 3e-7 rel vs reference. **Contains the page-stride fix** — see below. |
| 8 | `flashinfer/mla/_core.py` | Pad sparse-MLA decode indices with `-1` up to the next instantiated top-k. The DSpark drafter uses top-k 256; flashinfer's sm_120 decode kernels are only built for top-k {128, 512, 1024} at page 64. Verified bit-exact vs native dispatch. |
| 9 | `vllm/models/deepseek_v4/nvidia/dspark.py` | `load_weights`: `mtp.*` drafter experts ship in **preserved MXFP4** (e8m0 group-32 scales, exponents -9..-1) while the module is built NVFP4 — the loader renames `mtp.* -> model.layers.*` so the quant-config exclusion never matches. Requantize scales `e4m3 = 2^(e8m0+9)` group-16 (lossless), `weight_scale_2 = 2^-9`, `input_scale = 1.0`. Without this, drafts are NaN and acceptance is exactly 0%. |
| 11 | `vllm/entrypoints/openai/chat_completion/serving.py` | Chat path never called `parser.adjust_request` — the structural-tag tool grammar was dead code. Wire it in; absent `tool_choice` with tools present defaults to `auto`. |
| 11b | `vllm/tool_parsers/structural_tag_registry.py` | Build the grammar for non-strict tools too (many clients send `strict: None`). |
| 11c | same file | Trigger the grammar on the bare `<|DSML|>` token instead of the full `<\|DSML\|tool_calls>` opener the model rarely types unaided under long prompts. |
| 12 | `vllm/tokenizers/deepseek_v4.py` | Append a recency-position tool-format reminder as the last system message when tools are present (long prompts drift the opener). |
| 13 | `sparse_attn_indexer.py` + `new-files/dsv4_sm120_prefill.py` | Fused triton kernel for the prefill Lightning-Indexer logits (`fp8_mqa_logits`), replacing the chunked torch fallback that materialized an unfused `[H, M, N]` fp32 intermediate. fp8 `tl.dot` with fp32 accumulation, `k_scale` hoisted out of the relu (`relu(x*s) = s*relu(x)` for `s >= 0`), causal-window tiles skipped. 930 ms -> 12.3 ms at M=2048 / N=500k on this hardware (75.6x; geomean 61x across the shape grid), rel err ~5e-7 vs an fp32 reference. This is the kernel that turned a 502k-token prefill from ~18 min into minutes — see docs/BENCHMARKS.md. |

New files (not diffs):

- `new-files/dsv4_sm120_ops.py` -> site-packages root. Capture-safe triton
  decode-indexer kernel. The indexer cache pages are **planar** (block_size x D
  fp8 value bytes, then block_size fp32 scales); note in-file warning that
  vLLM's own `fp8_paged_mqa_logits_torch` `next_n>1` branch assumes a
  different layout.
- `new-files/dsv4_sm120_prefill.py` -> site-packages root. Fused triton
  prefill-indexer kernel (patch 13). Asserts contiguity on every input —
  fail loudly rather than read garbage (see the page-stride bug below for
  why that paranoia is earned). Development harness, an equivalent
  hand-written CUDA kernel (tied within noise), and raw per-shape results
  live in `bench/prefill/`.
- `new-files/configs/*.json` ->
  `vllm/model_executor/layers/quantization/utils/configs/`. Tuned triton
  w8a8-block GEMM configs for RTX PRO 6000 Blackwell (vLLM ships none for this
  device; worth ~+18 tok/s over defaults).

## The page-stride bug (inside patch 7's kernel)

The original kernel computed page offsets as `page * BLOCK_SIZE * (D + 4)` —
correct only for a contiguous cache. vLLM's paged pool hands the layer a
**strided view of a shared allocation**: observed `stride(0)` = 1,002,240
bytes where the payload is 8,448. The kernel read other layers' memory as
logits input: fp32 compressor states interpreted as scales produced logits up
to 3e38, zeroed regions made half the context invisible to top-k. The model
stayed locally fluent while long-range recall silently died (36k needle:
~1/8 correct), and the failure rate moved when `--max-model-len` or
`--max-num-batched-tokens` changed, because pool geometry changes the stride.

The shipped kernel takes `page_stride_bytes = int(kv_u8.stride(0))` from the
live tensor and asserts `page_stride_bytes >= block_size * (D + 4)`.

**If you write any custom kernel against vLLM's paged caches: never assume
contiguity; take the page stride from the live tensor.**
