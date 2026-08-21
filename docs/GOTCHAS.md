# Gotchas and war stories

Everything here cost real debugging time. Ordered by how much.

## 1. The strided paged-cache bug (silent long-context corruption)

**Symptom:** model is locally fluent but long-range recall is destroyed —
username-needle at 36k depth ~1/8 correct, garbled proper nouns from earlier
context, repetition loops in agent sessions. Failure rate changes when you
change `--max-model-len` or `--max-num-batched-tokens`, which makes it look
like a YaRN/rope or chunked-prefill bug. It is neither.

**Cause:** the custom sm_120 decode-indexer kernel computed page offsets as
`page * BLOCK_SIZE * (D + 4)` (contiguous assumption). vLLM's paged pool
hands each layer a *strided view* of one shared allocation — live
`stride(0)` was 1,002,240 bytes vs the assumed 8,448. The kernel read other
layers' memory: fp32 compressor states interpreted as fp8 scales produced
logits up to 3e38 (top-k selected garbage), zeroed regions made ~half the
context invisible. Pool geometry (hence the stride, hence the failure rate)
moves with every memory-related CLI flag.

**Fix:** read `kv_u8.stride(0)` from the live tensor in the wrapper, pass it
into the kernel, assert it's >= the payload size. Shipped inside patch 7.

**Debug method that cracked it** (after config-sweeps went nowhere):
1. Dump the kernel's exact inputs and outputs at decode time.
2. Recompute the logits offline in torch with a stride-aware gather —
   offline-sane vs live-garbage on identical inputs.
3. Standalone synthetic test of the kernel passes (max err 4e-6) — so the
   kernel is right *for contiguous inputs*.
4. Re-run the live inputs in place — bit-identical garbage. Only remaining
   difference: the live tensor's memory layout. Print `.stride()`. Done.

**Moral:** custom kernels against vLLM paged caches must never assume
contiguity. Take every stride from the live tensor.

## 2. NaN drafts: 0% acceptance with a perfectly coherent model

**Symptom:** dspark speculative decode runs, target output is coherent,
acceptance is exactly 0.000%.

**Cause:** the NVFP4 conversion only converted `layers.*.ffn.experts`; the
`mtp.*` drafter experts are preserved MXFP4 (e8m0 group-32 scales, exponents
-9..-1) and the HF quant config excludes `mtp.*` — but vLLM's dspark loader
renames `mtp.* -> model.layers.*` before matching, so the exclusion never
fires and the module is built NVFP4. e8m0 scales loaded into e4m3 params with
zero global scales => NaN draft logits.

**Fix (patch 9):** requantize in `load_weights`: scales
`e4m3 = 2^(e8m0+9)` group-16 (lossless), `weight_scale_2 = 2^-9`,
`input_scale = 1.0`.

**Debug heuristic:** 0% acceptance with a coherent target = drafter-side
structural breakage. Probe the drafted token ids first — constant token 0
means NaN logits.

## 3. Short-KV crash at concurrency > 1 (spec decode)

**Symptom:** engine dies with `Triton Error [CUDA]: device-side assert
triggered` surfacing in a random downstream GEMM; the real assert (earlier in
the log) is hundreds of `indexSelectSmallIndex ... srcIndex < srcSelectDimSize`.

**Cause:** with `--max-num-seqs > 1`, a short request (KV < 512 tokens =
`index_topk`) can decode *eagerly* in a mixed batch alongside another
request's chunked prefill. flashinfer has no tuned config for that shape and
falls back to `SparseMlaDecodeV3Runner`, which does an `index_select` with
out-of-range top-k indices when `kv_len < index_topk`. At `--max-num-seqs 1`
decode is always a uniform batch replaying the captured FULL graph — path
unreachable, long-term stable.

**Status:** open. Keep spec-decode configs at `--max-num-seqs 1`, or serve
multi-user without spec decode. Proper fix: pad/clamp top-k indices when
`kv_len < index_topk` in the fallback runner (or add tuned configs for the
small shapes).

This is not “run two autotune tactics in one launch.” The sm120 sparse-MLA
cubins are specialized on `(heads, topk, page)`. Different tuned params =
different cubin. The pooling trick is to make every decode look like the
instantiated shape:

1. Clamp/pad indices (`-1` for OOB / short KV) and pass `topk_length = kv_len`.
   The tuned kernel already treats `-1` as invalid; `SparseMlaDecodeV3Runner`
   does not.
2. Dummy KV pages so every seq has `kv_len >= 512` (two `block_size=256` pages).
3. Pad batch `T` to a captured FULL-graph size with masked dummy queries.
4. Do not mix prefill and decode in one scheduler step.

4x log (2026-08-21, TP=4, seqs=16, no spec): `No tuned config covers
sparse_mla_sm120_decode_dsv4` at batch 16/8/4/1, fallback
`SparseMlaDecodeV3Runner tactic=-1` (“perf cliff”), then JIT during
inference (`hc_prenorm_gemm_tilelang`, indexer topk, tool bitmask). Engine
generation throughput sat at 72.9 tok/s. See `docs/4X-RTX-PRO-6000.md`.

**Debugging note:** device-side asserts are asynchronous — the kernel named
in the Python traceback is the messenger, not the killer. Search the log
*upward* for the `Assertion ... failed` lines from the CUDA runtime.

## 4. Assorted sm_120 landmines

- **DeepGEMM:** vendored copy has zero sm_120 kernels; several call sites
  invoke it unconditionally (patches 1, 4-6). `VLLM_USE_DEEP_GEMM=0` is not
  sufficient on its own.
- **cutlass c3x fp8 block scaled-mm** rejects sm_120 — use
  `--kernel-config '{"linear_backend": "triton"}'` plus the tuned configs
  (vLLM ships no tuned w8a8-block triton configs for RTX PRO 6000).
- **TRTLLM MXFP4 MoE kernels** are capability-family-100 only, even though
  sm_120 has FP4 tensor-core hardware: vLLM's
  `fused_moe/experts/trtllm_mxfp4_moe.py` gates `is_supported_config` on
  `is_device_capability_family(100)` (SM100/SM103 datacenter Blackwell), and
  the flashinfer module it calls is literally
  `gen_trtllm_gen_fused_moe_sm100_module` (prebuilt trtllm-gen cubins).
  vLLM's mxfp4 backend priority is TRTLLM -> DeepGEMM (no sm_120 kernels) ->
  Marlin, so serving the *original MXFP4 checkpoint* on sm_120 lands on
  marlin W4A16 — the same kernels, on numerically identical weights, as this
  repo's NVFP4 + `moe_backend: marlin` config (the checkpoint is a lossless
  remap; see README). There is no faster native-MXFP4 path on sm_120 in this
  stack. The only sm_120 path with fp4 *activations* is FLASHINFER_CUTLASS
  NVFP4 W4A4 — measured slower at batch 1 (193 vs 202.7 decode; weight-
  bandwidth-bound either way, activation quant adds overhead) and it is the
  source of the converted-vs-reference quality delta.
- **flashinfer sm_120 sparse-MLA decode kernels** exist only for top-k
  {128, 512, 1024} at page 64. The drafter's SWA uses top-k 256 — pad
  indices with -1 up to 512 (patch 8).
- **Indexer cache pages are planar** (block_size x D fp8 bytes, then
  block_size fp32 scales). vLLM's own `fp8_paged_mqa_logits_torch`
  `next_n>1` branch assumes per-token interleaved — don't use it as a
  verification reference.
- **Contexts <= 512 tokens bypass the indexer logits path** entirely —
  short-prompt smoke tests will not exercise any of the indexer patches.
- **FULL CUDA graph capture dies on `.item()`** anywhere in the hot path;
  keep replacements capture-safe (no host syncs).
- **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` silently kills CUDA
  graph capture.** Never combine with graphs.
- **`--load-format instanttensor`** needs io_uring buffer registration above
  the default 8 MB `RLIMIT_MEMLOCK` — raise the ulimit or drop the flag.
- **Weights are ~87.8 GB per 96 GB GPU.** Single-chunk prefill configs above
  ~4096 batched tokens OOM at load; chunked prefill (2048-4096) is mandatory.
  The vLLM caching-allocator may log a one-off `memory allocation failed with
  OOM` warning under burst load and recover — treat repeated ones as a real
  headroom problem.

## 5. Measurement traps

- With spec decode, count generated tokens via `usage.completion_tokens`
  (`stream_options: {"include_usage": true}`), not SSE event count (~3x low).
- Simultaneous-start concurrency benches: the earliest stream decodes during
  the batch ramp and reads ~30% low on short runs; use 1024+ token
  generations for steady state.
- Temperature-0 nondeterminism run-to-run is real and harmless: prefill
  split-KV reduction order jitters. Don't chase bit-identical outputs across
  runs, compare content.

## 6. 4x / already-patched venv

- **`apply.sh` is not always idempotent.** `--forward` skips reversed hunks,
  but offset hunks on `sparse_attn_indexer.py` can still apply on a tree that
  already has patches 5-7/13. On 2026-08-21 that file grew 34,944 → 36,599
  bytes (`Hunk succeeded` / offset). If you see that instead of `Reversed
  (or previously applied)`, restore the file from the known-good venv.
- **Pinned nightly wheel rotates off.** `0.26.1rc1.dev303+g74295e3bd` was
  gone from `wheels.vllm.ai/nightly` on 2026-08-21. Copy the patched
  `vllm` + `flashinfer` + `dsv4_sm120_*.py` from a working venv, or build
  vLLM at commit `74295e3bd`.
- **`torchvision` is a hard import** in this vLLM cut. Workers die after
  weight load with `No module named 'torchvision'` if you only installed
  `torch` + `triton`.
- **Claude Code is the wrong client.** omp OpenAI (`…/v1`, model
  `dsv4-flash`) is how anvil talks to this stack. Pointing
  `ANTHROPIC_BASE_URL` at vLLM `/v1/messages` returns 200 and then Flash
  parrots Claude Code's stop protocol.
- **Verda 50 GB OS volume cannot hold the 176 GB checkpoint.** Extra NVMe
  attaches only while the VM is shutdown. tmpfs dies on reboot.

4x serve notes and the TP=4 script: `docs/4X-RTX-PRO-6000.md`,
`scripts/serve_256k_tp4.sh`.

