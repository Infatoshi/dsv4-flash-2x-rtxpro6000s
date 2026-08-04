#!/bin/bash
# vLLM serve for DeepSeek-V4-Flash-0731-NVFP4 on 2x RTX PRO 6000 (TP=2, sm_120).
#
# sm_120 accommodations (vllm 0.26.1rc1.dev303 nightly, flashinfer pinned 0.6.15.post1):
#   - CUDA on PATH: flashinfer JITs its sparse-MLA kernels, needs nvcc.
#   - VLLM_USE_DEEP_GEMM=0: vendored DeepGEMM has no sm_120 support
#     (SF-layout + hyperconnection asserts).
#   - linear_backend=triton: cutlass c3x fp8 block scaled_mm also rejects sm_120.
#   - venv patches (re-apply all if the venv is reinstalled):
#       1. mhc/tilelang.py: deep_gemm guard in mhc_pre_broadcast_tilelang
#       2. linear/scaled_mm/triton.py: upcast e8m0 scales to fp32 (triton 3.7 KeyError)
#       3. deepseek_v4/nvidia/flashinfer_sparse.py: both _o_proj -> rocm_inv_rope_einsum on sm_120
#       4. v1/attention/backends/mla/indexer.py: skip deep_gemm schedule-metadata build on sm_120
#       7. sparse_attn_indexer.py decode next_n==1 -> dsv4_sm120_ops.fp8_paged_mqa_logits_triton
#          (capture-safe; kernel source in opt/, copy in venv site-packages; verified 3e-7 rel)
#       5+6. sparse_attn_indexer.py: prefill/decode indexer logits -> torch reference
#            (fp8_mqa_logits_torch / fp8_paged_mqa_logits_torch) on sm_120 [slow: python loop]
#   - (old note) mhc/tilelang.py in the venv is patched to guard an unconditional DeepGEMM
#     call (first-layer mHC pre) - re-apply if the venv is reinstalled.
#   - no --load-format instanttensor: io_uring buffer registration exceeds the
#     8MB RLIMIT_MEMLOCK hard limit on our host (raise ulimit or drop the flag).
#
# Weights ~87.8GB/GPU of 96GB: context/batch kept small; drop --max-model-len
# to 32768 if KV allocation fails at startup.
# venv patch 8: flashinfer/mla/_core.py — pad sparse-MLA decode indices with -1
#   to the next instantiated topk (dspark drafter uses topk=256; sm120 decode
#   kernels only exist for 128/512/1024). Verified bit-exact vs native dispatch.
# venv patch 9: vllm deepseek_v4/nvidia/dspark.py load_weights — mtp.* experts
#   ship preserved MXFP4 (e8m0 group-32, exponents -9..-1) while the module is
#   built NVFP4; requantize scales e4m3=2^(e8m0+9) group-16 (lossless),
#   weight_scale_2=2^-9, input_scale=1.0. Without this drafts are NaN -> 0% accept.
# Patches 11/11c/12 (2026-08-04, tool calling for agent harnesses):
#   11:  entrypoints/openai/chat_completion/serving.py — chat path never called
#        parser.adjust_request, so structural-tag tool grammar was dead code;
#        wired in + absent tool_choice with tools defaults to auto.
#   11b: tool_parsers/structural_tag_registry.py — build grammar for non-strict
#        tools too (omp sends strict: None).
#   11c: same file — trigger grammar on bare <|DSML|> marker instead of the full
#        "<|DSML|tool_calls>" opener the model rarely types unaided.
#   12:  tokenizers/deepseek_v4.py — append recency-position format reminder as
#        last system message when tools present (long prompts drift the opener).
#   Result: omp 17k-token agentic prompt: 0/8 -> 8/8 structured tool calls.
# Patch 5b (2026-08-04): sparse_attn_indexer.py sm_120 prefill fallback now chunks
#   fp8_mqa_logits_torch over 64-query slices — the [H,M,N] fp32 einsum intermediate
#   OOMed at 16-stream chunked prefill (1.3GiB transient at 0.95 util).
# Sweep results (conc 1-16 x 512/2048/8192 x spec-vs-base): opt/sweep_results.jsonl,
#   tables via opt/report_sweep.py. dspark wins <=c4; base wins c>=8 at long ctx.
# Best single-user decode: opt/serve_dspark_full.sh (dspark k=3 + FULL_AND_PIECEWISE)
#   = 193.2 tok/s vs 109.3 non-spec (bench.py, 2026-08-03).
export CUDA_HOME=/usr/local/cuda-13.2
export PATH=/usr/local/cuda-13.2/bin:$PATH
export VLLM_USE_DEEP_GEMM=0
export MAX_JOBS=4
VENV="${VENV:-$HOME/venvs/vllm-dsv4}"
MODEL="${MODEL:-$HOME/kernels/DeepSeek-V4-Flash-0731-NVFP4}"
exec "$VENV/bin/vllm" serve "$MODEL" \
  --host 0.0.0.0 --port 8000 \
  --trust-remote-code --tokenizer-mode deepseek_v4 \
  --tensor-parallel-size 2 \
  --kernel-config '{"linear_backend": "triton"}' \
  --max-model-len 65536 --max-num-seqs 4 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.95 --kv-cache-dtype fp8 --served-model-name dsv4-flash \
  --block-size 256 --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' "$@"
# (eager fallback for debugging: opt/serve_eager_fallback.sh)
