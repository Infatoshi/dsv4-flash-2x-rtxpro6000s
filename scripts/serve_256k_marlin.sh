#!/bin/bash
# 256k x 1 stream, MARLIN W4A16 MoE (2026-08-04): bf16 activations — numerically
# identical to serving the source MXFP4 checkpoint natively on sm_120 (weights are
# a lossless e8m0->e4m3 remap; marlin ignores the calibrated input_scale). A/B vs
# serve_256k.sh (FLASHINFER_CUTLASS W4A4 static-calibrated fp4 activations).
# Measured 2026-08-04: decode 202.7 tok/s (vs 193 cutlass), KV pool 418,136 tok
# (vs 389k), dspark accept len 2.73, prefill 2,964 tok/s at 44.5k. Storage proof:
# expert e4m3 scales 100% powers of two, adjacent pairs equal, scale_2=2^-9 =>
# NVFP4 weights are a lossless remap of source MXFP4; marlin W4A16 is therefore
# bit-identical math to native MXFP4 serving (TRTLLM mxfp4 is SM100-only).
# Pool 389k tokens (1.48x at 262144). Served as model id: dsv4-flash (omp talks to this).
export VLLM_ENFORCE_STRICT_TOOL_CALLING=1
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
  --kernel-config '{"linear_backend": "triton", "moe_backend": "marlin"}' \
  --max-model-len 262144 --max-num-seqs 1 \
  --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.95 --kv-cache-dtype fp8 --served-model-name dsv4-flash --enable-auto-tool-choice --tool-call-parser deepseek_v4 \
  --block-size 256 --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' --speculative-config '{"method": "dspark", "num_speculative_tokens": 3}' "$@"
# (eager fallback for debugging: opt/serve_eager_fallback.sh)
