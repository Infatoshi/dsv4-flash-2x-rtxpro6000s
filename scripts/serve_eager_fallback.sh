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
  --block-size 256 --enforce-eager "$@"
