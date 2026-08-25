#!/bin/bash
# 4x RTX PRO 6000 Blackwell Server Edition (sm_120), 256k x 1 stream.
# Same flags as serve_256k_marlin.sh except TP=4. DSpark k=3 still requires
# --max-num-seqs 1 (docs/GOTCHAS.md #3). Notes: docs/4X-RTX-PRO-6000.md.
#
# Default CUDA_HOME is 13.2 (2x workstation). 4x Verda image was 13.0 —
# override: CUDA_HOME=/usr/local/cuda-13.0 ./scripts/serve_256k_tp4.sh
export VLLM_ENFORCE_STRICT_TOOL_CALLING=1
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.2}"
export PATH="$CUDA_HOME/bin:$PATH"
export VLLM_USE_DEEP_GEMM=0
export MAX_JOBS=4
VENV="${VENV:-$HOME/venvs/vllm-dsv4}"
MODEL="${MODEL:-$HOME/kernels/DeepSeek-V4-Flash-0731-NVFP4}"
exec "$VENV/bin/vllm" serve "$MODEL" \
  --host 0.0.0.0 --port 8000 \
  --trust-remote-code --tokenizer-mode deepseek_v4 \
  --reasoning-parser deepseek_v4 \
  --tensor-parallel-size 4 \
  --kernel-config '{"linear_backend": "triton", "moe_backend": "marlin"}' \
  --max-model-len 262144 --max-num-seqs 1 \
  --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.95 --kv-cache-dtype fp8 --served-model-name dsv4-flash --enable-auto-tool-choice --tool-call-parser deepseek_v4 \
  --block-size 256 --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' --speculative-config '{"method": "dspark", "num_speculative_tokens": 3}' "$@"
