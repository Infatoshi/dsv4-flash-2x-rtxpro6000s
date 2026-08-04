"""Capture-safe sm_120 replacements for DSV4 indexer kernels.

fp8_paged_mqa_logits_triton: decode Lightning-Indexer logits against the
fp8_ds_mla indexer KV cache. Pages are PLANAR: block_size*D fp8 value bytes
first, then block_size fp32 scales (verified against the CUDA gather kernel
and the deep_gemm reference). Supports next_n >= 1 (MTP): row (b, j) sees
tokens [0, limit) where limit = ctx[b, j] if 2D lens are given per position,
else ctx[b] - next_n + j + 1. No host syncs — safe inside CUDA graph capture.

NOTE: vLLM's fp8_paged_mqa_logits_torch next_n>1 branch reads the cache as
per-token interleaved, which does not match the planar cache layout — do not
verify against it; use a planar-corrected reference.

Source of truth: patches/new-files/dsv4_sm120_ops.py in the dsv4-flash-2x-rtxpro6000s repo
(installed into the venv site-packages root by patches/apply.sh).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _paged_mqa_logits_kernel(
    q_ptr,          # fp8e4m3 [R, H, D] contiguous (R = B * next_n)
    kv_u8_ptr,      # uint8 view of kv cache
    kv_f32_ptr,     # fp32 view of same storage
    w_ptr,          # [R, H] fp32
    lim_ptr,        # [R] int32 per-row token limit
    bt_ptr,         # [B, max_pages] int32
    out_ptr,        # [R, max_model_len] fp32, pre-filled -inf
    max_pages,
    max_model_len,
    bt_stride,
    page_stride_bytes,
    NEXT_N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    SUB_N: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    b = row // NEXT_N

    limit = tl.load(lim_ptr + row)
    start = tile * SUB_N
    if start >= limit:
        return
    page_rank = start // BLOCK_SIZE
    if page_rank >= max_pages:
        return
    page = tl.load(bt_ptr + b * bt_stride + page_rank).to(tl.int64)
    tok0 = start % BLOCK_SIZE

    offs_n = tl.arange(0, SUB_N)
    offs_d = tl.arange(0, D)
    offs_h = tl.arange(0, H)
    tok_valid = (start + offs_n) < limit

    # planar page: block_size*D value bytes, then block_size fp32 scales
    kv_base = kv_u8_ptr + page * page_stride_bytes
    k = tl.load(
        kv_base + (tok0 + offs_n[:, None]) * D + offs_d[None, :],
        mask=tok_valid[:, None], other=0,
    ).to(tl.float8e4nv, bitcast=True).to(tl.bfloat16)

    scale_base = kv_f32_ptr + page * (page_stride_bytes // 4) + BLOCK_SIZE * D // 4
    k_scale = tl.load(scale_base + tok0 + offs_n, mask=tok_valid, other=0.0)

    q = tl.load(
        q_ptr + row.to(tl.int64) * H * D + offs_h[:, None] * D + offs_d[None, :]
    ).to(tl.bfloat16)
    w = tl.load(w_ptr + row * H + offs_h)

    scores = tl.dot(q, tl.trans(k)).to(tl.float32)  # [H, SUB_N]
    scores = tl.maximum(scores, 0.0) * w[:, None]
    acc = tl.sum(scores, axis=0) * k_scale

    out = tl.where(tok_valid, acc, float("-inf"))
    tl.store(
        out_ptr + row.to(tl.int64) * max_model_len + start + offs_n, out,
        mask=(start + offs_n) < max_model_len,
    )


def fp8_paged_mqa_logits_triton(
    q: torch.Tensor,             # [B, next_n, H, D] fp8e4m3
    kv_cache: torch.Tensor,      # uint8 [num_blocks, block_size, 1, D+4]
    weights: torch.Tensor,       # [B*next_n, H] fp32
    context_lens: torch.Tensor,  # [B], [B,1], or [B,next_n] int32, device
    block_tables: torch.Tensor,  # [B, max_pages] int32
    max_model_len: int,
) -> torch.Tensor:
    assert q.dim() == 4
    B, next_n, H, D = q.shape
    R = B * next_n
    block_size = kv_cache.shape[1]

    if context_lens.dim() == 1:
        context_lens = context_lens.unsqueeze(-1)
    if context_lens.shape[1] == next_n:
        limits = context_lens
    else:
        assert context_lens.shape[1] == 1
        # row j of seq b attends to ctx_b - next_n + j + 1 tokens
        limits = context_lens + torch.arange(
            1 - next_n, 1, device=context_lens.device, dtype=context_lens.dtype
        ).unsqueeze(0)
    limits = limits.reshape(R).to(torch.int32).contiguous()

    q3 = q.reshape(R, H, D)
    if not q3.is_contiguous():
        q3 = q3.contiguous()
    w = weights[:R]
    if not w.is_contiguous():
        w = w.contiguous()
    kv_u8 = kv_cache.view(torch.uint8)
    kv_f32 = kv_cache.view(torch.float32)
    # Page stride from the LIVE tensor: the paged pool may hand this layer a
    # strided view of a shared allocation, so BLOCK_SIZE*(D+4) is only a lower
    # bound on the distance between consecutive pages.
    page_stride_bytes = int(kv_u8.stride(0))
    assert page_stride_bytes % 4 == 0 and page_stride_bytes >= block_size * (D + 4)
    out = torch.full(
        (R, max_model_len), float("-inf"), device=q.device, dtype=torch.float32
    )
    SUB_N = 64
    grid = (R, triton.cdiv(max_model_len, SUB_N))
    _paged_mqa_logits_kernel[grid](
        q3, kv_u8, kv_f32, w,
        limits, block_tables.to(torch.int32), out,
        block_tables.shape[1], max_model_len, block_tables.stride(0),
        page_stride_bytes,
        NEXT_N=next_n, H=H, D=D, BLOCK_SIZE=block_size, SUB_N=SUB_N,
        num_warps=4, num_stages=2,
    )
    return out
