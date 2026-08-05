#!/usr/bin/env python
"""Benchmark + correctness harness for the DSV4 prefill indexer logits op (sm_120).

The op
------
Given
    q        [M, H, D]  torch.float8_e4m3fn   (per-TP-rank indexer queries)
    k        [N, D]     torch.float8_e4m3fn   (indexer keys for the whole context)
    k_scale  [N]        torch.float32         (per-KV-token dequant scale)
    weights  [M, H]     torch.float32         (per-query per-head mixing weights)
    ks       [M]        torch.int32           (inclusive start of the valid K window)
    ke       [M]        torch.int32           (exclusive end   of the valid K window)
compute
    logits   [M, N]     torch.float32
    logits[m, n] = sum_h weights[m, h] * relu( dot(q[m, h, :], k[n, :]) * k_scale[n] )
    logits[m, n] = -inf                        for n outside [ks[m], ke[m])

Production dims: H = 32 (64 indexer heads split over TP=2), D = 128.

Registering a new implementation
--------------------------------
Kernel authors import this module and register a callable::

    from harness import gen_inputs, check, bench_impl, register, IMPLS

    def my_kernel(q, k, k_scale, weights, ks, ke):
        ...
        return logits           # [M, N] float32, -inf outside [ks[m], ke[m])

    register("my_kernel", my_kernel)

The callable signature is stable::

    fn(q, k, k_scale, weights, ks, ke) -> logits

    q       : [M, H, D] float8_e4m3fn, contiguous, cuda
    k       : [N, D]    float8_e4m3fn, contiguous, cuda
    k_scale : [N]       float32,       contiguous, cuda
    weights : [M, H]    float32,       contiguous, cuda
    ks, ke  : [M]       int32,         contiguous, cuda
    returns : [M, N]    float32, cuda

The function must not mutate its inputs. It is called repeatedly with the same
input tensors during timing, so it must be idempotent.

CLI
---
    harness.py list
    harness.py check <impl> [--shapes M,N ...]
    harness.py bench <impl> [--out baseline.json] [--append]

Shipped implementations:
    fallback   the production sm_120 chunked-python-loop path, replicated exactly
               (see vllm/model_executor/layers/sparse_attn_indexer.py, sm_120 branch)
    reference  a single unchunked fp8_mqa_logits_torch call (OOMs at large shapes)

Timing methodology
------------------
CUDA events with torch.cuda.synchronize fencing, >= 3 warmup iterations, then a
sustained measurement of max(10 reps, enough reps to cover 200 ms). A per-cell
wall budget (default 120 s) shrinks the rep count for very slow cells; the rep
count actually used is recorded in every result row. Effective TFLOPS counts the
GEMM term only: flops = 2 * M * H * D * N.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from typing import Callable, Dict, List, Tuple

import torch

# fp32 golden must not silently drop to TF32.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")

FP8 = torch.float8_e4m3fn

DEFAULT_H = 32
DEFAULT_D = 128
GRID_M = (256, 1024, 2048, 4096)
GRID_N = (8192, 32768, 102400, 262144, 500000)

LogitsFn = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    torch.Tensor,
]

IMPLS: Dict[str, LogitsFn] = {}


def register(name: str, fn: LogitsFn | None = None):
    """Register an implementation under ``name``.

    Both forms work::

        register("my_kernel", my_kernel)

        @register("my_kernel")
        def my_kernel(q, k, k_scale, weights, ks, ke): ...
    """
    if fn is None:
        def _deco(f: LogitsFn) -> LogitsFn:
            IMPLS[name] = f
            return f

        return _deco
    IMPLS[name] = fn
    return fn


# --------------------------------------------------------------------------- #
# Input generation
# --------------------------------------------------------------------------- #
def gen_inputs(
    M: int,
    N: int,
    H: int = DEFAULT_H,
    D: int = DEFAULT_D,
    device: str | torch.device = "cuda",
    seed: int = 0,
) -> Tuple[torch.Tensor, ...]:
    """Deterministic inputs for shape (M, N, H, D).

    Windows model realistic chunked prefill: the M queries are the LAST M
    positions of an N-token context, so ks[m] = 0 and ke[m] = (N - M) + m + 1.

    Returns (q, k, k_scale, weights, ks, ke).
    """
    if N < M:
        raise ValueError(f"need N >= M for causal-window generation, got M={M}, N={N}")
    device = torch.device(device)
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    q = (torch.randn(M, H, D, generator=g, device=device, dtype=torch.float32) * 0.5).to(FP8)
    k = (torch.randn(N, D, generator=g, device=device, dtype=torch.float32) * 0.5).to(FP8)

    lo, hi = 0.0078, 0.03125
    k_scale = (
        torch.rand(N, generator=g, device=device, dtype=torch.float32) * (hi - lo) + lo
    ).contiguous()

    # positive, magnitude ~ 1/H
    weights = ((torch.rand(M, H, generator=g, device=device, dtype=torch.float32) + 0.5) / H).contiguous()

    ks = torch.zeros(M, device=device, dtype=torch.int32)
    ke = ((N - M) + torch.arange(M, device=device, dtype=torch.int32) + 1).to(torch.int32)
    return q, k, k_scale, weights, ks, ke


# --------------------------------------------------------------------------- #
# Golden reference (fp32 accumulation, memory-safe tiling)
# --------------------------------------------------------------------------- #
def golden(
    q: torch.Tensor,
    k: torch.Tensor,
    k_scale: torch.Tensor,
    weights: torch.Tensor,
    ks: torch.Tensor,
    ke: torch.Tensor,
    m_block: int = 64,
    n_block: int = 8192,
) -> torch.Tensor:
    """Exact fp32 reference, tiled over (M, N) so it survives N = 500k."""
    M, H, D = q.shape
    N = k.shape[0]
    out = torch.empty((M, N), dtype=torch.float32, device=q.device)
    neg_inf = torch.tensor(float("-inf"), device=q.device, dtype=torch.float32)
    w32 = weights.float()

    for n0 in range(0, N, n_block):
        n1 = min(n0 + n_block, N)
        nb = n1 - n0
        kf = k[n0:n1].to(torch.float32)                       # [nb, D]
        sc = k_scale.reshape(-1)[n0:n1].float()               # [nb]
        col = torch.arange(n0, n1, device=q.device, dtype=torch.int32).view(1, nb)
        for m0 in range(0, M, m_block):
            m1 = min(m0 + m_block, M)
            mb = m1 - m0
            qf = q[m0:m1].to(torch.float32)                   # [mb, H, D]
            s = torch.matmul(qf.reshape(mb * H, D), kf.t()).reshape(mb, H, nb)
            s.mul_(sc.view(1, 1, nb)).relu_()
            o = (s * w32[m0:m1].view(mb, H, 1)).sum(dim=1)    # [mb, nb]
            mask = (col >= ks[m0:m1].view(mb, 1)) & (col < ke[m0:m1].view(mb, 1))
            out[m0:m1, n0:n1] = torch.where(mask, o, neg_inf)
            del s, o, qf, mask
        del kf, sc
    return out


# --------------------------------------------------------------------------- #
# Shipped implementations
# --------------------------------------------------------------------------- #
_vllm_ref = None


def _fp8_mqa_logits_torch():
    global _vllm_ref
    if _vllm_ref is None:
        from vllm.v1.attention.ops.rocm_aiter_mla_sparse import fp8_mqa_logits_torch

        _vllm_ref = fp8_mqa_logits_torch
    return _vllm_ref


@register("reference")
def impl_reference(q, k, k_scale, weights, ks, ke):
    """Plain fp8_mqa_logits_torch over the whole chunk; materializes [H, M, N] fp32."""
    return _fp8_mqa_logits_torch()(q, (k, k_scale), weights, ks, ke)


@register("fallback")
def impl_fallback(q, k, k_scale, weights, ks, ke):
    """Production sm_120 path: chunked python loop over query tokens.

    Replicates vllm/model_executor/layers/sparse_attn_indexer.py verbatim,
    including the CH = max(1, min(64, (1 << 26) // (H * N))) chunk formula.
    """
    fp8_mqa_logits_torch = _fp8_mqa_logits_torch()
    _parts = []
    _H = q.shape[1]
    _N = k.shape[0]
    _CH = max(1, min(64, (1 << 26) // max(1, _H * _N)))
    for _i in range(0, q.shape[0], _CH):
        _j = _i + _CH
        _parts.append(
            fp8_mqa_logits_torch(
                q[_i:_j],
                (k, k_scale),
                weights[_i:_j],
                ks[_i:_j],
                ke[_i:_j],
            )
        )
    return _parts[0] if len(_parts) == 1 else torch.cat(_parts)


# --------------------------------------------------------------------------- #
# Correctness
# --------------------------------------------------------------------------- #
def compare(out: torch.Tensor, ref: torch.Tensor, row_block: int = 256) -> Tuple[float, bool, float]:
    """Row-blocked comparison. Returns (max_rel_err, mask_exact, max_abs_err)."""
    M = ref.shape[0]
    max_rel = 0.0
    max_abs = 0.0
    mask_ok = True
    for m0 in range(0, M, row_block):
        m1 = min(m0 + row_block, M)
        r = ref[m0:m1]
        o = out[m0:m1]
        r_inf = torch.isneginf(r)
        o_inf = torch.isneginf(o)
        if not torch.equal(r_inf, o_inf):
            mask_ok = False
        fin = ~r_inf
        d = (o - r).abs()
        d = torch.where(fin, d, torch.zeros((), device=d.device, dtype=d.dtype))
        d = torch.nan_to_num(d, nan=float("inf"))
        rel = d / r.abs().clamp(min=1e-6)
        max_rel = max(max_rel, float(rel.max()))
        max_abs = max(max_abs, float(d.max()))
        del r, o, r_inf, o_inf, fin, d, rel
    return max_rel, mask_ok, max_abs


def check(
    fn: LogitsFn,
    M: int,
    N: int,
    H: int = DEFAULT_H,
    D: int = DEFAULT_D,
    seed: int = 0,
) -> Tuple[float, bool]:
    """Run ``fn`` at (M, N) and compare against the fp32 golden.

    Returns (max_rel_err_over_finite_positions, mask_positions_match_exactly).
    """
    q, k, k_scale, weights, ks, ke = gen_inputs(M, N, H, D, seed=seed)
    ref = golden(q, k, k_scale, weights, ks, ke)
    out = fn(q, k, k_scale, weights, ks, ke)
    torch.cuda.synchronize()
    if out.shape != (M, N):
        raise ValueError(f"impl returned shape {tuple(out.shape)}, expected {(M, N)}")
    max_rel, mask_ok, max_abs = compare(out.float(), ref)
    del q, k, k_scale, weights, ks, ke, ref, out
    torch.cuda.empty_cache()
    print(f"    max_abs_err={max_abs:.3e}")
    return max_rel, mask_ok


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #
def _hw_stamp() -> dict:
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    idx = cvd.split(",")[0] if cvd.strip() else "0"
    stamp = {
        "cuda_visible_devices": cvd,
        "gpu_name": None,
        "sm_clock_mhz": None,
        "mem_clock_mhz": None,
    }
    try:
        q = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                idx,
                "--query-gpu=name,clocks.sm,clocks.mem",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )
        if q.returncode == 0 and q.stdout.strip():
            name, sm, mem = [x.strip() for x in q.stdout.strip().splitlines()[0].split(",")]
            stamp["gpu_name"] = name
            stamp["sm_clock_mhz"] = int(sm)
            stamp["mem_clock_mhz"] = int(mem)
    except Exception:
        pass
    return stamp


def bench_impl(
    fn: LogitsFn,
    M: int,
    N: int,
    H: int = DEFAULT_H,
    D: int = DEFAULT_D,
    warmup: int = 3,
    min_ms: float = 200.0,
    min_reps: int = 10,
    max_seconds: float = 120.0,
    seed: int = 0,
) -> dict:
    """Time ``fn`` at (M, N). Returns a result row dict (see module docstring)."""
    row = {"M": M, "N": N, "H": H, "D": D, "oom": False}
    row.update(_hw_stamp())
    q = k = k_scale = weights = ks = ke = args = out = starts = ends = None

    try:
        q, k, k_scale, weights, ks, ke = gen_inputs(M, N, H, D, seed=seed)
        args = (q, k, k_scale, weights, ks, ke)

        for _ in range(warmup):
            out = fn(*args)
            del out
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        out = fn(*args)
        torch.cuda.synchronize()
        est_s = max(time.perf_counter() - t0, 1e-6)
        del out

        reps = max(min_reps, int(math.ceil((min_ms / 1000.0) / est_s)))
        if reps * est_s > max_seconds:
            reps = max(3, int(max_seconds / est_s))

        starts = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
        torch.cuda.synchronize()
        for i in range(reps):
            starts[i].record()
            out = fn(*args)
            ends[i].record()
            del out
        torch.cuda.synchronize()
        times = [starts[i].elapsed_time(ends[i]) for i in range(reps)]

        # Sample clocks while the GPU is still hot.
        row.update(_hw_stamp())

        t = torch.tensor(times, dtype=torch.float64)
        mean_ms = float(t.mean())
        std_ms = float(t.std(unbiased=False)) if reps > 1 else 0.0
        flops = 2.0 * M * H * D * N
        row.update(
            {
                "reps": reps,
                "warmup": warmup,
                "mean_ms": mean_ms,
                "std_ms": std_ms,
                "min_ms_observed": float(t.min()),
                "tflops": flops / (mean_ms * 1e-3) / 1e12,
            }
        )
    except torch.cuda.OutOfMemoryError as e:
        row["oom"] = True
        row["error"] = str(e).splitlines()[0][:200]
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            row["oom"] = True
            row["error"] = str(e).splitlines()[0][:200]
        else:
            raise
    finally:
        q = k = k_scale = weights = ks = ke = args = out = starts = ends = None
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    return row


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _self_test(M: int = 256, N: int = 8192) -> None:
    print(f"[self-test] vllm fp8_mqa_logits_torch vs fp32 golden at M={M}, N={N}")
    rel, mask_ok = check(impl_reference, M, N)
    print(f"    max_rel_err={rel:.3e}  mask_exact={mask_ok}")
    if not mask_ok or not (rel < 2e-2):
        raise SystemExit("self-test FAILED")
    print("    self-test OK (bf16-level error expected)")


def cmd_check(args) -> None:
    fn = IMPLS[args.impl]
    _self_test()
    shapes = args.shapes or [(256, 8192), (1024, 32768), (512, 262144), (256, 500000)]
    ok = True
    for M, N in shapes:
        print(f"[check {args.impl}] M={M} N={N}")
        rel, mask_ok = check(fn, M, N)
        passed = mask_ok and rel < 2e-2
        ok &= passed
        print(
            f"    max_rel_err={rel:.3e}  mask_exact={mask_ok}  "
            f"{'PASS' if passed else 'FAIL'}"
        )
    if not ok:
        raise SystemExit("check FAILED")
    print("ALL CHECKS PASSED")


def cmd_bench(args) -> None:
    fn = IMPLS[args.impl]
    rows: List[dict] = []
    if args.append and os.path.exists(args.out):
        with open(args.out) as f:
            rows = json.load(f)
    stamp_common = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    print(f"{'impl':>10} {'M':>6} {'N':>7} {'reps':>5} {'mean_ms':>11} {'std_ms':>9} {'TFLOPS':>8}")
    for M in args.M:
        for N in args.N:
            row = bench_impl(
                fn, M, N, H=args.H, D=args.D, max_seconds=args.max_seconds
            )
            row["impl"] = args.impl
            row.update(stamp_common)
            rows.append(row)
            if row["oom"]:
                print(f"{args.impl:>10} {M:>6} {N:>7} {'-':>5} {'OOM':>11} {'-':>9} {'-':>8}")
            else:
                print(
                    f"{args.impl:>10} {M:>6} {N:>7} {row['reps']:>5} "
                    f"{row['mean_ms']:>11.3f} {row['std_ms']:>9.3f} {row['tflops']:>8.2f}"
                )
            with open(args.out, "w") as f:
                json.dump(rows, f, indent=2)
    print(f"wrote {args.out} ({len(rows)} rows)")


def _parse_shape(s: str) -> Tuple[int, int]:
    m, n = s.split(",")
    return int(m), int(n)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list")

    pc = sub.add_parser("check")
    pc.add_argument("impl")
    pc.add_argument("--shapes", type=_parse_shape, nargs="+", default=None, metavar="M,N")

    pb = sub.add_parser("bench")
    pb.add_argument("impl")
    pb.add_argument("--out", default="baseline.json")
    pb.add_argument("--append", action="store_true")
    pb.add_argument("--M", type=int, nargs="+", default=list(GRID_M))
    pb.add_argument("--N", type=int, nargs="+", default=list(GRID_N))
    pb.add_argument("--H", type=int, default=DEFAULT_H)
    pb.add_argument("--D", type=int, default=DEFAULT_D)
    pb.add_argument("--max-seconds", type=float, default=120.0)

    args = p.parse_args()
    if args.cmd == "list":
        for name in IMPLS:
            print(name)
        return
    if args.impl not in IMPLS:
        raise SystemExit(f"unknown impl {args.impl!r}; available: {list(IMPLS)}")
    if args.cmd == "check":
        cmd_check(args)
    else:
        cmd_bench(args)


if __name__ == "__main__":
    main()
