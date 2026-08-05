// Fused DSV4 prefill indexer logits kernel for sm_120 (RTX PRO 6000 Blackwell).
//
//   logits[m,n] = sum_h weights[m,h] * relu( dot(q[m,h,:], k[n,:]) * k_scale[n] )
//   logits[m,n] = -inf for n outside [ks[m], ke[m])
//
// q [M,H,128] e4m3, k [N,128] e4m3, k_scale [N] f32, weights [M,H] f32, out [M,N] f32.
//
// Strategy: CTA owns a [BM x BN] output tile.  The k tile (BN x 128 fp8) is loaded
// once into smem and reused across all H heads.  q tiles (BM x 128 fp8 per head) are
// streamed with a cp.async multistage pipeline.  Per head we run m16n8k32 e4m3 mma
// with f32 accumulate, then fold relu(acc*scale)*w into a persistent fp32 tile.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {

constexpr int D = 128;             // head dim, fixed by the op
constexpr int ROWB = D;            // bytes per smem row
constexpr float NEG_INF = -__builtin_huge_valf();

__device__ __forceinline__ uint32_t sm_addr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void ldm_x4(uint32_t& d0, uint32_t& d1, uint32_t& d2,
                                       uint32_t& d3, uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
               : "r"(a));
}

__device__ __forceinline__ void mma_16_8_32(float* d, const uint32_t* a, const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Same mma but with an implicit zero C operand: saves the 4 accumulator-clearing
// MOVs per tile on the first k-step of every head.
__device__ __forceinline__ void mma_16_8_32_z(float* d, const uint32_t* a, const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(0.0f));
}

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, bool pred) {
  int bytes = pred ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
               :: "r"(dst), "l"(src), "r"(bytes));
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}

template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

struct TrueT { static constexpr bool value = true; };
struct FalseT { static constexpr bool value = false; };

// 128B XOR swizzle: logical 16B chunk c of row r lives at physical chunk c ^ (r & 7).
__device__ __forceinline__ int swz(int row, int chunk) {
  return row * ROWB + ((chunk ^ (row & 7)) << 4);
}

// ---------------------------------------------------------------------------
template <int BM, int BN, int WM, int WN, int STAGES, bool NMAJOR, bool BHOIST = false>
__global__ __launch_bounds__((BM / WM) * (BN / WN) * 32) void fused_logits_kernel(
    const uint8_t* __restrict__ q, const uint8_t* __restrict__ kmat,
    const float* __restrict__ k_scale, const float* __restrict__ weights,
    const int* __restrict__ ks, const int* __restrict__ ke, float* __restrict__ out,
    int M, int N, int H) {
  constexpr int WARPS_M = BM / WM;
  constexpr int WARPS_N = BN / WN;
  constexpr int NWARPS = WARPS_M * WARPS_N;
  constexpr int NTHREADS = NWARPS * 32;
  constexpr int MT = WM / 16;   // m16 tiles per warp
  constexpr int NT = WN / 8;    // n8  tiles per warp
  constexpr int KTILE_B = BN * ROWB;
  constexpr int QTILE_B = BM * ROWB;

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int gid = lane >> 2;    // groupID  0..7
  const int tig = lane & 3;     // thread in group 0..3

  const int wm = warp / WARPS_N;
  const int wn = warp % WARPS_N;
  const int m_base = wm * WM;
  const int n_base = wn * WN;

  const int m0 = (NMAJOR ? blockIdx.y : blockIdx.x) * BM;
  const int n0 = (NMAJOR ? blockIdx.x : blockIdx.y) * BN;

  extern __shared__ __align__(16) uint8_t smem_raw[];
  uint8_t* sK = smem_raw;                                    // BN * 128
  uint8_t* sQ = sK + KTILE_B;                                // STAGES * BM * 128
  float* sW = reinterpret_cast<float*>(sQ + STAGES * QTILE_B);  // BM * (H+1)
  float* sSc = sW + BM * (H + 1);                            // BN
  int* sKs = reinterpret_cast<int*>(sSc + BN);               // BM
  int* sKe = sKs + BM;                                       // BM
  int* sRed = sKe + BM;                                      // 4

  // ---- prologue: metadata ------------------------------------------------
  for (int i = tid; i < BM; i += NTHREADS) {
    int m = m0 + i;
    sKs[i] = (m < M) ? ks[m] : 0;
    sKe[i] = (m < M) ? ke[m] : 0;
  }
  for (int i = tid; i < BN; i += NTHREADS) {
    int n = n0 + i;
    sSc[i] = (n < N) ? k_scale[n] : 0.f;
  }
  {
    const int total = BM * H;
    for (int idx = tid; idx < total; idx += NTHREADS) {
      int r = idx / H;
      int c = idx - r * H;
      int m = m0 + r;
      sW[r * (H + 1) + c] = (m < M) ? weights[(size_t)m * H + c] : 0.f;
    }
  }

  // ---- k tile (resident for all heads) -----------------------------------
  {
    const uint8_t* src = kmat + (size_t)n0 * D;
    constexpr int NCHUNK = BN * 8;
#pragma unroll
    for (int i = 0; i < (NCHUNK + NTHREADS - 1) / NTHREADS; ++i) {
      int cid = tid + i * NTHREADS;
      if (cid < NCHUNK) {
        int r = cid >> 3, c = cid & 7;
        bool ok = (n0 + r) < N;
        cp_async16(sm_addr(sK + swz(r, c)), src + (size_t)r * D + c * 16, ok);
      }
    }
  }
  cp_async_commit();

  // ---- window reduction --------------------------------------------------
  __syncthreads();
  if (tid < 32) {
    int mn = INT_MAX, mx = INT_MIN;
    for (int i = lane; i < BM; i += 32) {
      if (m0 + i < M) { mn = min(mn, sKs[i]); mx = max(mx, sKe[i]); }
    }
    float smin = 0.f;
    for (int i = lane; i < BN; i += 32) smin = fminf(smin, sSc[i]);
#pragma unroll
    for (int o = 16; o; o >>= 1) {
      mn = min(mn, __shfl_xor_sync(0xffffffffu, mn, o));
      mx = max(mx, __shfl_xor_sync(0xffffffffu, mx, o));
      smin = fminf(smin, __shfl_xor_sync(0xffffffffu, smin, o));
    }
    if (lane == 0) { sRed[0] = mn; sRed[1] = mx; sRed[2] = (smin >= 0.f); }
  }
  __syncthreads();
  const bool all_masked = (n0 >= sRed[1]) || (n0 + BN <= sRed[0]);
  // When every k_scale in this column tile is >= 0, relu(x*s) == s*relu(x), so the
  // per-column scale can be hoisted out of the head loop and applied once at the
  // store.  That turns the per-head epilogue from 3 fp32 ops per accumulator
  // element into 2.  Negative scales fall back to the exact in-loop form.
  const bool scale_pos = sRed[2] != 0;

  float outacc[MT][NT][4];
#pragma unroll
  for (int mi = 0; mi < MT; ++mi)
#pragma unroll
    for (int ni = 0; ni < NT; ++ni)
#pragma unroll
      for (int t = 0; t < 4; ++t) outacc[mi][ni][t] = 0.f;

  // The per-column scale is read from smem where it is used rather than held in
  // NT*2 registers: in the fast path it is only needed once, at the store.
  const float* sc_lane = sSc + n_base + 2 * tig;

  if (!all_masked) {
    const int ld_row = (lane & 7) + 8 * ((lane >> 3) & 1);
    const int ld_ch = (lane >> 3) >> 1;
    const int ld_xor = lane & 7;

    // q pipeline prologue
    const uint8_t* qbase = q + ((size_t)m0 * H) * D;
    auto issue_q = [&](int stage, int h) {
      if (h < H) {
        const uint8_t* src = qbase + (size_t)h * D;
        uint8_t* dst = sQ + stage * QTILE_B;
        constexpr int NCHUNK = BM * 8;
#pragma unroll
        for (int i = 0; i < (NCHUNK + NTHREADS - 1) / NTHREADS; ++i) {
          int cid = tid + i * NTHREADS;
          if (cid < NCHUNK) {
            int r = cid >> 3, c = cid & 7;
            bool ok = (m0 + r) < M;
            cp_async16(sm_addr(dst + swz(r, c)),
                       src + (size_t)r * H * D + c * 16, ok);
          }
        }
      }
      cp_async_commit();
    };

#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) issue_q(s, s);
    // pending groups now: [k, q0 .. q(STAGES-2)] -> drain until only the q groups
    // remain, i.e. the k tile is guaranteed resident before any ldmatrix on sK.
    cp_async_wait<STAGES - 1>();

    // The k fragments are invariant across the whole head loop (same sK tile, same
    // warp columns), so with enough registers they are loaded once per CTA instead
    // of once per head: 3x fewer ldmatrix and 3x less smem read traffic.
    uint32_t bfall[BHOIST ? 4 : 1][BHOIST ? NT : 1][2];
    if constexpr (BHOIST) {
      __syncthreads();
#pragma unroll
      for (int kk = 0; kk < 4; ++kk)
#pragma unroll
        for (int nj = 0; nj < NT / 2; ++nj) {
          int row = n_base + nj * 16 + ld_row;
          uint32_t r0, r1, r2, r3;
          ldm_x4(r0, r1, r2, r3,
                 sm_addr(sK + row * ROWB + (((2 * kk + ld_ch) ^ ld_xor) << 4)));
          bfall[kk][2 * nj][0] = r0; bfall[kk][2 * nj][1] = r2;
          bfall[kk][2 * nj + 1][0] = r1; bfall[kk][2 * nj + 1][1] = r3;
        }
    }

    // Column-pair streaming: all A fragments for the head are pulled up front, then
    // each pair of n8 tiles runs its 4 k-steps and folds into `outacc` immediately.
    // That keeps only 4*MT*2 accumulator registers live and lets ptxas overlap the
    // fp32 epilogue of one column pair with the mma of the next.
    auto run_heads = [&](auto spos_tag) {
      constexpr bool SPOS = decltype(spos_tag)::value;
      for (int h = 0; h < H; ++h) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();
        issue_q((h + STAGES - 1) % STAGES, h + STAGES - 1);

        const uint8_t* qb = sQ + (h % STAGES) * QTILE_B;

        if constexpr (BHOIST) {
          float wr[MT][2];
#pragma unroll
          for (int mi = 0; mi < MT; ++mi) {
            wr[mi][0] = sW[(m_base + mi * 16 + gid) * (H + 1) + h];
            wr[mi][1] = sW[(m_base + mi * 16 + gid + 8) * (H + 1) + h];
          }
          // All A fragments for the head issue up front so the mma that consume
          // them are far from the ldmatrix that produce them.
          uint32_t af[4][MT][4];
#pragma unroll
          for (int kk = 0; kk < 4; ++kk)
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
              int row = m_base + mi * 16 + ld_row;
              ldm_x4(af[kk][mi][0], af[kk][mi][1], af[kk][mi][2], af[kk][mi][3],
                     sm_addr(qb + row * ROWB + (((2 * kk + ld_ch) ^ ld_xor) << 4)));
            }
#pragma unroll
          for (int nj = 0; nj < NT / 2; ++nj) {
            float acc[MT][2][4];
#pragma unroll
            for (int kk = 0; kk < 4; ++kk)
#pragma unroll
              for (int mi = 0; mi < MT; ++mi)
#pragma unroll
                for (int t = 0; t < 2; ++t) {
                  if (kk == 0) mma_16_8_32_z(acc[mi][t], af[0][mi], bfall[0][2 * nj + t]);
                  else         mma_16_8_32(acc[mi][t], af[kk][mi], bfall[kk][2 * nj + t]);
                }
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
              const float w0 = wr[mi][0], w1 = wr[mi][1];
#pragma unroll
              for (int t = 0; t < 2; ++t) {
                const int ni = 2 * nj + t;
                if constexpr (SPOS) {
                  outacc[mi][ni][0] = fmaf(w0, fmaxf(acc[mi][t][0], 0.f), outacc[mi][ni][0]);
                  outacc[mi][ni][1] = fmaf(w0, fmaxf(acc[mi][t][1], 0.f), outacc[mi][ni][1]);
                  outacc[mi][ni][2] = fmaf(w1, fmaxf(acc[mi][t][2], 0.f), outacc[mi][ni][2]);
                  outacc[mi][ni][3] = fmaf(w1, fmaxf(acc[mi][t][3], 0.f), outacc[mi][ni][3]);
                } else {
                  const float s0 = sc_lane[ni * 8], s1 = sc_lane[ni * 8 + 1];
                  outacc[mi][ni][0] = fmaf(w0, fmaxf(acc[mi][t][0] * s0, 0.f), outacc[mi][ni][0]);
                  outacc[mi][ni][1] = fmaf(w0, fmaxf(acc[mi][t][1] * s1, 0.f), outacc[mi][ni][1]);
                  outacc[mi][ni][2] = fmaf(w1, fmaxf(acc[mi][t][2] * s0, 0.f), outacc[mi][ni][2]);
                  outacc[mi][ni][3] = fmaf(w1, fmaxf(acc[mi][t][3] * s1, 0.f), outacc[mi][ni][3]);
                }
              }
            }
          }
          continue;
        }

        uint32_t af[4][MT][4];
#pragma unroll
        for (int kk = 0; kk < 4; ++kk)
#pragma unroll
          for (int mi = 0; mi < MT; ++mi) {
            int row = m_base + mi * 16 + ld_row;
            ldm_x4(af[kk][mi][0], af[kk][mi][1], af[kk][mi][2], af[kk][mi][3],
                   sm_addr(qb + row * ROWB + (((2 * kk + ld_ch) ^ ld_xor) << 4)));
          }

        float wr[MT][2];
#pragma unroll
        for (int mi = 0; mi < MT; ++mi) {
          wr[mi][0] = sW[(m_base + mi * 16 + gid) * (H + 1) + h];
          wr[mi][1] = sW[(m_base + mi * 16 + gid + 8) * (H + 1) + h];
        }

#pragma unroll
        for (int nj = 0; nj < NT / 2; ++nj) {
          uint32_t bf[4][2][2];
#pragma unroll
          for (int kk = 0; kk < 4; ++kk) {
            int row = n_base + nj * 16 + ld_row;
            uint32_t r0, r1, r2, r3;
            ldm_x4(r0, r1, r2, r3,
                   sm_addr(sK + row * ROWB + (((2 * kk + ld_ch) ^ ld_xor) << 4)));
            bf[kk][0][0] = r0; bf[kk][0][1] = r2;
            bf[kk][1][0] = r1; bf[kk][1][1] = r3;
          }
          float acc[MT][2][4];
#pragma unroll
          for (int mi = 0; mi < MT; ++mi)
#pragma unroll
            for (int t = 0; t < 2; ++t) {
              mma_16_8_32_z(acc[mi][t], af[0][mi], bf[0][t]);
#pragma unroll
              for (int kk = 1; kk < 4; ++kk)
                mma_16_8_32(acc[mi][t], af[kk][mi], bf[kk][t]);
            }
#pragma unroll
          for (int mi = 0; mi < MT; ++mi)
#pragma unroll
            for (int t = 0; t < 2; ++t) {
              const int ni = 2 * nj + t;
              const float w0 = wr[mi][0], w1 = wr[mi][1];
              if constexpr (SPOS) {
                outacc[mi][ni][0] = fmaf(w0, fmaxf(acc[mi][t][0], 0.f), outacc[mi][ni][0]);
                outacc[mi][ni][1] = fmaf(w0, fmaxf(acc[mi][t][1], 0.f), outacc[mi][ni][1]);
                outacc[mi][ni][2] = fmaf(w1, fmaxf(acc[mi][t][2], 0.f), outacc[mi][ni][2]);
                outacc[mi][ni][3] = fmaf(w1, fmaxf(acc[mi][t][3], 0.f), outacc[mi][ni][3]);
              } else {
                const float s0 = sc_lane[ni * 8], s1 = sc_lane[ni * 8 + 1];
                outacc[mi][ni][0] = fmaf(w0, fmaxf(acc[mi][t][0] * s0, 0.f), outacc[mi][ni][0]);
                outacc[mi][ni][1] = fmaf(w0, fmaxf(acc[mi][t][1] * s1, 0.f), outacc[mi][ni][1]);
                outacc[mi][ni][2] = fmaf(w1, fmaxf(acc[mi][t][2] * s0, 0.f), outacc[mi][ni][2]);
                outacc[mi][ni][3] = fmaf(w1, fmaxf(acc[mi][t][3] * s1, 0.f), outacc[mi][ni][3]);
              }
            }
        }
      }
    };

    if (scale_pos) {
      run_heads(TrueT{});
      // fold the hoisted per-column scale in once
#pragma unroll
      for (int mi = 0; mi < MT; ++mi)
#pragma unroll
        for (int ni = 0; ni < NT; ++ni) {
          const float s0 = sc_lane[ni * 8], s1 = sc_lane[ni * 8 + 1];
          outacc[mi][ni][0] *= s0;
          outacc[mi][ni][1] *= s1;
          outacc[mi][ni][2] *= s0;
          outacc[mi][ni][3] *= s1;
        }
    } else {
      run_heads(FalseT{});
    }
  }

  // ---- store -------------------------------------------------------------
  const bool even_n = ((N & 1) == 0);
#pragma unroll
  for (int mi = 0; mi < MT; ++mi) {
    const int lr0 = m_base + mi * 16 + gid;
    const int lr1 = lr0 + 8;
    const int gr0 = m0 + lr0;
    const int gr1 = m0 + lr1;
    const int ks0 = sKs[lr0], ke0 = sKe[lr0];
    const int ks1 = sKs[lr1], ke1 = sKe[lr1];
#pragma unroll
    for (int ni = 0; ni < NT; ++ni) {
      const int c = n0 + n_base + ni * 8 + 2 * tig;
      float2 v0, v1;
      v0.x = (c     >= ks0 && c     < ke0) ? outacc[mi][ni][0] : NEG_INF;
      v0.y = (c + 1 >= ks0 && c + 1 < ke0) ? outacc[mi][ni][1] : NEG_INF;
      v1.x = (c     >= ks1 && c     < ke1) ? outacc[mi][ni][2] : NEG_INF;
      v1.y = (c + 1 >= ks1 && c + 1 < ke1) ? outacc[mi][ni][3] : NEG_INF;
      // c is always even; the pair store needs (row*N + c) even, so it is only
      // safe to vectorise when N is even.
      if (even_n && c + 1 < N) {
        if (gr0 < M) *reinterpret_cast<float2*>(out + (size_t)gr0 * N + c) = v0;
        if (gr1 < M) *reinterpret_cast<float2*>(out + (size_t)gr1 * N + c) = v1;
      } else {
        if (c < N) {
          if (gr0 < M) out[(size_t)gr0 * N + c] = v0.x;
          if (gr1 < M) out[(size_t)gr1 * N + c] = v1.x;
        }
        if (c + 1 < N) {
          if (gr0 < M) out[(size_t)gr0 * N + c + 1] = v0.y;
          if (gr1 < M) out[(size_t)gr1 * N + c + 1] = v1.y;
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
template <int BM, int BN, int WM, int WN, int STAGES, bool NMAJOR, bool BHOIST = false>
void launch(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& k_scale,
            const torch::Tensor& weights, const torch::Tensor& ks, const torch::Tensor& ke,
            torch::Tensor& out, int M, int N, int H) {
  constexpr int NWARPS = (BM / WM) * (BN / WN);
  constexpr int NTHREADS = NWARPS * 32;
  size_t smem = (size_t)BN * D + (size_t)STAGES * BM * D +
                (size_t)BM * (H + 1) * 4 + (size_t)BN * 4 + (size_t)(2 * BM + 4) * 4;
  smem = (smem + 15) & ~(size_t)15;

  auto kern = fused_logits_kernel<BM, BN, WM, WN, STAGES, NMAJOR, BHOIST>;
  static size_t attr_smem = 0;
  if (attr_smem < smem) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    attr_smem = smem;
  }
  dim3 grid;
  int gm = (M + BM - 1) / BM;
  int gn = (N + BN - 1) / BN;
  if (NMAJOR) grid = dim3(gn, gm); else grid = dim3(gm, gn);
  auto stream = at::cuda::getCurrentCUDAStream();
  kern<<<grid, NTHREADS, smem, stream>>>(
      reinterpret_cast<const uint8_t*>(q.data_ptr()),
      reinterpret_cast<const uint8_t*>(k.data_ptr()), k_scale.data_ptr<float>(),
      weights.data_ptr<float>(), ks.data_ptr<int>(), ke.data_ptr<int>(),
      out.data_ptr<float>(), M, N, H);
}

}  // namespace

torch::Tensor fp8_mqa_logits_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor k_scale,
                                  torch::Tensor weights, torch::Tensor ks, torch::Tensor ke,
                                  int64_t cfg) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda(), "inputs must be cuda");
  TORCH_CHECK(q.dim() == 3 && k.dim() == 2, "bad dims");
  const int M = q.size(0), H = q.size(1), Dq = q.size(2), N = k.size(0);
  TORCH_CHECK(Dq == D, "D must be 128");
  auto out = torch::empty({M, N}, q.options().dtype(torch::kFloat32));

  switch (cfg) {
    case 0: launch<64, 128, 32, 32, 3, true>(q, k, k_scale, weights, ks, ke, out, M, N, H); break;
    case 1: launch<64, 128, 32, 64, 3, true>(q, k, k_scale, weights, ks, ke, out, M, N, H); break;
    case 2: launch<64, 256, 32, 64, 3, true>(q, k, k_scale, weights, ks, ke, out, M, N, H); break;
    case 3: launch<128, 128, 32, 64, 3, true>(q, k, k_scale, weights, ks, ke, out, M, N, H); break;
    case 4: launch<64, 256, 32, 64, 4, true>(q, k, k_scale, weights, ks, ke, out, M, N, H); break;
    case 5: launch<128, 256, 32, 64, 3, true>(q, k, k_scale, weights, ks, ke, out, M, N, H); break;
    case 6: launch<64, 128, 32, 32, 3, false>(q, k, k_scale, weights, ks, ke, out, M, N, H); break;
    case 7: launch<64, 256, 32, 64, 3, false>(q, k, k_scale, weights, ks, ke, out, M, N, H); break;
    case 8: launch<64, 512, 32, 64, 3, true>(q, k, k_scale, weights, ks, ke, out, M, N, H); break;
    case 9: launch<32, 256, 32, 64, 3, true>(q, k, k_scale, weights, ks, ke, out, M, N, H); break;
    case 10: launch<64, 256, 32, 32, 3, true>(q, k, k_scale, weights, ks, ke, out, M, N, H); break;
    case 11: launch<128, 256, 64, 32, 3, true>(q, k, k_scale, weights, ks, ke, out, M, N, H); break;
    case 12: launch<64, 256, 64, 64, 3, true>(q, k, k_scale, weights, ks, ke, out, M, N, H); break;
    case 13: launch<64, 256, 32, 64, 5, true>(q, k, k_scale, weights, ks, ke, out, M, N, H); break;
    case 14: launch<128, 256, 32, 32, 3, true>(q, k, k_scale, weights, ks, ke, out, M, N, H); break;
    case 15: launch<64, 128, 64, 64, 3, true>(q, k, k_scale, weights, ks, ke, out, M, N, H); break;
    case 16: launch<32, 256, 32, 32, 3, true>(q, k, k_scale, weights, ks, ke, out, M, N, H); break;
    case 17: launch<64, 256, 32, 32, 4, true>(q, k, k_scale, weights, ks, ke, out, M, N, H); break;
    case 18: launch<128, 128, 32, 32, 3, true>(q, k, k_scale, weights, ks, ke, out, M, N, H); break;
    case 19: launch<128, 256, 32, 64, 2, true>(q, k, k_scale, weights, ks, ke, out, M, N, H); break;
    case 20: launch<64, 512, 32, 64, 2, true>(q, k, k_scale, weights, ks, ke, out, M, N, H); break;
    case 21: launch<128, 128, 32, 64, 4, true>(q, k, k_scale, weights, ks, ke, out, M, N, H); break;
    case 22: launch<64, 512, 32, 128, 3, true>(q, k, k_scale, weights, ks, ke, out, M, N, H); break;
    case 23: launch<128, 256, 32, 128, 3, true>(q, k, k_scale, weights, ks, ke, out, M, N, H); break;
    case 24: launch<64, 256, 32, 32, 3, true, true>(q, k, k_scale, weights, ks, ke, out, M, N, H); break;
    case 25: launch<64, 256, 32, 64, 3, true, true>(q, k, k_scale, weights, ks, ke, out, M, N, H); break;
    case 26: launch<64, 128, 32, 32, 3, true, true>(q, k, k_scale, weights, ks, ke, out, M, N, H); break;
    case 27: launch<128, 256, 32, 64, 3, true, true>(q, k, k_scale, weights, ks, ke, out, M, N, H); break;
    case 28: launch<64, 512, 32, 64, 3, true, true>(q, k, k_scale, weights, ks, ke, out, M, N, H); break;
    case 29: launch<128, 128, 32, 32, 3, true, true>(q, k, k_scale, weights, ks, ke, out, M, N, H); break;
    case 30: launch<64, 256, 32, 128, 3, true, true>(q, k, k_scale, weights, ks, ke, out, M, N, H); break;
    case 31: launch<64, 128, 32, 64, 3, true, true>(q, k, k_scale, weights, ks, ke, out, M, N, H); break;
    case 32: launch<32, 256, 32, 64, 3, true, true>(q, k, k_scale, weights, ks, ke, out, M, N, H); break;
    case 33: launch<32, 512, 32, 64, 3, true, true>(q, k, k_scale, weights, ks, ke, out, M, N, H); break;
    case 34: launch<64, 256, 32, 64, 4, true, true>(q, k, k_scale, weights, ks, ke, out, M, N, H); break;
    case 35: launch<64, 256, 32, 64, 3, false, true>(q, k, k_scale, weights, ks, ke, out, M, N, H); break;
    case 36: launch<32, 256, 32, 64, 4, true, true>(q, k, k_scale, weights, ks, ke, out, M, N, H); break;
    default: TORCH_CHECK(false, "bad cfg");
  }
  C10_CUDA_CHECK(cudaGetLastError());
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fp8_mqa_logits_cuda", &fp8_mqa_logits_cuda, "fused prefill indexer logits (sm_120)");
}
