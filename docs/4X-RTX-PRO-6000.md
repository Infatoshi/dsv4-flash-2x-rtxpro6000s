# 4x RTX PRO 6000 Blackwell (Server Edition)

Bring-up 2026-08-21 on Verda FIN-01 (`4RTXPRO6000.120V`, 4x 96 GB, sm_120,
CUDA toolkit 13.0, driver 580.126.09). Same checkpoint and same patched
vLLM/flashinfer stack as the 2x workstation recipe. This is not a replacement
for `bench/scoreboard.jsonl`. The 2x numbers stay the contract.

## What transferred

- Patches 1-13 still apply. Copy a known-good `site-packages` tree (anvil
  `~/venvs/vllm-dsv4`) if the pinned nightly wheel is gone — as of 2026-08-21
  `wheels.vllm.ai/nightly` had rotated off `0.26.1rc1.dev303+g74295e3bd`.
- `sm_120` kernel gaps are unchanged: no DeepGEMM, no cutlass c3x fp8 block
  scaled-mm, no TRTLLM MXFP4. Marlin W4A16 is still the MoE path. Weights are
  NVFP4; marlin dequants to bf16 activations. That is not a second quant.
- DSpark k=3 still requires `--max-num-seqs 1` (GOTCHAS #3).

## What changed

| | 2x workstation (validated) | 4x server (this box) |
|---|---|---|
| GPUs | 2x Workstation Edition | 4x Server Edition |
| CUDA toolkit | 13.2 | 13.0 (no 13.2 on the image) |
| TP | 2 | 4 |
| weights / GPU | ~87.8 GB | ~44 GB |
| 256k + dspark KV pool | 418,136 tok | **4,695,313 tok** (17.91x at 262144) |
| OS disk | plenty | 50 GB default; 176 GB weights do not fit |

Verda extra NVMe attaches only while the VM is **shutdown**. Create the
volume, shutdown, `PUT /v1/volumes` action `attach`, start, `mkfs` + mount.
A tmpfs `/workspace` dies on reboot.

`apply.sh` on an already-patched tree is not always a no-op. Offset hunks
on `sparse_attn_indexer.py` applied a second time (file grew 34,944 → 36,599
bytes). Restore that file from the known-good snapshot if `patch` reports
`Hunk succeeded` instead of `Reversed (or previously applied)`.

## Serve flags that were wrong, then fixed

First boot (slow decode): TP=4, `--max-num-seqs 16`, `--max-num-batched-tokens 4096`, **no DSpark**.

Engine log, 1 running request, 10 s windows:

| time | prompt tok/s | generation tok/s |
|---|---|---|
| 11:57:00–11:57:50 | 0 | **72.9** |

That is the non-spec path with graphs mostly not covering the live shape.

Missing vs `scripts/serve_256k_marlin.sh`:

- no `--speculative-config '{"method":"dspark","num_speculative_tokens":3}'`
- `--max-num-seqs 16` (illegal with spec; even without spec it eager-decoded
  shapes the FULL graph did not capture)
- `--max-num-batched-tokens 4096` (256k marlin uses 2048)

Log evidence on that first boot:

- CUDA graphs: `PIECEWISE=7 (largest=32), FULL=5 (largest=16)`. Pool 0.36 GiB.
- Flashinfer: `No tuned config covers sparse_mla_sm120_decode_dsv4` at batch
  16/8/4/1, `falling back to runner=SparseMlaDecodeV3Runner tactic=-1`
  (“perf cliff”). Same miss on the live request at page=256.
- JIT **during inference** (after capture): TileLang `hc_prenorm_gemm_tilelang`,
  Triton `_compute_global_topk_indices_and_lens_kernel`,
  `apply_token_bitmask_inplace_kernel` (tool-grammar bitmask),
  `_prefill_logits_kernel`.

Second boot: same script as marlin except `TP=4`, CUDA 13.0, `max-num-seqs 1`,
batched tokens 2048, DSpark k=3. Draft model loaded (`DSpark draft model
loaded: 100 params`). Health 200.

One-shot warm decode (not the scoreboard harness): 400
`usage.completion_tokens` in 1.703 s wall, 19-token prompt, temperature 0,
`stream_options.include_usage` → **234.8 tok/s**. Do not treat that as a
5-rep mean. Use `bench/bench.py` before comparing to 202.7.

## Multi-user vs spec (unchanged law)

DSpark + mixed batch is still unsafe. Short KV (`kv_len < 512` = padded
topk) eager-decodes next to someone else's prefill, V3Runner `index_select`s
OOB, device-assert kills the engine.

That is not “two autotune tactics in one launch.” The sm120 sparse-MLA
cubins are specialized on `(heads, topk, page)`. Different tuned params =
different cubin. The pooling trick is to **make every decode look like the
instantiated shape**:

1. Clamp/pad indices: `indices = where(indices >= kv_len | indices < 0, -1, indices)`,
   keep `topk_length = kv_len`. The tuned kernel already treats `-1` as
   invalid. V3Runner does not. This is the GOTCHAS #3 fix.
2. Dummy KV pages so every seq has `kv_len >= 512` (two `block_size=256`
   pages). Same cubin, uses pool tokens.
3. Pad batch `T` to a captured FULL size (1/4/8/16 here) with masked dummy
   queries. Stops “shape not in bucket.” Does not fix short KV alone.
4. Do not mix prefill and decode in one step. Host scheduler.

Until 1 ships, spec decode stays `--max-num-seqs 1`. Multi-user = drop DSpark
(`serve_64k_multiuser.sh` pattern, or TP=4 non-spec with 256k — KV pool on
4x is large enough for ~11 concurrent 256k users, measured 3,008,608 tok
without spec).

## Clients

omp talks OpenAI to vLLM (`baseUrl …/v1`, model id `dsv4-flash`, tools on,
temp 0.3 / top_p 0.95, stop `["\n<tool","\n<invoke","\n<bash","\n<skill"]`).
That is how anvil is wired.

Do not point Claude Code `ANTHROPIC_BASE_URL` at this server and expect
Claude. vLLM's `/v1/messages` is 200 and returns `pong`, but Flash then
parrots Claude Code's “final / stop / done” protocol. Use omp
(`verda/dsv4-flash` or `anvil/dsv4-flash`).

vLLM `0.26.1rc1.dev303` also needs `torchvision` (and `torchaudio`) in the
venv or workers die with `No module named 'torchvision'` after weight load.

## Script

`scripts/serve_256k_tp4.sh` — marlin + DSpark k=3 + FULL_AND_PIECEWISE,
TP=4, seqs=1, batched tokens 2048, plus `--reasoning-parser deepseek_v4`.
The reasoning parser returns DeepSeek-V4 thinking as a separate channel
(`delta.reasoning`), so a "thinking" block renders in clients (omp, the
DeepSeek harness) and `content` stays clean — the same contract as the
official DeepSeek API. Override `CUDA_HOME` if the box is 13.2.
