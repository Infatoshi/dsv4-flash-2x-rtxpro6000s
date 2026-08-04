M is decode_toks_per_s (maximize), measured by: opt/bench.py — fixed prompt, max_tokens=256,
temperature=0, ignore_eos, completion_tokens/elapsed vs localhost:8000, 1 warmup + 5 reps.
C: temperature-0 output text identical to output_baseline.txt (e2e); kernel swaps verified
atol/rtol=1e-2 vs torch reference before any e2e measure. epsilon=0.02, noise_k=2.
