#!/usr/bin/env bash
# Apply all sm_120 patches to a venv's site-packages.
#
#   ./apply.sh /path/to/venv/lib/python3.12/site-packages
#
# Idempotent: --forward skips already-applied hunks. Diffs are against
# vLLM 0.26.1rc1.dev303+g74295e3bd and flashinfer-python 0.6.15.post1;
# a reject (.rej file) means your tree drifted from those versions.
set -euo pipefail

SP="${1:?usage: apply.sh /path/to/venv/lib/python3.12/site-packages}"
[ -d "$SP/vllm" ] || { echo "error: $SP does not contain vllm/" >&2; exit 1; }
[ -d "$SP/flashinfer" ] || { echo "error: $SP does not contain flashinfer/" >&2; exit 1; }

cd "$(dirname "$0")"

for p in vllm/*.patch flashinfer/*.patch; do
  echo "== $p"
  patch -p1 -d "$SP" --forward --no-backup-if-mismatch < "$p" || {
    # exit 1 with --forward means already applied; anything else is real
    rc=$?
    [ $rc -eq 1 ] && echo "   (already applied)" || exit $rc
  }
done

echo "== new files"
cp -v new-files/dsv4_sm120_ops.py "$SP/"
cp -v new-files/dsv4_sm120_prefill.py "$SP/"
cp -v new-files/configs/*.json "$SP/vllm/model_executor/layers/quantization/utils/configs/"

echo "done. Verify: $SP/../..../bin/python -c 'import dsv4_sm120_ops, dsv4_sm120_prefill'"
