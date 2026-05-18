#!/usr/bin/env bash
set -euo pipefail

mkdir -p retrieval_files/mmlu logs

python scripts/00_inspect_mmlu_pool.py --limit 3

python scripts/01_make_mmlu_bm25.py \
  --output retrieval_files/mmlu/debug_bm25_original.jsonl \
  --raw_query_output retrieval_files/mmlu/debug_raw_queries.jsonl \
  --top_k 100 \
  --limit 20

python scripts/03_check_retrieval_file.py \
  --retrieval_file retrieval_files/mmlu/debug_bm25_original.jsonl \
  --min_ctxs 1 \
  --show 2

echo "Smoke test completed."
