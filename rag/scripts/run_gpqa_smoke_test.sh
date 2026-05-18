#!/usr/bin/env bash
# GPQA smoke test: massive-serve 서버가 실행 중이어야 함
set -euo pipefail

mkdir -p retrieval_files/gpqa logs

# 1. MassiveDS 검색 (5문제)
python scripts/00_search_gpqa_massiveds.py \
  --config gpqa_diamond \
  --output retrieval_files/gpqa/debug_pool.jsonl \
  --n_docs 100 \
  --limit 5

# 2. BM25 reranking
python scripts/01_make_gpqa_bm25.py \
  --pool retrieval_files/gpqa/debug_pool.jsonl \
  --output retrieval_files/gpqa/debug_bm25.jsonl \
  --raw_query_output retrieval_files/gpqa/debug_raw_queries.jsonl \
  --top_k 100

# 3. 검증
python scripts/03_check_retrieval_file.py \
  --retrieval_file retrieval_files/gpqa/debug_bm25.jsonl \
  --min_ctxs 1 \
  --show 2

echo "GPQA smoke test completed."
