#!/usr/bin/env bash
set -euo pipefail

mkdir -p retrieval_files/mmlu logs

# 1. BM25
python scripts/01_make_mmlu_bm25.py \
  --output retrieval_files/mmlu/bm25_original.jsonl \
  --raw_query_output retrieval_files/mmlu/raw_queries.jsonl \
  --top_k 100

# 2. ReasonIR
python scripts/02_make_mmlu_dense.py \
  --backend reasonir \
  --model_name_or_path reasonir/ReasonIR-8B \
  --output retrieval_files/mmlu/reasonir_original.jsonl \
  --raw_query_output retrieval_files/mmlu/raw_queries.jsonl \
  --top_k 100 \
  --doc_batch_size 8 \
  --cache_dir cache/reasonir_mmlu

# 3. RaDeR / ST-compatible
# python scripts/02_make_mmlu_dense.py \
#   --backend sentence_transformer \
#   --model_name_or_path YOUR_RADER_MODEL_OR_PATH \
#   --output retrieval_files/mmlu/rader_original.jsonl \
#   --raw_query_output retrieval_files/mmlu/raw_queries.jsonl \
#   --top_k 100 \
#   --doc_batch_size 32 \
#   --normalize

# 4. Ours: merged HF checkpoint
# python scripts/02_make_mmlu_dense.py \
#   --backend hf_causal_eos \
#   --model_name_or_path /path/to/your/merged/checkpoint \
#   --output retrieval_files/mmlu/ours_original.jsonl \
#   --raw_query_output retrieval_files/mmlu/raw_queries.jsonl \
#   --top_k 100 \
#   --doc_batch_size 8 \
#   --query_prefix "Query: " \
#   --passage_prefix "Passage: " \
#   --append_eos \
#   --normalize
