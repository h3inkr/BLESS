#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: bash scripts/run_reasonir_mmlu_eval.sh /path/to/ReasonIR /abs/path/retrieval_file.jsonl /abs/path/raw_queries.jsonl"
  exit 1
fi

REASONIR_REPO="$1"
RETRIEVAL_FILE="$2"
RAW_QUERY_FILE="$3"

if [ ! -d "$REASONIR_REPO" ]; then
  echo "ReasonIR repo not found: $REASONIR_REPO"
  exit 1
fi

if [ ! -f "$RETRIEVAL_FILE" ]; then
  echo "retrieval_file not found: $RETRIEVAL_FILE"
  exit 1
fi

if [ ! -f "$RAW_QUERY_FILE" ]; then
  echo "raw_query_file not found: $RAW_QUERY_FILE"
  exit 1
fi

cd "$REASONIR_REPO"

export retrieval_file="$RETRIEVAL_FILE"
export raw_query_file="$RAW_QUERY_FILE"

bash evaluation/rag/mmlu_cot/scripts/eval_llama_3_8b_mmlu_rag.sh
