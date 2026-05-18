# MMLU Table-2-style Retrieval File Reproduction

This package builds MMLU retrieval files from the public MassiveDS candidate pool.

- Public candidate pool: `rulins/mmlu_searched_results_from_massiveds`
- Query setting: **original query only**
- Output: ReasonIR MMLU RAG evaluator-compatible JSONL
- Supported retrievers:
  - BM25
  - ReasonIR
  - RaDeR or any SentenceTransformer-compatible model
  - Your Tevatron/Qwen-style EOS-pooling checkpoint

## What this package does

This package performs **candidate-pool reranking**:

```text
MMLU original query
→ public MassiveDS candidate pool ctxs
→ BM25 / dense model rerank
→ retrieval_file.jsonl
→ official ReasonIR MMLU RAG evaluator
```

It does not build or search the full MassiveDS index. For MMLU this is practical because the public dataset already contains MassiveDS retrieved candidates.

## 0. Install

```bash
conda create -n mmlu_repro python=3.10 -y
conda activate mmlu_repro

pip install -r requirements.txt
```

For your LoRA checkpoint:

```bash
pip install peft
```

## 1. Inspect candidate pool

```bash
python scripts/00_inspect_mmlu_pool.py \
  --dataset rulins/mmlu_searched_results_from_massiveds \
  --limit 5
```

Expected fields:

```text
query
raw_query
ctxs
```

Each `ctxs` item should include:

```text
retrieval text
id
source
```

## 2. Create BM25 retrieval file

Smoke test:

```bash
python scripts/01_make_mmlu_bm25.py \
  --output retrieval_files/mmlu/debug_bm25_original.jsonl \
  --raw_query_output retrieval_files/mmlu/debug_raw_queries.jsonl \
  --top_k 100 \
  --limit 20
```

Full run:

```bash
python scripts/01_make_mmlu_bm25.py \
  --output retrieval_files/mmlu/bm25_original.jsonl \
  --raw_query_output retrieval_files/mmlu/raw_queries.jsonl \
  --top_k 100
```

## 3. Create ReasonIR retrieval file

Smoke test:

```bash
python scripts/02_make_mmlu_dense.py \
  --backend reasonir \
  --model_name_or_path reasonir/ReasonIR-8B \
  --output retrieval_files/mmlu/debug_reasonir_original.jsonl \
  --raw_query_output retrieval_files/mmlu/debug_raw_queries.jsonl \
  --top_k 100 \
  --doc_batch_size 8 \
  --limit 20
```

Full run:

```bash
python scripts/02_make_mmlu_dense.py \
  --backend reasonir \
  --model_name_or_path reasonir/ReasonIR-8B \
  --output retrieval_files/mmlu/reasonir_original.jsonl \
  --raw_query_output retrieval_files/mmlu/raw_queries.jsonl \
  --top_k 100 \
  --doc_batch_size 8 \
  --cache_dir cache/reasonir_mmlu
```

ReasonIR default query instruction in the script:

```text
Given this reasoning-intensive query, find relevant documents that could help answer the question. 
```

You can override it:

```bash
--query_instruction ""
```

## 4. Create RaDeR retrieval file

Use the exact model path you want to evaluate:

```bash
python scripts/02_make_mmlu_dense.py \
  --backend sentence_transformer \
  --model_name_or_path YOUR_RADER_MODEL_OR_PATH \
  --output retrieval_files/mmlu/rader_original.jsonl \
  --raw_query_output retrieval_files/mmlu/raw_queries.jsonl \
  --top_k 100 \
  --doc_batch_size 32 \
  --normalize
```

If your RaDeR checkpoint is not SentenceTransformer-compatible, replace the encoder backend logic in:

```text
src/mmlu_repro/encoders.py
```

## 5. Create retrieval file for your method

### Merged HF checkpoint

```bash
python scripts/02_make_mmlu_dense.py \
  --backend hf_causal_eos \
  --model_name_or_path /path/to/your/merged/checkpoint \
  --output retrieval_files/mmlu/ours_original.jsonl \
  --raw_query_output retrieval_files/mmlu/raw_queries.jsonl \
  --top_k 100 \
  --doc_batch_size 8 \
  --query_prefix "Query: " \
  --passage_prefix "Passage: " \
  --append_eos \
  --normalize
```

### Base model + LoRA adapter

```bash
python scripts/02_make_mmlu_dense.py \
  --backend hf_causal_eos \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --peft_adapter_path /path/to/your/lora_adapter \
  --output retrieval_files/mmlu/ours_original.jsonl \
  --raw_query_output retrieval_files/mmlu/raw_queries.jsonl \
  --top_k 100 \
  --doc_batch_size 8 \
  --query_prefix "Query: " \
  --passage_prefix "Passage: " \
  --append_eos \
  --normalize
```

This backend assumes your retriever uses:

```text
pooling = eos
append_eos_token = true
normalize = true
```

## 6. Validate retrieval file

```bash
python scripts/03_check_retrieval_file.py \
  --retrieval_file retrieval_files/mmlu/bm25_original.jsonl \
  --min_ctxs 1 \
  --show 2
```

Expected output row:

```json
{
  "raw_query": "The following are multiple choice questions ...",
  "query": "The following are multiple choice questions ...",
  "ctxs": [
    {
      "id": "3652706",
      "source": "math",
      "retrieval text": "...",
      "score": 12.34,
      "rank": 1
    }
  ]
}
```

## 7. Run official ReasonIR MMLU RAG evaluator

Clone official repo:

```bash
git clone https://github.com/facebookresearch/ReasonIR.git
```

Then run:

```bash
bash scripts/run_reasonir_mmlu_eval.sh \
  /path/to/ReasonIR \
  $(pwd)/retrieval_files/mmlu/bm25_original.jsonl \
  $(pwd)/retrieval_files/mmlu/raw_queries.jsonl
```

The wrapper sets:

```bash
export retrieval_file=...
export raw_query_file=...
bash evaluation/rag/mmlu_cot/scripts/eval_llama_3_8b_mmlu_rag.sh
```

## Recommended order

1. `bash scripts/run_smoke_test.sh`
2. Full BM25 retrieval file
3. Official MMLU evaluator with BM25
4. ReasonIR dense smoke test
5. Full ReasonIR retrieval file
6. Ours retrieval file
7. RaDeR retrieval file

## Runtime tip

Run a timed `--limit 100` first. Then estimate full time:

```bash
python scripts/04_estimate_runtime.py \
  --seconds_for_limit 600 \
  --limit 100
```

For example, if 100 queries take 600 seconds, the full 33.5k-query run is roughly 56 hours.

## Important caveats

- This package reproduces the **MMLU candidate-pool reranking** path, not full MassiveDS retrieval.
- BM25 here reranks each query's candidate passages; it is not BM25 over the full MassiveDS corpus.
- Dense reranking can still be slow because the candidate pool is large.
- `--cache_dir` may reduce repeated document encoding if duplicate passages appear across queries.
- Always validate retrieval files before launching the reader LLM evaluation.
