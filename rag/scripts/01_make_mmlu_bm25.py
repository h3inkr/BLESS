#!/usr/bin/env python
"""BM25 reranking over the MassiveDS MMLU candidate pool.

Changes vs. original:
  [FIX] simple_tokenize_cached() 사용: 중복 passage tokenization 제거.
  [OPT] --workers 옵션: 멀티프로세스로 쿼리 병렬 처리.
  [OPT] --chunk_size 옵션: 프로세스별 배치 크기 조절.
"""
import argparse
import multiprocessing as mp
import sys
from functools import partial
from pathlib import Path

from datasets import load_dataset
from rank_bm25 import BM25Okapi
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mmlu_repro.utils import (
    ensure_parent,
    get_ctxs,
    get_query,
    simple_tokenize,
    simple_tokenize_cached,
    write_jsonl_line,
)


# ---------------------------------------------------------------------------
# Worker (멀티프로세스: 자식 프로세스에서 실행)
# ---------------------------------------------------------------------------

def _process_example(args_tuple):
    """단일 예제에 대한 BM25 reranking. Pool.imap용 standalone 함수."""
    ex, top_k = args_tuple
    raw_query = get_query(ex)
    query = raw_query
    ctxs = get_ctxs(ex)

    if not ctxs:
        return {"raw_query": raw_query, "query": query, "ctxs": []}

    docs = [c["retrieval text"] for c in ctxs]
    # 멀티프로세스 환경에서는 캐시가 프로세스별로 분리되므로 simple_tokenize 사용
    bm25 = BM25Okapi([simple_tokenize(d) for d in docs])
    scores = bm25.get_scores(simple_tokenize(query))
    order = sorted(range(len(ctxs)), key=lambda j: scores[j], reverse=True)

    reranked = []
    for rank, j in enumerate(order[:top_k], start=1):
        c = dict(ctxs[j])
        c["score"] = float(scores[j])
        c["rank"] = rank
        reranked.append(c)

    return {"raw_query": raw_query, "query": query, "ctxs": reranked}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="rulins/mmlu_searched_results_from_massiveds")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", default="retrieval_files/mmlu/bm25_original.jsonl")
    parser.add_argument("--raw_query_output", default="retrieval_files/mmlu/raw_queries.jsonl")
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--limit", type=int, default=-1)
    # [OPT] 병렬 처리 옵션
    parser.add_argument("--workers", type=int, default=1,
                        help="병렬 워커 수 (기본 1 = 단일 프로세스). "
                             "CPU 코어 수에 맞게 설정 권장.")
    parser.add_argument("--chunk_size", type=int, default=64,
                        help="Pool.imap chunksize (workers > 1 시 유효).")
    args = parser.parse_args()

    ensure_parent(args.output)
    ensure_parent(args.raw_query_output)

    ds = load_dataset(args.dataset, split=args.split)
    n = len(ds) if args.limit < 0 else min(args.limit, len(ds))

    # 예제 목록 생성 (HF dataset은 슬라이싱이 효율적)
    examples = [ds[i] for i in range(n)]
    job_args = [(ex, args.top_k) for ex in examples]

    with open(args.output, "w", encoding="utf-8") as fout, \
         open(args.raw_query_output, "w", encoding="utf-8") as fraw:

        if args.workers <= 1:
            # 단일 프로세스: simple_tokenize_cached 활용
            for ex in tqdm(examples, total=n, desc="BM25 reranking"):
                raw_query = get_query(ex)
                query = raw_query
                ctxs = get_ctxs(ex)

                if not ctxs:
                    out = {"raw_query": raw_query, "query": query, "ctxs": []}
                else:
                    docs = [c["retrieval text"] for c in ctxs]
                    # [FIX+OPT] cached tokenization: 중복 passage 재처리 방지
                    bm25 = BM25Okapi([simple_tokenize_cached(d) for d in docs])
                    scores = bm25.get_scores(simple_tokenize_cached(query))
                    order = sorted(range(len(ctxs)), key=lambda j: scores[j], reverse=True)

                    reranked = []
                    for rank, j in enumerate(order[:args.top_k], start=1):
                        c = dict(ctxs[j])
                        c["score"] = float(scores[j])
                        c["rank"] = rank
                        reranked.append(c)
                    out = {"raw_query": raw_query, "query": query, "ctxs": reranked}

                write_jsonl_line(fout, out)
                write_jsonl_line(fraw, {"query": raw_query})

        else:
            # [OPT] 멀티프로세스: CPU bound tokenization 병렬화
            ctx = mp.get_context("spawn")
            with ctx.Pool(args.workers) as pool:
                for out in tqdm(
                    pool.imap(_process_example, job_args, chunksize=args.chunk_size),
                    total=n,
                    desc=f"BM25 reranking (workers={args.workers})",
                ):
                    write_jsonl_line(fout, out)
                    write_jsonl_line(fraw, {"query": out["raw_query"]})

    print(f"Saved retrieval file:  {args.output}")
    print(f"Saved raw-query file:  {args.raw_query_output}")


if __name__ == "__main__":
    main()
