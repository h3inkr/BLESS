#!/usr/bin/env python
"""GPQA candidate pool에 대한 BM25 reranking.

00_search_gpqa_massiveds.py로 생성한 pool JSONL을 받아
01_make_mmlu_bm25.py와 동일한 로직으로 처리.
"""
import argparse
import json
import sys
from pathlib import Path

from rank_bm25 import BM25Okapi
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mmlu_repro.utils import (
    ensure_parent,
    get_ctxs,
    simple_tokenize_cached,
    write_jsonl_line,
)


def get_query_from_pool(ex: dict) -> str:
    """pool JSONL에서 쿼리 추출. query / raw_query / Question 순으로 fallback."""
    result = ex.get("raw_query") or ex.get("query") or ex.get("Question")
    if not result:
        raise ValueError(f"No usable query field. Keys: {list(ex.keys())}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pool",
        required=True,
        help="00_search_gpqa_massiveds.py 출력 JSONL 경로",
    )
    parser.add_argument("--output", required=True, help="BM25 retrieval file 저장 경로")
    parser.add_argument(
        "--raw_query_output",
        default="retrieval_files/gpqa/raw_queries.jsonl",
    )
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--limit", type=int, default=-1)
    args = parser.parse_args()

    ensure_parent(args.output)
    ensure_parent(args.raw_query_output)

    # pool 로드
    pool = []
    with open(args.pool, "r", encoding="utf-8") as f:
        for line in f:
            pool.append(json.loads(line))
    n = len(pool) if args.limit < 0 else min(args.limit, len(pool))
    print(f"Pool size: {len(pool)} | Processing: {n}")

    with open(args.output, "w", encoding="utf-8") as fout, \
         open(args.raw_query_output, "w", encoding="utf-8") as fraw:
        for ex in tqdm(pool[:n], total=n, desc="BM25 reranking (GPQA)"):
            raw_query = get_query_from_pool(ex)
            query = raw_query
            ctxs = get_ctxs(ex)

            if not ctxs:
                write_jsonl_line(fout, {"raw_query": raw_query, "query": query, "ctxs": []})
                write_jsonl_line(fraw, {"query": raw_query})
                continue

            docs = [c["retrieval text"] for c in ctxs]
            bm25 = BM25Okapi([simple_tokenize_cached(d) for d in docs])
            scores = bm25.get_scores(simple_tokenize_cached(query))
            order = sorted(range(len(ctxs)), key=lambda j: scores[j], reverse=True)

            reranked = []
            for rank, j in enumerate(order[:args.top_k], start=1):
                c = dict(ctxs[j])
                c["score"] = float(scores[j])
                c["rank"] = rank
                reranked.append(c)

            write_jsonl_line(fout, {"raw_query": raw_query, "query": query, "ctxs": reranked})
            write_jsonl_line(fraw, {"query": raw_query})

    print(f"Saved retrieval file:  {args.output}")
    print(f"Saved raw-query file:  {args.raw_query_output}")


if __name__ == "__main__":
    main()
