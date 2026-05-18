#!/usr/bin/env python
"""Dense reranking over the MassiveDS MMLU candidate pool.

Changes vs. original:
  [FIX] cache_prefix에 append_eos 포함 → 캐시 충돌 버그 수정.
  [FIX] encode_docs_with_cache: result에 None 잔존 시 명시적 에러.
  [OPT] --query_batch_size > 1 지원: 쿼리를 묶어서 인코딩.
  [OPT] doc embedding: 쿼리별 재인코딩 대신 전체 unique doc 사전 인코딩 후 재사용.
  [OPT] scores matmul을 float32로 명시 캐스팅 유지.
"""
import argparse
import gc
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mmlu_repro.encoders import EncodeConfig, build_encoder
from mmlu_repro.utils import (
    NumpyDiskCache,
    ensure_parent,
    get_ctxs,
    get_query,
    sha1_text,
    write_jsonl_line,
)


DEFAULT_REASONIR_QUERY_INSTRUCTION = (
    "Given this reasoning-intensive query, find relevant documents that could help answer the question. "
)


# ---------------------------------------------------------------------------
# Doc encoding with disk cache
# ---------------------------------------------------------------------------

def encode_docs_with_cache(
    encoder,
    docs: List[str],
    batch_size: int,
    cache: NumpyDiskCache,
    cache_prefix: str,
) -> torch.Tensor:
    if not cache.enabled():
        return encoder.encode_docs(docs, batch_size=batch_size)

    result: List[Optional[torch.Tensor]] = [None] * len(docs)
    missing_texts: List[str] = []
    missing_indices: List[int] = []
    missing_keys: List[str] = []

    for i, text in enumerate(docs):
        key = sha1_text(cache_prefix + "\nDOC\n" + text)
        cached = cache.get(key)
        if cached is not None:
            result[i] = cached
        else:
            missing_texts.append(text)
            missing_indices.append(i)
            missing_keys.append(key)

    if missing_texts:
        new_embs = encoder.encode_docs(missing_texts, batch_size=batch_size)
        for idx, key, emb in zip(missing_indices, missing_keys, new_embs):
            cache.set(key, emb)
            result[idx] = emb

    # [FIX] None이 남아 있으면 명시적 에러 (조용한 stack 실패 방지)
    none_indices = [i for i, r in enumerate(result) if r is None]
    if none_indices:
        raise RuntimeError(
            f"encode_docs_with_cache: {len(none_indices)} embeddings are None "
            f"after encoding. Indices: {none_indices[:10]}"
        )

    return torch.stack(result, dim=0).float()


# ---------------------------------------------------------------------------
# [OPT] 전체 unique doc 사전 인코딩
# ---------------------------------------------------------------------------

def precompute_all_doc_embeddings(
    ds,
    n: int,
    encoder,
    doc_batch_size: int,
    cache: NumpyDiskCache,
    cache_prefix: str,
) -> Dict[str, torch.Tensor]:
    """모든 unique passage를 한 번만 인코딩해서 {text -> embedding} dict 반환.

    candidate pool 내 중복 passage가 많을수록 효과적.
    메모리가 부족하면 --no_precompute 옵션으로 비활성화.
    """
    print("Pre-collecting unique docs...")
    unique_docs_ordered: List[str] = []
    seen: set = set()
    for i in range(n):
        for c in get_ctxs(ds[i]):
            t = c["retrieval text"]
            if t not in seen:
                seen.add(t)
                unique_docs_ordered.append(t)
    print(f"  unique docs: {len(unique_docs_ordered):,}")

    doc2emb: Dict[str, torch.Tensor] = {}

    # 캐시 히트 먼저 처리
    missing_texts: List[str] = []
    missing_keys: List[str] = []
    for text in unique_docs_ordered:
        key = sha1_text(cache_prefix + "\nDOC\n" + text)
        if cache.enabled():
            cached = cache.get(key)
            if cached is not None:
                doc2emb[text] = cached
                continue
        missing_texts.append(text)
        missing_keys.append(key)

    print(f"  cache hits: {len(doc2emb):,}  |  to encode: {len(missing_texts):,}")

    if missing_texts:
        all_embs = []
        for i in tqdm(range(0, len(missing_texts), doc_batch_size), desc="Encoding docs"):
            batch = missing_texts[i:i + doc_batch_size]
            embs = encoder.encode_docs(batch, batch_size=doc_batch_size)
            all_embs.append(embs)

        all_embs_cat = torch.cat(all_embs, dim=0)
        for text, key, emb in zip(missing_texts, missing_keys, all_embs_cat):
            one = emb.unsqueeze(0)
            if cache.enabled():
                cache.set(key, one)
            doc2emb[text] = one

    return doc2emb


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="rulins/mmlu_searched_results_from_massiveds")
    parser.add_argument("--split", default="train")
    parser.add_argument("--backend", required=True,
                        choices=["reasonir", "sentence_transformer", "st",
                                 "hf_causal_eos", "tevatron", "qwen_eos"])
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--peft_adapter_path", default=None)

    parser.add_argument("--output", required=True)
    parser.add_argument("--raw_query_output", default="retrieval_files/mmlu/raw_queries.jsonl")
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--limit", type=int, default=-1)

    parser.add_argument("--query_batch_size", type=int, default=1)
    parser.add_argument("--doc_batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    # [NEW] device_map: None(단일 GPU 기본), "auto"(멀티 GPU)
    parser.add_argument("--device_map", default=None,
                        help="accelerate device_map. 'auto'로 설정하면 멀티 GPU 자동 배분.")
    parser.add_argument("--torch_dtype", default="auto")
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    parser.add_argument("--no_trust_remote_code", dest="trust_remote_code", action="store_false")

    parser.add_argument("--query_prefix", default="")
    parser.add_argument("--passage_prefix", default="")
    parser.add_argument("--append_eos", action="store_true")
    parser.add_argument("--normalize", action="store_true")

    parser.add_argument("--query_instruction", default=None)
    parser.add_argument("--doc_instruction", default="")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--flush_every", type=int, default=100)
    # [OPT] 사전 인코딩 제어
    parser.add_argument("--no_precompute", action="store_true",
                        help="unique doc 사전 인코딩 비활성화 (메모리 절약, 속도 저하).")
    args = parser.parse_args()

    ensure_parent(args.output)
    ensure_parent(args.raw_query_output)

    if args.query_instruction is None:
        query_instruction = DEFAULT_REASONIR_QUERY_INSTRUCTION if args.backend == "reasonir" else ""
    else:
        query_instruction = args.query_instruction

    cfg = EncodeConfig(
        backend=args.backend,
        model_name_or_path=args.model_name_or_path,
        peft_adapter_path=args.peft_adapter_path,
        device=args.device,
        device_map=args.device_map,   # [NEW]
        torch_dtype=args.torch_dtype,
        max_length=args.max_length,
        query_prefix=args.query_prefix,
        passage_prefix=args.passage_prefix,
        append_eos=args.append_eos,
        normalize=args.normalize,
        query_instruction=query_instruction,
        doc_instruction=args.doc_instruction,
        trust_remote_code=args.trust_remote_code,
    )
    encoder = build_encoder(cfg)
    cache = NumpyDiskCache(args.cache_dir)

    # [FIX] cache_prefix에 append_eos 추가 (기존 누락으로 캐시 충돌 발생)
    cache_prefix = (
        f"{args.backend}:{args.model_name_or_path}:{args.passage_prefix}"
        f":{args.max_length}:{args.normalize}:{args.append_eos}"
    )

    ds = load_dataset(args.dataset, split=args.split)
    n = len(ds) if args.limit < 0 else min(args.limit, len(ds))

    # [OPT] 사전 인코딩: unique doc을 한 번만 encode
    doc2emb: Optional[Dict[str, torch.Tensor]] = None
    if not args.no_precompute:
        doc2emb = precompute_all_doc_embeddings(
            ds=ds,
            n=n,
            encoder=encoder,
            doc_batch_size=args.doc_batch_size,
            cache=cache,
            cache_prefix=cache_prefix,
        )

    with open(args.output, "w", encoding="utf-8") as fout, \
         open(args.raw_query_output, "w", encoding="utf-8") as fraw:

        # [OPT] query_batch_size > 1: 여러 쿼리를 묶어서 인코딩
        query_buffer: List[Tuple[int, str, str, List[dict]]] = []

        def flush_queries(buf: List[Tuple[int, str, str, List[dict]]]):
            if not buf:
                return
            queries = [b[1] for b in buf]
            q_embs = encoder.encode_queries(queries, batch_size=len(queries))
            for (idx, raw_query, query, ctxs), q_emb in zip(buf, q_embs):
                _write_one(fout, fraw, raw_query, query, ctxs, q_emb, idx)

        def _write_one(fout, fraw, raw_query, query, ctxs, q_emb, idx):
            if doc2emb is not None:
                # 사전 인코딩된 임베딩 조회
                d_emb = torch.stack(
                    [doc2emb[c["retrieval text"]] for c in ctxs], dim=0
                ).float().squeeze(1)
            else:
                docs = [c["retrieval text"] for c in ctxs]
                d_emb = encode_docs_with_cache(
                    encoder=encoder,
                    docs=docs,
                    batch_size=args.doc_batch_size,
                    cache=cache,
                    cache_prefix=cache_prefix,
                )

            scores = torch.matmul(d_emb, q_emb.float())
            order = torch.argsort(scores, descending=True).tolist()

            reranked = []
            for rank, j in enumerate(order[:args.top_k], start=1):
                c = dict(ctxs[j])
                c["score"] = float(scores[j])
                c["rank"] = rank
                reranked.append(c)

            write_jsonl_line(fout, {"raw_query": raw_query, "query": query, "ctxs": reranked})
            write_jsonl_line(fraw, {"query": raw_query})

            if args.flush_every > 0 and (idx + 1) % args.flush_every == 0:
                fout.flush()
                fraw.flush()
                if doc2emb is None:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        for i in tqdm(range(n), total=n, desc=f"{args.backend} reranking"):
            ex = ds[i]
            raw_query = get_query(ex)
            query = raw_query
            ctxs = get_ctxs(ex)

            if not ctxs:
                write_jsonl_line(fout, {"raw_query": raw_query, "query": query, "ctxs": []})
                write_jsonl_line(fraw, {"query": raw_query})
                continue

            if args.query_batch_size > 1:
                # [OPT] 쿼리 배치 버퍼링
                query_buffer.append((i, raw_query, query, ctxs))
                if len(query_buffer) >= args.query_batch_size:
                    flush_queries(query_buffer)
                    query_buffer.clear()
            else:
                q_emb = encoder.encode_queries([query], batch_size=1)[0]
                _write_one(fout, fraw, raw_query, query, ctxs, q_emb, i)

        # 남은 버퍼 처리
        if query_buffer:
            flush_queries(query_buffer)

    print(f"Saved retrieval file:  {args.output}")
    print(f"Saved raw-query file:  {args.raw_query_output}")


if __name__ == "__main__":
    main()
