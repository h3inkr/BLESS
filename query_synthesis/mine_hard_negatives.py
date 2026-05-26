#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hard negative mining for reasoning-aware retrieval training data.

Strategy: BM25 corpus mining
  - Build a BM25 index over all unique pos documents in the corpus
  - For each (query, pos) pair, retrieve top-K documents by BM25
  - Remove the actual positive from results
  - Take the top remaining documents as hard negatives

Fixes over previous version:
  [FIX 1] doc_id now reflects the actual corpus index of the document,
           not the local enumeration index within neg_candidates.
  [FIX 2] Default n_neg raised to 12; top_k raised to 50 to ensure
           enough candidates survive positive-exclusion filtering.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import re
import string
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# ─── IO ───────────────────────────────────────────────────────────────────────

def read_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def write_jsonl(path: str, rows: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ─── tokenization ─────────────────────────────────────────────────────────────

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "for", "from", "with",
    "into", "over", "under", "of", "in", "on", "at", "to", "by", "as",
    "that", "this", "is", "are", "was", "were", "be", "been", "have",
    "has", "had", "do", "does", "did", "not", "no", "so", "we", "it",
}

def tokenize(text: str) -> List[str]:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return [
        t for t in text.split()
        if t and t not in _STOPWORDS and not t.isdigit() and len(t) >= 2
    ]


# ─── BM25 implementation ──────────────────────────────────────────────────────

class BM25:
    """
    Inverted-index BM25.
    Scores only documents containing at least one query term (via inverted index),
    making it O(|candidates|) per query instead of O(|corpus|).
    """

    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
        import math
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.avgdl = sum(len(d) for d in corpus) / max(self.corpus_size, 1)

        self.tf: List[Dict[str, int]] = []
        self.dl: List[int] = []
        self.df: Dict[str, int] = defaultdict(int)
        self.inverted: Dict[str, List[int]] = defaultdict(list)

        for doc_idx, doc in enumerate(corpus):
            freq: Dict[str, int] = defaultdict(int)
            for token in doc:
                freq[token] += 1
            self.tf.append(dict(freq))
            self.dl.append(len(doc))
            for token in set(doc):
                self.df[token] += 1
                self.inverted[token].append(doc_idx)

        self.idf: Dict[str, float] = {}
        for term, df in self.df.items():
            self.idf[term] = math.log(
                (self.corpus_size - df + 0.5) / (df + 0.5) + 1
            )

    def get_top_n(
        self,
        query_tokens: List[str],
        top_n: int = 10,
        max_candidates: int = 50000,
    ) -> List[Tuple[int, float]]:
        candidate_set: Set[int] = set()
        for token in query_tokens:
            if token in self.inverted:
                candidate_set.update(self.inverted[token])
            if len(candidate_set) >= max_candidates:
                break

        if not candidate_set:
            return []

        scores: List[Tuple[int, float]] = []
        for doc_idx in candidate_set:
            dl = self.dl[doc_idx]
            score = 0.0
            for token in query_tokens:
                if token not in self.idf:
                    continue
                tf = self.tf[doc_idx].get(token, 0)
                if tf == 0:
                    continue
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1))
                score += self.idf[token] * numerator / max(denominator, 1e-9)
            scores.append((doc_idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_n]


# ─── corpus building ──────────────────────────────────────────────────────────

def extract_pos_text(row: Dict) -> str:
    passages = row.get("positive_passages")
    if passages and isinstance(passages, list) and len(passages) > 0:
        return passages[0].get("text", "") or ""
    return row.get("pos", "") or ""


def build_corpus(rows: List[Dict]) -> Tuple[List[str], List[List[str]]]:
    seen: Set[str] = set()
    unique_docs: List[str] = []

    for row in rows:
        pos = extract_pos_text(row)
        if not isinstance(pos, str) or not pos.strip():
            continue
        key = re.sub(r"\s+", " ", pos).strip()[:500]
        if key not in seen:
            seen.add(key)
            unique_docs.append(pos)

    tokenized = [tokenize(doc) for doc in unique_docs]
    return unique_docs, tokenized


# ─── Global state set BEFORE fork ─────────────────────────────────────────────

_worker_bm25: BM25 = None
_worker_unique_docs: List[str] = None
_worker_pos_norm_map: Dict[str, int] = None
_worker_top_k: int = 50        # [FIX 2] default raised: need enough candidates after positive exclusion
_worker_n_neg: int = 12        # [FIX 2] default raised to 12
_worker_min_neg_score: float = 0.0
_worker_max_candidates: int = 50000


def _set_globals(
    bm25: BM25,
    unique_docs: List[str],
    pos_norm_map: Dict[str, int],
    top_k: int,
    n_neg: int,
    min_neg_score: float,
    max_candidates: int,
) -> None:
    global _worker_bm25, _worker_unique_docs, _worker_pos_norm_map
    global _worker_top_k, _worker_n_neg, _worker_min_neg_score, _worker_max_candidates
    _worker_bm25 = bm25
    _worker_unique_docs = unique_docs
    _worker_pos_norm_map = pos_norm_map
    _worker_top_k = top_k
    _worker_n_neg = n_neg
    _worker_min_neg_score = min_neg_score
    _worker_max_candidates = max_candidates


# ─── worker ───────────────────────────────────────────────────────────────────

def _process_indexed(args_tuple: Tuple[int, Dict]) -> Tuple[int, Dict, bool]:
    idx, row = args_tuple
    updated_row, found = _process_row(row)
    return idx, updated_row, found


def _process_row(row: Dict) -> Tuple[Dict, bool]:
    query = row.get("query", "")
    pos = extract_pos_text(row)

    if not query or not pos:
        return row, False

    pos_key = re.sub(r"\s+", " ", pos).strip()[:500]
    pos_idx = _worker_pos_norm_map.get(pos_key)

    query_tokens = tokenize(query)
    if not query_tokens:
        return row, False

    top_results = _worker_bm25.get_top_n(
        query_tokens,
        top_n=_worker_top_k,
        max_candidates=_worker_max_candidates,
    )

    neg_candidates = [
        (corpus_idx, score)
        for corpus_idx, score in top_results
        if corpus_idx != pos_idx and score >= _worker_min_neg_score
    ]

    if not neg_candidates:
        return row, False

    # ── [FIX 1] doc_id = actual corpus index, not local enumeration ──────────
    # Previously: {"doc_id": i, ...} where i was enumerate() over neg_candidates
    #             → always 0, 1, 2, ... regardless of actual corpus position
    # Now:        {"doc_id": corpus_idx, ...} reflects the real BM25 corpus index
    # ─────────────────────────────────────────────────────────────────────────
    row["negative_passages"] = [
        {
            "doc_id": corpus_idx,                        # [FIX 1] real corpus index
            "text": _worker_unique_docs[corpus_idx],
            "title": "",
        }
        for corpus_idx, _ in neg_candidates[:_worker_n_neg]   # [FIX 2] up to 12
    ]
    return row, True


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--top_k", type=int, default=50,
                    help="BM25 retrieval depth. Must be > n_neg + 1 (for positive exclusion). "
                         "Default raised to 50.")
    ap.add_argument("--n_neg", type=int, default=12,
                    help="Number of hard negatives per sample. Default 12.")
    ap.add_argument("--min_neg_score", type=float, default=0.0)
    ap.add_argument("--max_candidates", type=int, default=50000)
    ap.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count(),
        help="Worker processes. Defaults to nproc.",
    )
    args = ap.parse_args()

    # ── validate ───────────────────────────────────────────────────────────────
    if args.top_k <= args.n_neg:
        logger.warning(
            f"top_k ({args.top_k}) <= n_neg ({args.n_neg}): "
            "after excluding the positive, you may not have enough candidates. "
            f"Raising top_k to {args.n_neg + 10}."
        )
        args.top_k = args.n_neg + 10

    # ── load data ──────────────────────────────────────────────────────────────
    rows = read_jsonl(args.input)
    logger.info(f"Loaded {len(rows)} rows from {args.input}")

    corpus_rows = rows
    if args.corpus:
        corpus_rows = read_jsonl(args.corpus)
        logger.info(f"Loaded {len(corpus_rows)} corpus rows from {args.corpus}")

    # ── build corpus ───────────────────────────────────────────────────────────
    unique_docs, tokenized_docs = build_corpus(corpus_rows)
    logger.info(f"Corpus: {len(unique_docs)} unique documents")

    pos_norm_map: Dict[str, int] = {}
    for idx, doc in enumerate(unique_docs):
        key = re.sub(r"\s+", " ", doc).strip()[:500]
        pos_norm_map[key] = idx

    # ── build BM25 index ───────────────────────────────────────────────────────
    logger.info("Building BM25 index...")
    bm25 = BM25(tokenized_docs)
    logger.info("BM25 index ready")

    # ── set globals BEFORE Pool() ──────────────────────────────────────────────
    _set_globals(
        bm25,
        unique_docs,
        pos_norm_map,
        args.top_k,
        args.n_neg,
        args.min_neg_score,
        args.max_candidates,
    )

    n_workers = min(args.workers, len(rows))
    logger.info(f"Launching {n_workers} worker processes (nproc={os.cpu_count()})...")
    logger.info(f"top_k={args.top_k} | n_neg={args.n_neg}")

    n_found = 0
    n_skipped = 0
    out_rows = [None] * len(rows)

    chunksize = max(1, len(rows) // (n_workers * 4))
    logger.info(f"chunksize={chunksize}")

    ctx = mp.get_context("fork")
    with ctx.Pool(processes=n_workers) as pool:
        indexed_rows = list(enumerate(rows))

        if HAS_TQDM:
            iterator = tqdm(
                pool.imap_unordered(_process_indexed, indexed_rows, chunksize=chunksize),
                total=len(rows),
                desc="Mining negatives",
                unit="rows",
                dynamic_ncols=True,
            )
        else:
            logger.info("(install tqdm for progress bar: pip install tqdm)")
            iterator = pool.imap_unordered(_process_indexed, indexed_rows, chunksize=chunksize)

        for result in iterator:
            orig_idx, row, found = result
            out_rows[orig_idx] = row
            if found:
                n_found += 1
            else:
                n_skipped += 1

    # ── write output ───────────────────────────────────────────────────────────
    write_jsonl(args.output, out_rows)

    logger.info(
        f"Done. {n_found}/{len(rows)} ({n_found/len(rows)*100:.1f}%) rows "
        f"have hard negatives | skipped={n_skipped} → {args.output}"
    )

    # ── quality check ─────────────────────────────────────────────────────────
    logger.info("\n=== Sample hard negatives (5 examples) ===")
    count = 0
    for row in out_rows:
        neg_passages = row.get("negative_passages", [])
        if not neg_passages:
            continue
        q = row.get("query", "")[:100]
        pos_snip = extract_pos_text(row)[:80]
        logger.info(f"  Q:        {q}")
        logger.info(f"  POS:      {pos_snip}")
        logger.info(f"  #negs:    {len(neg_passages)}")                         # [FIX 2] count check
        logger.info(f"  doc_ids:  {[p['doc_id'] for p in neg_passages]}")       # [FIX 1] id check
        logger.info(f"  NEG[-1]:  {neg_passages[-1].get('text', '')[:80]}")
        logger.info("")
        count += 1
        if count >= 5:
            break


if __name__ == "__main__":
    main()