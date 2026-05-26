#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Wikidata subgraph extractor (MediaWiki API, NOT WDQS/SPARQL)
with word-split-based keyword fallback + concurrent API calls.

Parallelism:
  - Topic level  : topics within one JSONL line processed concurrently
                   (--topic_workers, default 4)
  - Batch level  : neighbor QID chunks fetched concurrently
                   (--batch_workers, default 4)
  - Line level   : multiple JSONL lines processed concurrently
                   (--line_workers, default 1; increase with caution)

Performance fixes applied vs. previous version:
  - sleep removed from normal flow; only on 429/5xx retry
  - str.maketrans / regex pre-compiled as module constants
  - _norm_for_match(query) computed once per search_best_qid call
  - early-exit when neighbor_pairs is empty and min_edges > 0
  - normalize_ws double-call eliminated

USAGE:
  python extract_subgraph_api_wordsplit.py \\
      --input in.jsonl --output out.jsonl --lang en \\
      --props P31,P279,P361,P1269,P366 \\
      --max_topics 5 --max_neighbors 80 --min_edges 1 \\
      --attach_mode augment \\
      --split_budget 8 \\
      --topic_workers 4 --batch_workers 4 --line_workers 1
"""

from __future__ import annotations

import argparse
import json
import re
import string
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

import requests
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Module-level constants  (compiled once)
# ---------------------------------------------------------------------------

API_ENDPOINT = "https://www.wikidata.org/w/api.php"
USER_AGENT   = "wikidata-subgraph-extractor/4.0 (contact: you@example.com)"

DEFAULT_PROPS = [
    "P31",    # instance of         : 개념의 분류
    "P279",   # subclass of         : 상위 개념
    "P361",   # part of             : 전체-부분 관계
    "P527",   # has part            : P361의 역방향 쌍
    "P1269",  # facet of            : 측면/관점 관계
    "P460",   # said to be same as  : 동의어/동일 개념 연결
    "P2579",  # studied in          : 해당 분야에서 연구되는 개념
]

# 역방향 엣지를 수집할 때 확인할 property 집합
REVERSE_CHECK_PROPS = [
    "P279",   # 이웃이 seed의 subclass인지
    "P31",    # 이웃이 seed의 instance인지
    "P361",   # 이웃이 seed의 part인지
    "P527",   # 이웃이 seed를 has part로 포함하는지
    "P1269",  # 이웃이 seed의 facet인지
    "P2579",  # 이웃이 seed 분야에서 연구되는지
]

# Pre-compiled regex patterns for topic extraction
_TOPICS_REGEXES: List[re.Pattern[str]] = [
    re.compile(r'Document\s*Key\s*Topics\\?":\s*"([^"\n]+)"'),
    re.compile(r'Document\s*Key\s*Topics\\?\":\s*"([^"\n]+)"'),
    re.compile(r'Document\s*Key\s*Topics\\?":\s*([^"\n]+)'),
    re.compile(r'Document\s*Key\s*Topics\\?\":\s*([^"\n]+)'),
    re.compile(r'Key\s*Topics\\?\":\s*([^"\n]+)'),
    re.compile(r'Document\s*Key\s*Topics\s*:\s*([^\n]+)'),
]

# Pre-built translation table for punctuation stripping
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)

# Pre-compiled whitespace normalizer
_WS_RE = re.compile(r"\s+")

STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "of", "in", "on", "at", "to", "for",
    "and", "or", "but", "with", "by", "from", "as", "is",
    "are", "be", "been", "being", "was", "were", "that", "this",
    "it", "its", "about", "between", "into", "through", "during",
})


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------

def safe_json_loads(line: str) -> Optional[Dict[str, Any]]:
    line = line.strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


def normalize_ws(s: str) -> str:
    return _WS_RE.sub(" ", s).strip()


def strip_quotes(s: str) -> str:
    return s.strip().strip('"').strip("'")


def _dedup_keep_order(items: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for it in items:
        if it and it not in seen:
            seen.add(it)
            out.append(it)
    return out


def extract_center_topics(keywords_field: Any) -> List[str]:
    """Extract comma-separated topics from obj["keywords"] (str or list[str])."""
    if keywords_field is None:
        return []

    if isinstance(keywords_field, list):
        topics: List[str] = []
        for x in keywords_field:
            if isinstance(x, str):
                for t in x.split(","):
                    normed = normalize_ws(t)
                    if normed:
                        topics.append(normed)
        return _dedup_keep_order(topics)

    if not isinstance(keywords_field, str):
        return []

    payload: Optional[str] = None
    for rgx in _TOPICS_REGEXES:
        m = rgx.search(keywords_field)
        if m:
            payload = m.group(1)
            break
    if payload is None:
        payload = keywords_field

    payload = strip_quotes(payload)
    topics = []
    for t in payload.split(","):
        normed = normalize_ws(t)
        if normed:
            topics.append(normed)
    return _dedup_keep_order(topics)


# ---------------------------------------------------------------------------
# Word-split fallback
# ---------------------------------------------------------------------------

def _tokenize(phrase: str) -> List[str]:
    """Lower-case, strip punctuation, split into tokens."""
    return [w for w in phrase.lower().translate(_PUNCT_TABLE).split() if w]


def word_split_candidates(topic: str, budget: int = 8) -> List[str]:
    """
    Decompose phrase into bigrams -> unigrams (stop-words filtered).

    "triangle area formula"  ->  ["triangle area", "area formula",
                                   "triangle", "area", "formula"]
    "law of cosines"         ->  ["law cosines", "law", "cosines"]
    """
    tokens  = _tokenize(topic)
    content = [t for t in tokens if t not in STOP_WORDS]

    candidates: List[str] = []
    for i in range(len(content) - 1):
        candidates.append(f"{content[i]} {content[i + 1]}")
    for t in content:
        if len(t) >= 3:
            candidates.append(t)

    orig_norm = normalize_ws(topic.lower())
    return [c for c in _dedup_keep_order(candidates) if c != orig_norm][:budget]


def _orig_tokens(topic: str) -> frozenset[str]:
    """원본 토픽의 의미 있는 토큰 집합 (stop words 제거)."""
    return frozenset(
        t for t in _tokenize(topic) if t not in STOP_WORDS and len(t) >= 2
    )


def aligned_split_candidates(topic: str, budget: int = 8) -> List[str]:
    """
    원본 토픽과 align되는 후보만 생성.

    조건: 후보의 모든 토큰이 원본 토픽 토큰의 subset이어야 함.
    → 원본에 없는 단어가 섞인 노이즈 후보를 제거.

    우선순위:
      1. 앞 N단어 슬라이스 (핵심 개념이 앞에 오는 경향)
      2. bigram
      3. 단일 토큰 (3글자 이상)
    """
    orig_tok = _orig_tokens(topic)
    if not orig_tok:
        return []

    content = [t for t in _tokenize(topic) if t not in STOP_WORDS]
    orig_norm = normalize_ws(topic.lower())

    candidates: List[str] = []

    # 1. 앞 N단어 슬라이스 (N=2,3,4)
    for n in (2, 3, 4):
        if len(content) >= n:
            candidates.append(" ".join(content[:n]))

    # 2. bigram
    for i in range(len(content) - 1):
        candidates.append(f"{content[i]} {content[i + 1]}")

    # 3. 단일 토큰
    for t in content:
        if len(t) >= 3:
            candidates.append(t)

    def _is_aligned(cand: str) -> bool:
        cand_tok = frozenset(_tokenize(cand))
        return cand_tok.issubset(orig_tok)

    filtered = [
        c for c in _dedup_keep_order(candidates)
        if c != orig_norm
        and _is_aligned(c)
        and len(c.split()) >= 2  # 단일 단어 후보 제거 — 너무 모호해서 엉뚱한 QID로 매핑될 위험
    ]
    return filtered[:budget]


# ---------------------------------------------------------------------------
# Similarity helpers
# ---------------------------------------------------------------------------

def _norm_for_match(s: str) -> str:
    return _WS_RE.sub(" ", s.lower().translate(_PUNCT_TABLE)).strip()


def _sim_normed(query_norm: str, target: str) -> float:
    """Similarity between a pre-normalised query and a raw target string."""
    t2 = _norm_for_match(target)
    if not query_norm or not t2:
        return 0.0
    return SequenceMatcher(None, query_norm, t2).ratio()


# ---------------------------------------------------------------------------
# Wikidata API client  (one instance per thread is safe)
# ---------------------------------------------------------------------------

class WikidataAPIClient:
    def __init__(
        self,
        lang: str    = "en",
        retries: int = 5,
        timeout: int = 30,
    ) -> None:
        self.lang    = lang
        self.retries = max(0, retries)
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    def _request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        HTTP GET with retry on 429/5xx only.
        No unconditional sleep — network RTT is the natural throttle.
        """
        backoff = 0.5
        last_r  = None
        for _attempt in range(self.retries + 1):
            r = self.session.get(API_ENDPOINT, params=params, timeout=self.timeout)
            last_r = r
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(backoff)
                backoff = min(16.0, backoff * 2)
                continue
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
        assert last_r is not None
        raise RuntimeError(
            f"API failed after {self.retries} retries: "
            f"status={last_r.status_code}, body={last_r.text[:200]}"
        )

    def search(self, text: str, limit: int = 8) -> List[Dict[str, Any]]:
        text = text.strip()
        if not text:
            return []
        data = self._request({
            "action": "wbsearchentities",
            "format": "json",
            "language": self.lang,
            "search": text,
            "limit": max(1, min(50, limit)),
        })
        return data.get("search", []) or []

    def search_best_qid(self, text: str, limit: int = 8) -> Optional[str]:
        """
        match.type 기반 QID 선택.
        label/alias 매칭이 있으면 그 중 첫 번째를 반환.
        없으면 None (similarity fallback 제거).
        """
        hits = self.search(text, limit=limit)
        if not hits:
            return None

        for h in hits:
            match_type = h.get("match", {}).get("type", "")
            if match_type in ("label", "alias"):
                return h.get("id")

        return None  # label/alias 매칭 없으면 실패로 간주

    def get_entities(self, qids: List[str]) -> Dict[str, Any]:
        return self._request({
            "action":    "wbgetentities",
            "format":    "json",
            "languages": self.lang,
            "props":     "labels|descriptions|aliases|claims",
            "ids":       "|".join(qids),
        })

    def get_label_desc_alias(
        self, entity: Dict[str, Any]
    ) -> Tuple[str, str, List[str]]:
        label   = entity.get("labels",       {}).get(self.lang, {}).get("value", "")
        desc    = entity.get("descriptions", {}).get(self.lang, {}).get("value", "")
        aliases = [
            a.get("value", "")
            for a in entity.get("aliases", {}).get(self.lang, [])
        ]
        aliases = _dedup_keep_order([normalize_ws(a) for a in aliases if normalize_ws(a)])
        return label, desc, aliases

    def extract_neighbors(
        self, entity: Dict[str, Any], props: List[str]
    ) -> List[Tuple[str, str]]:
        """(prop, neighbor_qid) pairs for 1-hop wikibase-item neighbors (outgoing)."""
        claims = entity.get("claims", {}) or {}
        out: List[Tuple[str, str]] = []
        for p in props:
            for cl in claims.get(p, []) or []:
                value = cl.get("mainsnak", {}).get("datavalue", {}).get("value", {})
                if isinstance(value, dict) and value.get("entity-type") == "item":
                    qid = value.get("id")
                    if qid:
                        out.append((p, qid))
        return out

    def extract_incoming_edges(
        self,
        seed_qid:          str,
        neighbor_entities: Dict[str, Any],
        rev_props:         List[str],
    ) -> List[Dict[str, str]]:
        """
        이미 fetch된 이웃 entities의 claims를 재활용해서
        seed를 향하는 역방향 엣지를 추출한다.

        추가 API 호출 없이 역방향 엣지를 수집하는 핵심 트릭:
          - 이웃 노드 X의 claims 중 seed_qid를 value로 가지는 것 탐색
          - 발견되면 edge: X --prop--> seed (incoming)

        예시:
          seed = Q11660 (AI)
          이웃 X = Q2539 (머신러닝), X.claims[P279] = [Q11660]
          -> {"src": Q2539, "prop": "P279", "dst": Q11660, "direction": "incoming"}
        """
        incoming: List[Dict[str, str]] = []
        for nbr_qid, nbr_entity in neighbor_entities.items():
            if not nbr_entity or "missing" in nbr_entity:
                continue
            claims = nbr_entity.get("claims", {}) or {}
            for p in rev_props:
                for cl in claims.get(p, []) or []:
                    value = (
                        cl.get("mainsnak", {})
                          .get("datavalue", {})
                          .get("value", {})
                    )
                    if (
                        isinstance(value, dict)
                        and value.get("entity-type") == "item"
                        and value.get("id") == seed_qid
                    ):
                        incoming.append({
                            "src":       nbr_qid,
                            "prop":      p,
                            "dst":       seed_qid,
                            "direction": "incoming",
                        })
        return incoming


# ---------------------------------------------------------------------------
# Single-attempt subgraph build
# ---------------------------------------------------------------------------

def _try_build(
    client:        WikidataAPIClient,
    search_text:   str,
    props:         List[str],
    max_neighbors: int,
    min_edges:     int,
    batch_workers: int = 1,
    rev_props:     Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    One attempt to build a subgraph for *search_text*.

    batch_workers > 1: neighbor QID chunks are fetched concurrently.
    rev_props: 역방향 엣지를 탐색할 property 목록.
               None이면 REVERSE_CHECK_PROPS 사용.
    """
    if rev_props is None:
        rev_props = list(REVERSE_CHECK_PROPS)

    seed_qid = client.search_best_qid(search_text, limit=8)
    if not seed_qid:
        return {"ok": False, "reason": "no_candidate", "search_text": search_text}

    data = client.get_entities([seed_qid])
    seed_entity = data.get("entities", {}).get(seed_qid, {})
    if not seed_entity or "missing" in seed_entity:
        return {"ok": False, "reason": "seed_missing",
                "search_text": search_text, "seed_qid": seed_qid}

    seed_label, seed_desc, seed_aliases = client.get_label_desc_alias(seed_entity)
    neighbor_pairs = client.extract_neighbors(seed_entity, props=props)

    # Early exit: no neighbors and we need at least one edge
    if not neighbor_pairs and min_edges > 0:
        return {
            "ok": False, "reason": "too_few_edges",
            "search_text": search_text, "seed_qid": seed_qid,
            "seed_label": seed_label, "seed_desc": seed_desc,
            "seed_aliases": seed_aliases,
            "nodes": {seed_qid: {"label": seed_label, "description": seed_desc,
                                  "aliases": seed_aliases}},
            "edges": [],
        }

    # De-dup pairs, cap at max_neighbors
    seen_nb: set[Tuple[str, str]] = set()
    neighbor_pairs_dedup: List[Tuple[str, str]] = []
    for pair in neighbor_pairs:
        if pair not in seen_nb:
            seen_nb.add(pair)
            neighbor_pairs_dedup.append(pair)
    neighbor_pairs_dedup = neighbor_pairs_dedup[:max_neighbors]
    neighbor_qids = _dedup_keep_order([q for (_, q) in neighbor_pairs_dedup])

    nodes: Dict[str, Dict[str, Any]] = {
        seed_qid: {"label": seed_label, "description": seed_desc,
                   "aliases": seed_aliases}
    }

    # ── Concurrent neighbor batch fetch ─────────────────────────────────────
    # neighbor entities를 저장해두어 역방향 엣지 탐색에 재활용
    neighbor_entities: Dict[str, Any] = {}

    if neighbor_qids:
        chunk_size = 40
        chunks = [
            neighbor_qids[i: i + chunk_size]
            for i in range(0, len(neighbor_qids), chunk_size)
        ]

        def _fetch_chunk(chunk: List[str]) -> Dict[str, Any]:
            return client.get_entities(chunk).get("entities", {})

        if batch_workers > 1 and len(chunks) > 1:
            with ThreadPoolExecutor(max_workers=batch_workers) as ex:
                for nd in ex.map(_fetch_chunk, chunks):
                    neighbor_entities.update(nd)
        else:
            for chunk in chunks:
                neighbor_entities.update(_fetch_chunk(chunk))

        for qid, ent in neighbor_entities.items():
            if ent and "missing" not in ent:
                lbl, dsc, als = client.get_label_desc_alias(ent)
                nodes[qid] = {"label": lbl, "description": dsc, "aliases": als}
    # ────────────────────────────────────────────────────────────────────────

    # outgoing edges: seed → neighbor
    outgoing_edges: List[Dict[str, str]] = [
        {"src": seed_qid, "prop": p, "dst": q, "direction": "outgoing"}
        for p, q in neighbor_pairs_dedup
        if q in nodes
    ]

    # incoming edges: neighbor → seed (추가 API 호출 없이 이웃 claims 재활용)
    incoming_edges: List[Dict[str, str]] = client.extract_incoming_edges(
        seed_qid=seed_qid,
        neighbor_entities=neighbor_entities,
        rev_props=rev_props,
    )

    edges = outgoing_edges + incoming_edges

    ok = len(edges) >= min_edges
    return {
        "ok": ok, "reason": "ok" if ok else "too_few_edges",
        "search_text": search_text,
        "seed_qid": seed_qid, "seed_label": seed_label,
        "seed_desc": seed_desc, "seed_aliases": seed_aliases,
        "nodes": nodes, "edges": edges,
    }


# ---------------------------------------------------------------------------
# Subgraph builder with word-split fallback
# ---------------------------------------------------------------------------

def build_subgraph_for_topic(
    client:        WikidataAPIClient,
    topic:         str,
    props:         List[str],
    max_neighbors: int = 80,
    min_edges:     int = 1,
    split_budget:  int = 8,
    batch_workers: int = 1,
    rev_props:     Optional[List[str]] = None,
) -> Dict[str, Any]:
    attempts: List[Dict[str, Any]] = []

    def _record(t: Dict[str, Any]) -> None:
        attempts.append({
            "search_text": t.get("search_text"),
            "reason":      t.get("reason"),
            "seed_qid":    t.get("seed_qid"),
        })

    def _pack(t: Dict[str, Any], *, refined: bool) -> Dict[str, Any]:
        return {
            "topic": t["search_text"],
            "seed":  {"qid": t["seed_qid"], "label": t["seed_label"],
                      "description": t["seed_desc"]},
            "nodes": t["nodes"],
            "edges": t["edges"],
            "meta":  {
                "status": "ok", "refined": refined,
                "refine_method":  "word_split" if refined else None,
                "original_topic": topic,
                "chosen_topic":   t["search_text"],
                "attempts":       attempts,
                "props_used":     props,
                "num_nodes":      len(t["nodes"]),
                "num_edges":      len(t["edges"]),
            },
        }

    kw = dict(props=props, max_neighbors=max_neighbors,
              min_edges=min_edges, batch_workers=batch_workers,
              rev_props=rev_props)

    # 1. Original phrase
    first = _try_build(client, topic, **kw)
    _record(first)
    if first["ok"]:
        return _pack(first, refined=False)

    # 2. Aligned word-split candidates
    split_cands = aligned_split_candidates(topic, budget=split_budget)
    for cand in split_cands:
        trial = _try_build(client, cand, **kw)
        _record(trial)
        if trial["ok"]:
            return _pack(trial, refined=True)

    # 3. All failed
    return {
        "topic": topic, "seed": None, "nodes": {}, "edges": [],
        "meta": {
            "status": first.get("reason", "fail"),
            "refined": False,
            "refine_method": "word_split",
            "original_topic": topic,
            "chosen_topic": None,
            "split_candidates_tried": split_cands,
            "attempts": attempts,
        },
    }


# ---------------------------------------------------------------------------
# Per-line processor
# ---------------------------------------------------------------------------

def process_line(
    obj:           Dict[str, Any],
    client:        WikidataAPIClient,
    props:         List[str],
    max_topics:    int,
    max_neighbors: int,
    min_edges:     int,
    split_budget:  int,
    topic_workers: int,
    batch_workers: int,
    attach_mode:   str,
    rev_props:     Optional[List[str]] = None,
) -> str:
    topics = extract_center_topics(obj.get("keywords", ""))
    topics = topics[:max(0, max_topics)]

    if not topics:
        subgraphs = [{"topic": None, "seed": None, "nodes": {}, "edges": [],
                      "meta": {"status": "no_topic"}}]
    else:
        kw = dict(props=props, max_neighbors=max_neighbors,
                  min_edges=min_edges, split_budget=split_budget,
                  batch_workers=batch_workers, rev_props=rev_props)

        def _build(t: str) -> Dict[str, Any]:
            try:
                return build_subgraph_for_topic(client=client, topic=t, **kw)
            except Exception as e:
                return {"topic": t, "seed": None, "nodes": {}, "edges": [],
                        "meta": {"status": "error", "error": repr(e)}}

        # ── Concurrent topic processing ──────────────────────────────────
        if topic_workers > 1 and len(topics) > 1:
            subgraphs_map: Dict[int, Dict[str, Any]] = {}
            with ThreadPoolExecutor(max_workers=topic_workers) as ex:
                fut_to_idx = {ex.submit(_build, t): i
                              for i, t in enumerate(topics)}
                for fut in as_completed(fut_to_idx):
                    subgraphs_map[fut_to_idx[fut]] = fut.result()
            subgraphs = [subgraphs_map[i] for i in range(len(topics))]
        else:
            subgraphs = [_build(t) for t in topics]
        # ────────────────────────────────────────────────────────────────

    if attach_mode == "augment":
        obj["center_topics"] = topics
        obj["subgraphs"]     = subgraphs
        return json.dumps(obj, ensure_ascii=False)
    return json.dumps({"center_topics": topics, "subgraphs": subgraphs},
                      ensure_ascii=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Wikidata subgraph extractor — word-split fallback + concurrent API."
    )
    ap.add_argument("--input",   required=True)
    ap.add_argument("--output",  required=True)
    ap.add_argument("--lang",    default="en")
    ap.add_argument("--retries", type=int, default=5)
    ap.add_argument("--timeout", type=int, default=30)

    ap.add_argument("--props",         default=",".join(DEFAULT_PROPS))
    ap.add_argument("--rev_props",     default=",".join(REVERSE_CHECK_PROPS),
                    help="역방향 엣지 탐색에 사용할 property (기본: REVERSE_CHECK_PROPS)")
    ap.add_argument("--max_topics",    type=int, default=5)
    ap.add_argument("--max_neighbors", type=int, default=80)
    ap.add_argument("--min_edges",     type=int, default=1)
    ap.add_argument("--split_budget",  type=int, default=8)
    ap.add_argument("--attach_mode",
                    choices=["augment", "subgraph_only"], default="augment")

    # ── Concurrency knobs ────────────────────────────────────────────────────
    ap.add_argument("--topic_workers", type=int, default=4,
                    help="Threads per line for parallel topic processing (default 4)")
    ap.add_argument("--batch_workers", type=int, default=4,
                    help="Threads for parallel neighbor-chunk fetching (default 4)")
    ap.add_argument("--line_workers",  type=int, default=1,
                    help="Parallel JSONL lines; each gets its own HTTP session (default 1)")
    # ────────────────────────────────────────────────────────────────────────
    args = ap.parse_args()

    props = [p.strip() for p in args.props.split(",") if p.strip()]
    props = [p if p.startswith("P") else f"P{p}" for p in props]

    rev_props = [p.strip() for p in args.rev_props.split(",") if p.strip()]
    rev_props = [p if p.startswith("P") else f"P{p}" for p in rev_props]

    def _make_client() -> WikidataAPIClient:
        return WikidataAPIClient(lang=args.lang, retries=args.retries,
                                 timeout=args.timeout)

    process_kw = dict(
        props=props,
        rev_props=rev_props,
        max_topics=args.max_topics,
        max_neighbors=args.max_neighbors,
        min_edges=args.min_edges,
        split_budget=args.split_budget,
        topic_workers=args.topic_workers,
        batch_workers=args.batch_workers,
        attach_mode=args.attach_mode,
    )

    with (
        open(args.input,  "r", encoding="utf-8") as fin,
        open(args.output, "w", encoding="utf-8") as fout,
    ):
        if args.line_workers <= 1:
            # ── Single-threaded line loop ────────────────────────────────
            client = _make_client()
            for line in tqdm(fin, desc="Extracting subgraphs"):
                obj = safe_json_loads(line)
                if obj is None:
                    continue
                fout.write(process_line(obj, client=client, **process_kw) + "\n")

        else:
            # ── Multi-line parallel ──────────────────────────────────────
            # Each worker thread gets its own requests.Session via thread-local
            lines = [safe_json_loads(l) for l in fin]
            lines = [o for o in lines if o is not None]

            _local = threading.local()

            def _get_client() -> WikidataAPIClient:
                if not hasattr(_local, "client"):
                    _local.client = _make_client()
                return _local.client

            results: List[Optional[str]] = [None] * len(lines)

            def _worker(idx: int, obj: Dict[str, Any]) -> Tuple[int, str]:
                return idx, process_line(obj, client=_get_client(), **process_kw)

            with ThreadPoolExecutor(max_workers=args.line_workers) as ex:
                fut_to_idx = {ex.submit(_worker, i, o): i
                              for i, o in enumerate(lines)}
                for fut in tqdm(as_completed(fut_to_idx),
                                total=len(lines), desc="Extracting subgraphs"):
                    idx, result = fut.result()
                    results[idx] = result

            for r in results:
                if r is not None:
                    fout.write(r + "\n")


if __name__ == "__main__":
    main()
