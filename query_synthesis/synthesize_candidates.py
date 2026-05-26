#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Candidate query synthesizer — v7.

Changes from v6:
  - `sample_subgraph_nodes` now returns structured center + neighbors WITH
    their relationship (prop label + direction) to the center node.
    Shape: [{"topic", "center": {label, desc}, "neighbors": [{label, desc, relation}]}]
  - `build_prompt` restructured around two principles:
      1. Relationship-aware: each neighbor is presented with how it relates to
         the hidden center concept, giving the LLM a semantic hook to frame queries.
      2. Multi-node combination: LLM is explicitly instructed to weave 2-3
         neighbor concepts into a single query, producing richer, more specific
         queries than single-node instantiation.
  - Center node is surfaced explicitly as the "hidden concept" (forbidden term),
    separate from the generic leakage_terms list, so the LLM understands WHY
    it cannot use that label.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import time
from typing import Dict, List, Optional, Set, Tuple

from vllm import LLM, SamplingParams

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# ─── constants ────────────────────────────────────────────────────────────────

N_QUERIES = 5

TAXONOMY_PROPS = {"P31", "P279"}
NOISY_PROPS = {"P366"}

PROP_LABEL: Dict[str, str] = {
    "P31":   "instance of",
    "P279":  "subclass of",
    "P361":  "part of",
    "P1269": "facet of",
    "P366":  "has use",
    "P527":  "has part",
    "P1542": "has effect",
    "P828":  "has cause",
    "P2283": "uses",
}

# Human-readable relation phrases for the prompt.
# Outgoing = center → neighbor, Incoming = neighbor → center.
RELATION_PHRASE: Dict[str, Tuple[str, str]] = {
    # prop:   (outgoing_phrase,            incoming_phrase)
    "P31":   ("is an instance of",        "has instance"),
    "P279":  ("is a subclass of",         "has subclass"),
    "P361":  ("is part of",               "includes"),
    "P1269": ("is a facet of",            "has facet"),
    "P366":  ("is used for",              "uses"),
    "P527":  ("has part",                 "is part of"),
    "P1542": ("has effect",               "is an effect of"),
    "P828":  ("has cause",                "causes"),
    "P2283": ("uses",                     "is used by"),
}

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "for", "from", "with",
    "into", "over", "under", "through", "using", "find", "determine",
    "compute", "calculate", "what", "which", "when", "where", "why", "how",
    "does", "do", "can", "could", "would", "is", "are", "was", "were", "be",
    "of", "in", "on", "at", "to", "by", "as", "that", "this", "it", "its",
    "not", "no", "we", "you", "they", "document", "query", "result",
}

_word_re = re.compile(r"[A-Za-z0-9]+")
_json_fence_re = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_query_start_re = re.compile(
    r"^(?:what|which|when|where|why|how|who|is|are|was|were|do|does|did|"
    r"can|could|would|find|determine|compute|calculate|evaluate|identify|"
    r"prove|show|given|suppose|let)\b",
    re.IGNORECASE,
)


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


# ─── text helpers ─────────────────────────────────────────────────────────────

def tokenize(text: str) -> Set[str]:
    return {
        t for t in _word_re.findall((text or "").lower())
        if len(t) >= 3 and t not in STOPWORDS and not t.isdigit()
    }


def normalize(s: str) -> str:
    return "".join(_word_re.findall((s or "").lower()))


def truncate(text: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    last = max(cut.rfind("."), cut.rfind("?"), cut.rfind("!"))
    if last > max_chars * 0.6:
        return cut[: last + 1] + " …"
    return cut.rstrip() + " …"


# ─── leakage terms ────────────────────────────────────────────────────────────

def get_leakage_terms(row: Dict) -> List[str]:
    terms: List[str] = []
    kw = row.get("keywords", "")
    if isinstance(kw, str) and kw.strip():
        tail = kw.split(":")[-1]
        for t in re.split(r"[,\n;]+", tail):
            t = t.strip().strip('"').strip("'")
            if len(t) >= 2:
                terms.append(t)
    for t in row.get("center_topics", []) or []:
        if isinstance(t, str) and t.strip():
            terms.append(t.strip())

    seen: Set[str] = set()
    out: List[str] = []
    for t in terms:
        k = normalize(t)
        if k and k not in seen:
            seen.add(k)
            out.append(t)
    return out


def has_leakage(query: str, leakage_terms: List[str]) -> bool:
    q_norm = normalize(query)
    q_tokens = set(_word_re.findall(query.lower()))
    for term in leakage_terms:
        t = term.strip()
        if not t:
            continue
        t_norm = normalize(t)
        if " " in t or "-" in t:
            if t_norm in q_norm:
                return True
        else:
            if len(t_norm) >= 6 and t.lower() in q_tokens:
                return True
    return False


# ─── per-subgraph node sampling (v7: relationship-aware) ──────────────────────

def sample_subgraph_nodes(
    row: Dict,
    extra_nodes_per_sg: int = 2,
    rng: Optional[random.Random] = None,
) -> List[Dict]:
    """
    For every subgraph, extract:
      - center: the seed node info (label, description)
      - neighbors: sampled non-seed nodes, each annotated with their
        relationship TO the center node (relation phrase + direction arrow).

    The relation annotation is the key v7 addition:
      "weighted mean" via [subclass of ↑] means the center is a superclass —
      so the LLM can reason "I know about weighted mean, what's the more
      general form?" without seeing the center label.

    Returns:
      [
        {
          "topic": str,
          "center": {"label": str, "description": str},
          "neighbors": [
            {
              "label": str,
              "description": str,
              "relation": str,   # e.g. "is a subclass of"
              "arrow": str,      # "→" (outgoing) or "←" (incoming)
            },
            ...
          ]
        },
        ...
      ]
    """
    if rng is None:
        rng = random.Random()

    subgraphs = [sg for sg in (row.get("subgraphs", []) or []) if isinstance(sg, dict)]
    result: List[Dict] = []

    for sg in subgraphs:
        topic = sg.get("topic", "")
        nodes_dict: Dict = sg.get("nodes") or {}
        edges_list: List[Dict] = sg.get("edges") or []
        if not nodes_dict:
            continue

        seed_qid = (sg.get("seed") or {}).get("qid", "")

        # ── Build QID → node info map ─────────────────────────────────────
        node_info: Dict[str, Dict] = {}
        for qid, info in nodes_dict.items():
            if not isinstance(info, dict):
                continue
            label = (info.get("label") or "").strip()
            if not label:
                continue
            node_info[qid] = {
                "label": label,
                "description": (info.get("description") or "").strip(),
            }

        if not node_info:
            continue

        center = node_info.get(seed_qid)
        if center is None:
            # Fall back to first node as center if seed not found
            first_qid = next(iter(node_info))
            center = node_info[first_qid]
            seed_qid = first_qid

        # ── Build neighbor → relation map from edges ──────────────────────
        # Each edge touches the seed; we want to characterize how each
        # neighbor node relates to the center from the center's perspective.
        neighbor_relations: Dict[str, Dict] = {}  # qid → {relation, arrow}

        for edge in edges_list:
            if not isinstance(edge, dict):
                continue
            prop = edge.get("prop", "")
            src = edge.get("src", "")
            dst = edge.get("dst", "")
            direction = edge.get("direction", "outgoing")

            phrases = RELATION_PHRASE.get(prop, (PROP_LABEL.get(prop, prop), PROP_LABEL.get(prop, prop)))

            if src == seed_qid:
                # center → neighbor  (outgoing)
                neighbor_qid = dst
                relation = phrases[0]   # e.g. "is a subclass of"
                arrow = "→"
            elif dst == seed_qid:
                # neighbor → center  (incoming)
                neighbor_qid = src
                relation = phrases[1]   # e.g. "has subclass"
                arrow = "←"
            else:
                continue

            if neighbor_qid != seed_qid and neighbor_qid in node_info:
                # Keep first relation found per neighbor (edges are pre-ranked)
                if neighbor_qid not in neighbor_relations:
                    neighbor_relations[neighbor_qid] = {
                        "relation": relation,
                        "arrow": arrow,
                    }

        # ── Sample neighbors ──────────────────────────────────────────────
        candidate_qids = [q for q in node_info if q != seed_qid]
        k = min(extra_nodes_per_sg, len(candidate_qids))
        sampled_qids = rng.sample(candidate_qids, k) if k > 0 else []

        neighbors: List[Dict] = []
        for qid in sampled_qids:
            info = node_info[qid]
            rel_info = neighbor_relations.get(qid, {"relation": "related to", "arrow": "↔"})
            neighbors.append({
                "label": info["label"],
                "description": info["description"],
                "relation": rel_info["relation"],
                "arrow": rel_info["arrow"],
            })

        if not neighbors:
            continue

        result.append({
            "topic": topic,
            "center": center,
            "neighbors": neighbors,
        })

    return result


# ─── subgraph edge selection (retained from v5/v6 as fallback) ────────────────

def select_edges(row: Dict, max_edges: int = 8) -> Tuple[List[Dict], bool]:
    subgraphs = [sg for sg in (row.get("subgraphs", []) or []) if isinstance(sg, dict)]
    if not subgraphs:
        return [], False

    pos_tokens = tokenize(row.get("pos", ""))
    candidates: List[Tuple[float, Dict]] = []

    for sg in subgraphs:
        nodes = sg.get("nodes") or {}
        for e in (sg.get("edges") or []):
            if not isinstance(e, dict):
                continue
            prop = e.get("prop", "")
            src_id = e.get("src", "")
            dst_id = e.get("dst", "")
            src_label = str((nodes.get(src_id, {}) or {}).get("label", src_id))
            dst_label = str((nodes.get(dst_id, {}) or {}).get("label", dst_id))

            src_tok = tokenize(src_label)
            dst_tok = tokenize(dst_label)
            overlap = len((src_tok | dst_tok) & pos_tokens)

            if prop in TAXONOMY_PROPS and overlap < 2:
                continue
            if prop in NOISY_PROPS and len(tokenize(dst_label) & pos_tokens) < 1:
                continue

            candidates.append((overlap, {
                "prop": prop,
                "src": src_label,
                "dst": dst_label,
                "label": PROP_LABEL.get(prop, prop),
            }))

    if not candidates:
        return [], False

    candidates.sort(key=lambda x: x[0], reverse=True)
    seen_pairs: Set[Tuple[str, str]] = set()
    edges: List[Dict] = []
    for _, e in candidates:
        pair = (normalize(e["src"]), normalize(e["dst"]))
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            edges.append(e)
        if len(edges) >= max_edges:
            break

    return edges, True


# ─── prompt ───────────────────────────────────────────────────────────────────

def build_prompt(
    row: Dict,
    sampled_nodes: List[Dict],
    edges: List[Dict],
    has_useful_edges: bool,
    leakage_terms: List[str],
) -> str:
    """
    v7 prompt: two structural changes from v6.

    1. Relationship-aware presentation:
       Each neighbor node is shown with HOW it relates to the hidden center
       concept. This gives the LLM a semantic scaffold to frame queries from
       the neighbor's perspective without naming the center.

       e.g.  "weighted mean  [center is a subclass of this →]"
             → LLM can ask: "What is the most basic form of average that
               weighted mean and Pythagorean mean are special cases of?"

    2. Multi-node combination instruction:
       LLM is explicitly told to weave 2–3 neighbor concepts into each query,
       producing richer, more specific queries than single-node instantiation.
       This also increases the chance that the query is retrievable ONLY via
       the center-node document.
    """
    pos_excerpt = truncate(row.get("pos", ""), 800)
    concept_names = ", ".join(f'"{t}"' for t in leakage_terms[:6]) if leakage_terms else "none"

    # ── Build knowledge-graph hints section ──────────────────────────────────
    subgraph_section = ""

    if sampled_nodes:
        lines: List[str] = []
        for sg_info in sampled_nodes:
            topic = sg_info["topic"]
            center = sg_info["center"]
            neighbors = sg_info["neighbors"]

            # Mark the center explicitly as the hidden concept
            center_desc = f' — {center["description"]}' if center.get("description") else ""
            lines.append(
                f'  [keyword: "{topic}"]'
                f'  HIDDEN concept (never name this): {center["label"]}{center_desc}'
            )

            # Each neighbor with its directional relation to the center
            for nb in neighbors:
                desc = f' — {nb["description"]}' if nb.get("description") else ""
                # Show the relation from center's perspective so LLM understands
                # whether to approach from sub→super or super→sub direction
                lines.append(
                    f'    • {nb["label"]}{desc}'
                    f'  [hidden concept {nb["arrow"]} {nb["relation"]}]'
                )

        if lines:
            subgraph_section = (
                "\nKnowledge-graph context:\n"
                + "\n".join(lines)
                + "\n"
            )

    elif has_useful_edges and edges:
        # Fallback (v6 behaviour): edge-only hints
        edge_lines = "\n".join(
            f"  {e['src']} --[{e['label']}]--> {e['dst']}"
            for e in edges
        )
        subgraph_section = (
            "\nRelated concepts (may hint at the problem domain):\n"
            + edge_lines
            + "\n"
        )

    return (
        "Return ONLY valid JSON in ONE LINE. No markdown, no explanation.\n\n"
        "Document:\n"
        f"{pos_excerpt}\n"
        f"{subgraph_section}\n"
        f"Task: Generate {N_QUERIES} queries that a person would search "
        "to find this document.\n\n"
        "=== CORE CONSTRAINT — the hidden concept ===\n"
        "The document is about a specific concept that is intentionally hidden above.\n"
        "A real searcher does NOT know its name — that is WHY they search.\n"
        "They know the SURROUNDING concepts (the bullet points above) and their PROBLEM.\n"
        f"  Do NOT use these concept names in any query: {concept_names}\n\n"
        "=== HOW to use the knowledge-graph context ===\n"
        "  1. Read the relation arrow to understand the angle:\n"
        "       center → neighbor  means neighbor is a more specific/applied form.\n"
        "       center ← neighbor  means neighbor is a broader category or container.\n"
        "  2. Combine 2–3 neighbor concepts per query to create a concrete situation.\n"
        "     Single-node queries are too generic — mix neighbors to narrow the scenario.\n"
        "     Example: 'weighted mean' + 'Pythagorean mean' → ask what the most basic\n"
        "     form of average is that these two are special cases of.\n"
        "  3. The relation type hints at the query framing:\n"
        "       'subclass of' → ask for the parent/general concept\n"
        "       'instance of' → ask what category this belongs to\n"
        "       'has use'     → describe the application domain, ask the underlying method\n"
        "       'part of'     → describe the whole, ask about the component\n\n"
        "=== Diversity (cover different angles AND different lengths) ===\n"
        "  - A calculation or procedure problem\n"
        "  - A conceptual or definitional question\n"
        "  - A condition, constraint, or edge case\n"
        "  - An application or real-world scenario\n"
        "  Do NOT just vary numbers — vary what aspect of the problem is asked.\n\n"
        "Query length variety (mix both styles):\n"
        "  - Medium (40–120 chars): direct natural-language question\n"
        "  - Long (150–400 chars): detailed problem description before the question\n"
        "  Aim for roughly half medium, half long.\n\n"
        "Each query: ends with ?, self-contained.\n\n"
        '{"queries":['
        '{"query":"...","type":"analytical|causal|deductive|analogical",'
        '"used_neighbors":["neighbor label 1","neighbor label 2"]},'
        "..."
        "]}\n"
    )


# ─── output parsing ───────────────────────────────────────────────────────────

def parse_output(text: str) -> List[Dict]:
    text = (text or "").strip()
    if not text:
        return []

    m = _json_fence_re.search(text)
    if m:
        text = m.group(1).strip()

    for candidate in [
        text,
        text[text.find("{") : text.rfind("}") + 1] if "{" in text else "",
    ]:
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
            if isinstance(obj, list):
                return [q for q in obj if isinstance(q, dict)]
            if isinstance(obj, dict):
                for key in ("queries", "generated_queries", "candidates", "items"):
                    v = obj.get(key)
                    if isinstance(v, list):
                        return [q for q in v if isinstance(q, dict)]
        except Exception:
            pass

    return []


# ─── query cleanup + dedup ────────────────────────────────────────────────────

def clean_query(text: str) -> Optional[str]:
    q = (text or "").strip().strip("\"'")
    if not q:
        return None

    q = re.sub(r"^(?:query|question|answer)\s*:\s*", "", q, flags=re.IGNORECASE)
    q = re.sub(r"\s+", " ", q).strip()

    if "?" in q:
        q = q.split("?")[0].strip() + "?"

    if len(q.split()) < 5:
        return None

    if not (q.endswith("?") or _query_start_re.match(q)):
        return None

    if not q.endswith("?") and _query_start_re.match(q):
        q = q.rstrip(".,;:") + "?"

    return q


def abstract_structure(q: str) -> str:
    s = q.lower()
    s = re.sub(r"\b\d+\.?\d*\b", "NUM", s)
    s = re.sub(r"\b[a-z]{1,2}\b", "V", s)
    return re.sub(r"\s+", " ", s).strip()


def collect_queries(
    raw: List[Dict],
    leakage_terms: List[str],
    edges: List[Dict],
    sampled_nodes: List[Dict],
    cap: int = N_QUERIES,
) -> List[Dict]:
    """
    v7 addition: record `used_neighbors` alongside `used_edges`.
    `used_neighbors` tracks which neighbor node labels the LLM actually
    combined in each query — useful for downstream analysis of which
    relation types produce the best queries.
    """
    # Build a flat set of all neighbor labels for validation
    all_neighbor_labels: Set[str] = set()
    for sg_info in sampled_nodes:
        for nb in sg_info.get("neighbors", []):
            label = nb.get("label", "")
            if label:
                all_neighbor_labels.add(normalize(label))

    out: List[Dict] = []
    seen_norm: Set[str] = set()
    seen_struct: Set[str] = set()
    rt_count: Dict[str, int] = {}
    rt_cap = max(1, (cap + 1) // 2)

    for item in raw:
        if len(out) >= cap:
            break
        if not isinstance(item, dict):
            continue

        q = clean_query(item.get("query", ""))
        if not q:
            continue
        if has_leakage(q, leakage_terms):
            continue

        nq = normalize(q)
        sq = abstract_structure(q)
        if nq in seen_norm or sq in seen_struct:
            continue

        rt = str(item.get("type") or item.get("reasoning_type") or "analytical").lower()
        if rt not in {"analytical", "causal", "deductive", "analogical"}:
            rt = "analytical"
        if rt_count.get(rt, 0) >= rt_cap:
            continue

        # ── used_edges (v5/v6 compat) ──────────────────────────────────────
        used_ids = (
            item.get("used_edge_ids")
            or item.get("used_edges")
            or item.get("edge_ids")
            or []
        )
        used_edges: List[Dict] = []
        if isinstance(used_ids, list):
            seen_idx: Set[int] = set()
            for idx in used_ids:
                if isinstance(idx, str) and idx.isdigit():
                    idx = int(idx)
                if isinstance(idx, int) and 0 <= idx < len(edges) and idx not in seen_idx:
                    used_edges.append(edges[idx])
                    seen_idx.add(idx)

        # ── used_neighbors (v7 new) ────────────────────────────────────────
        # Validate LLM-reported neighbor labels against the actual sampled set.
        reported: List[str] = item.get("used_neighbors") or []
        used_neighbors: List[str] = []
        if isinstance(reported, list):
            for label in reported:
                if isinstance(label, str) and normalize(label) in all_neighbor_labels:
                    used_neighbors.append(label)

        seen_norm.add(nq)
        seen_struct.add(sq)
        rt_count[rt] = rt_count.get(rt, 0) + 1
        out.append({
            "query": q,
            "reasoning_type": rt,
            "used_edges": used_edges,
            "used_neighbors": used_neighbors,   # v7 new field
            "notes": "",
        })

    return out


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--max_rows", type=int, default=0, help="0 = all")
    ap.add_argument("--temperature", type=float, default=0.85)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--repetition_penalty", type=float, default=1.05)
    ap.add_argument("--max_retries", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--gpu_mem_util", type=float, default=0.92)
    ap.add_argument("--max_num_seqs", type=int, default=256)
    ap.add_argument(
        "--extra_nodes_per_sg", type=int, default=2,
        help="Extra nodes to randomly sample (beyond seed) per subgraph (default: 2)",
    )
    ap.add_argument(
        "--node_sample_seed", type=int, default=None,
        help="Random seed for node sampling (default: None = non-deterministic)",
    )
    args = ap.parse_args()

    rows = read_jsonl(args.input)
    if args.max_rows > 0:
        rows = rows[: args.max_rows]
    logger.info(f"Loaded {len(rows)} rows from {args.input}")

    rng = random.Random(args.node_sample_seed)

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        enable_prefix_caching=True,
        gpu_memory_utilization=args.gpu_mem_util,
        max_num_seqs=args.max_num_seqs,
    )
    sp = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        repetition_penalty=args.repetition_penalty,
    )

    # ── build tasks ──────────────────────────────────────────────────────────
    tasks: List[Dict] = []
    pos_only_count = 0
    node_hint_count = 0
    edge_hint_count = 0

    for row in rows:
        leakage_terms = get_leakage_terms(row)

        sampled_nodes = sample_subgraph_nodes(
            row,
            extra_nodes_per_sg=args.extra_nodes_per_sg,
            rng=rng,
        )

        if sampled_nodes:
            node_hint_count += 1

        edges, has_useful = select_edges(row)
        if has_useful:
            edge_hint_count += 1
        elif not sampled_nodes:
            pos_only_count += 1

        prompt = build_prompt(row, sampled_nodes, edges, has_useful, leakage_terms)

        tasks.append({
            "row": row,
            "prompt": prompt,
            "leakage_terms": leakage_terms,
            "edges": edges,
            "sampled_nodes": sampled_nodes,   # v7: pass through for collect_queries
            "generated_queries": [],
            "done": False,
        })

    logger.info(
        f"Tasks: {len(tasks)} total | "
        f"{node_hint_count} with node hints | "
        f"{edge_hint_count} with edge-only hints | "
        f"{pos_only_count} pos-only"
    )

    # ── retry loop ────────────────────────────────────────────────────────────
    batch_size = max(1, min(args.batch_size, args.max_num_seqs))
    pending = list(range(len(tasks)))
    done_count = 0
    t0 = time.time()

    for attempt in range(args.max_retries + 1):
        if not pending:
            break
        next_pending: List[int] = []
        logger.info(f"[Attempt {attempt}] pending={len(pending)}/{len(tasks)}")

        for i in range(0, len(pending), batch_size):
            batch = pending[i : i + batch_size]
            prompts = [tasks[j]["prompt"] for j in batch]
            outputs = llm.generate(prompts, sp)

            for j, out in zip(batch, outputs):
                text = out.outputs[0].text.strip() if out.outputs else ""
                raw = parse_output(text)

                print("RAW TEXT:", text)
                print("PARSED RAW:", raw)

                queries = collect_queries(
                    raw,
                    tasks[j]["leakage_terms"],
                    tasks[j]["edges"],
                    tasks[j]["sampled_nodes"],   # v7: pass sampled_nodes
                )

                if queries:
                    tasks[j]["generated_queries"] = queries
                    tasks[j]["done"] = True
                    done_count += 1
                elif attempt == args.max_retries:
                    tasks[j]["done"] = True
                    done_count += 1
                else:
                    next_pending.append(j)

            elapsed = time.time() - t0
            speed = done_count / elapsed if elapsed > 0 else 0
            eta = (len(tasks) - done_count) / speed if speed > 0 else 0
            logger.info(
                f"  batch {i // batch_size + 1} | "
                f"done={done_count}/{len(tasks)} ({done_count / len(tasks) * 100:.1f}%) | "
                f"ETA={eta:.0f}s"
            )

        pending = next_pending

    # ── write in input order ──────────────────────────────────────────────────
    out_rows = []
    for task in tasks:
        row = dict(task["row"])
        row["generated_queries"] = task["generated_queries"]
        out_rows.append(row)

    write_jsonl(args.output, out_rows)

    n_success = sum(1 for t in tasks if t["generated_queries"])
    logger.info(
        f"Done. {n_success}/{len(tasks)} ({n_success / len(tasks) * 100:.1f}%) rows "
        f"with generated_queries → {args.output}"
    )


if __name__ == "__main__":
    main()