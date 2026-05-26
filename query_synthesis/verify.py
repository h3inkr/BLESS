#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Faster usable-query selector for generated queries.

Speed-up strategy:
- Batch connectivity judging ACROSS many rows (instead of row-by-row tiny batches)
- Deduplicate identical candidates within each row
- Use shorter JSON-only prompt and shorter generation
- Enable vLLM prefix caching
- Avoid expensive fsync on every save by default

Selection policy is preserved:
1) Evaluate candidates by conservative keyword leakage + LLM-based connectivity.
2) Prefer a clean candidate ONLY when it is strongly connected to `pos`
   and not much worse than the best usable candidate.
3) If all candidates are weak, skip selection (`selected_query=None`).
4) Also store top_usable_query and top_clean_query separately.

Order-preserving write strategy:
- Each row gets an incremental `order_idx`.
- Finalized rows are temporarily stored in `finalized_by_order`.
- We only move rows to the write buffer when they become available in exact input order.
- This keeps append order consistent with the original input order.

v2 changes:
- [FIX] Judge retry for parsing failures: when connectivity_score=0 and reason=""
  (indicating a JSON parse failure, not a genuine low score), those candidates are
  flagged for a second LLM judge call. This rescues ~33% of all_candidates_weak cases.
- [NEW] all_good_queries field: all candidates with connectivity_score >= good_save_threshold
  and clean=True are saved together, not just the single best. This improves training
  data volume and query diversity for retrieval model training.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from vllm import LLM, SamplingParams


# -----------------------------
# IO
# -----------------------------
def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_processed_uids(output_path: str) -> Set[str]:
    if not os.path.exists(output_path):
        return set()
    out = set()
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                uid = obj.get("_row_uid")
                if isinstance(uid, str) and uid:
                    out.add(uid)
            except Exception:
                pass
    return out


class JsonlAppender:
    def __init__(self, path: str, do_fsync: bool = False):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        self.do_fsync = do_fsync
        self.f = open(path, "a", encoding="utf-8")

    def write_rows(self, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        for row in rows:
            self.f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.f.flush()
        if self.do_fsync:
            os.fsync(self.f.fileno())

    def close(self):
        try:
            self.f.flush()
            if self.do_fsync:
                os.fsync(self.f.fileno())
        finally:
            self.f.close()


# -----------------------------
# text utils
# -----------------------------
def norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def parse_keywords_field(keywords: Any) -> List[str]:
    """
    Only use the `keywords` field.
    Supports:
      - list[str]
      - "Document Key Topics: a, b, c"
      - "a, b, c"
    """
    if keywords is None:
        return []

    if isinstance(keywords, list):
        vals = [norm(x) for x in keywords if isinstance(x, str) and norm(x)]
    elif isinstance(keywords, str):
        s = keywords.strip()
        if ":" in s:
            s = s.split(":", 1)[1].strip()
        s = s.strip().strip('"').strip("'")
        vals = [norm(x) for x in re.split(r"[,\n;|]+", s) if norm(x)]
    else:
        vals = []

    seen = set()
    out = []
    for v in vals:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def keyword_hits_exact_phrase(query: str, keyword_phrases: List[str]) -> List[str]:
    """
    Conservative leakage detection:
    - Multiword keywords: exact normalized substring match
    - Single-word keywords: exact word-boundary match
    """
    qn = norm(query)
    hits = []
    for kw in keyword_phrases:
        if not kw:
            continue
        if " " in kw:
            if kw in qn:
                hits.append(kw)
        else:
            if re.search(rf"\b{re.escape(kw)}\b", qn):
                hits.append(kw)
    return hits


def keyword_score_from_hits(hits: List[str]) -> int:
    """
    Keep this gentle.
      0 hits -> 100
      1 hit  -> 70
      2 hits -> 40
      3+     -> 10
    """
    n = len(hits)
    if n == 0:
        return 100
    if n == 1:
        return 70
    if n == 2:
        return 40
    return 10


# -----------------------------
# prompt
# -----------------------------
def build_connectivity_prompt(pos: str, candidate_query: str) -> str:
    return (
        "Return ONLY valid JSON in ONE LINE (minified). No markdown, no code fences.\n"
        "You are given:\n"
        "- a document excerpt called pos\n"
        "- a generated query\n\n"
        "Judge whether the generated query preserves RELEVANCE with respect to pos.\n"
        "Important distinction:\n"
        "- High score requires POS-SPECIFIC relevance, not merely being in the same broad math topic.\n"
        "- If many other nearby documents could answer the query just as well, do NOT give a high score.\n"
        "- Reward the candidate only when the key concept, theorem, constraint, derivation, or problem mechanism in pos is genuinely needed or strongly justified for the query.\n\n"
        "Scoring rubric:\n"
        "- 90-100: pos is directly needed to answer/justify the query; the query clearly targets the main concept/theorem/mechanism in pos.\n"
        "- 80-89: pos is strongly and specifically useful; most of the query's core requirement aligns with pos.\n"
        "- 60-79: meaningful relevance is preserved, but the query could also fit several nearby documents with similar methods/concepts.\n"
        "- 40-59: only partial relevance; some concept or method overlap exists, but pos is not specifically targeted.\n"
        "- 20-39: weak relation; mostly broad topical overlap.\n"
        "- 0-19: essentially unrelated.\n\n"
        "Important:\n"
        "- Do NOT compare against any original query.\n"
        "- Do NOT score by superficial word overlap.\n"
        "- Focus on whether the positive passage itself is a good supervision target for this candidate.\n"
        "- Broad topic similarity alone should not exceed 59.\n"
        "- If the candidate mainly matches the subject area but not the positive passage specifically, keep the score below 60.\n\n"
        "Output schema EXACTLY:\n"
        "{\"connectivity_score\":<int 0..100>,\"confidence\":<float 0..1>,\"label\":\"high|medium|low\",\"reason\":\"...\"}\n\n"
        f"pos:\n{pos}\n\n"
        f"generated_query:\n{candidate_query}\n"
    )


def safe_parse_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


def is_judge_parse_failure(parsed: Dict[str, Any]) -> bool:
    """
    [v2 NEW] Detect whether a parsed judge output represents a genuine score=0
    vs a JSON parsing failure that defaulted to 0.

    A parse failure is indicated by:
    - connectivity_score == 0 AND reason is empty/missing
    An intentional score of 0 would always include a reason.
    """
    if not parsed:
        return True
    conn = parsed.get("connectivity_score")
    reason = parsed.get("reason", "")
    if conn == 0 and (not reason or not str(reason).strip()):
        return True
    return False


def coerce_int_0_100(value: Any, default: int = 0) -> int:
    try:
        if isinstance(value, bool):
            raise ValueError("bool is not a valid score")
        if isinstance(value, int):
            num = value
        elif isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("non-finite float")
            num = int(round(value))
        elif isinstance(value, str):
            s = value.strip()
            if not s:
                raise ValueError("empty string")
            num = int(round(float(s)))
        else:
            raise ValueError("unsupported type")
    except Exception:
        num = default
    return max(0, min(100, int(num)))


def coerce_float_0_1(value: Any, default: float = 0.5) -> float:
    try:
        if isinstance(value, bool):
            raise ValueError("bool is not a valid score")
        if isinstance(value, (int, float)):
            num = float(value)
        elif isinstance(value, str):
            s = value.strip()
            if not s:
                raise ValueError("empty string")
            num = float(s)
        else:
            raise ValueError("unsupported type")
        if not math.isfinite(num):
            raise ValueError("non-finite number")
    except Exception:
        num = default
    return float(max(0.0, min(1.0, num)))


# -----------------------------
# data models
# -----------------------------
@dataclass
class CandEval:
    candidate_id: int
    query: str
    keyword_hits: List[str]
    keyword_score: int
    connectivity_score: int
    confidence: float
    label: str
    reason: str
    final_score: float
    clean: bool
    judge_failed: bool = False  # [v2] True if parse failure, not genuine score


@dataclass
class CandidateMeta:
    candidate_id: int
    query: str
    keyword_hits: List[str]
    keyword_score: int
    prompt: str


@dataclass
class RowState:
    order_idx: int
    out: Dict[str, Any]
    candidates: List[CandidateMeta]
    # [v2] track which candidate indices need judge retry
    retry_indices: List[int] = field(default_factory=list)
    parsed_outputs: List[Dict[str, Any]] = field(default_factory=list)


# -----------------------------
# candidate extraction
# -----------------------------
def extract_candidates_dedup(
    generated_queries: Any,
    max_candidates: int,
    dedup: bool = True,
) -> List[Tuple[int, str]]:
    """
    Returns list of (original_candidate_id, query).
    Deduplicates exact normalized duplicates within the row.
    """
    if not isinstance(generated_queries, list):
        return []

    out: List[Tuple[int, str]] = []
    seen_norm = set()

    for i, item in enumerate(generated_queries[:max_candidates]):
        if isinstance(item, str):
            q = item.strip()
        elif isinstance(item, dict):
            q = str(item.get("query", "")).strip()
        else:
            q = ""

        if not q:
            continue

        qn = norm(q)
        if dedup and qn in seen_norm:
            continue
        seen_norm.add(qn)
        out.append((i, q))

    return out


def pick_best_clean(evals: List[CandEval]) -> Optional[CandEval]:
    clean = [e for e in evals if e.clean]
    if not clean:
        return None
    return sorted(
        clean,
        key=lambda e: (e.connectivity_score, e.final_score, -len(e.keyword_hits)),
        reverse=True,
    )[0]


def pick_best_usable(evals: List[CandEval]) -> Optional[CandEval]:
    if not evals:
        return None
    return sorted(
        evals,
        key=lambda e: (
            e.connectivity_score,
            e.clean,
            e.final_score,
            -len(e.keyword_hits),
        ),
        reverse=True,
    )[0]


def choose_selected_candidate(
    best_clean: Optional[CandEval],
    best_usable: Optional[CandEval],
    min_select_threshold: int,
    clean_connectivity_threshold: int,
    clean_margin: int,
) -> tuple[Optional[CandEval], str]:
    if best_usable is None:
        return None, "no_candidates"

    if best_usable.connectivity_score < min_select_threshold:
        return None, "all_candidates_weak"

    if best_clean is not None:
        clean_strong_enough = best_clean.connectivity_score >= clean_connectivity_threshold
        clean_close_enough = best_clean.connectivity_score >= (best_usable.connectivity_score - clean_margin)
        if clean_strong_enough and clean_close_enough:
            return best_clean, "selected clean candidate with strong connectivity and near-best relevance"

    return best_usable, "selected most usable candidate because it preserved relevance better"


def collect_all_good_queries(
    evals: List[CandEval],
    good_save_threshold: int,
) -> List[Dict[str, Any]]:
    """
    [v2 NEW] Collect all clean candidates above good_save_threshold for training data.
    Returns them sorted by connectivity_score descending.

    Rationale: for retrieval model training, multiple (query, pos) pairs per document
    increase data volume and query diversity. Discarding high-quality candidates just
    because a better one exists wastes good training signal.
    """
    good = [
        e for e in evals
        if e.clean
        and not e.judge_failed
        and e.connectivity_score >= good_save_threshold
    ]
    good_sorted = sorted(good, key=lambda e: e.connectivity_score, reverse=True)
    return [
        {
            "candidate_id": e.candidate_id,
            "query": e.query,
            "connectivity_score": e.connectivity_score,
            "keyword_score": e.keyword_score,
            "final_score": e.final_score,
        }
        for e in good_sorted
    ]


# -----------------------------
# row preparation / finalization
# -----------------------------
def prepare_row_state(
    row: Dict[str, Any],
    uid: str,
    order_idx: int,
    args: argparse.Namespace,
) -> RowState:
    pos = str(row.get("pos", ""))
    keyword_phrases = parse_keywords_field(row.get("keywords"))
    extracted = extract_candidates_dedup(
        row.get("generated_queries"),
        max_candidates=args.max_candidates,
        dedup=(not args.no_dedup_candidates),
    )

    out = dict(row)
    out["_row_uid"] = uid
    out["keyword_phrases_used"] = keyword_phrases
    out["selection_policy"] = {
        "min_select_threshold": args.min_select_threshold,
        "clean_connectivity_threshold": args.clean_connectivity_threshold,
        "clean_margin": args.clean_margin,
        "good_connectivity_threshold": args.good_connectivity_threshold,
        "ok_connectivity_threshold": args.ok_connectivity_threshold,
        "good_save_threshold": args.good_save_threshold,  # [v2]
    }

    candidates: List[CandidateMeta] = []
    for candidate_id, q in extracted:
        hits = keyword_hits_exact_phrase(q, keyword_phrases)
        kw_score = keyword_score_from_hits(hits)
        prompt = build_connectivity_prompt(pos, q)
        candidates.append(
            CandidateMeta(
                candidate_id=candidate_id,
                query=q,
                keyword_hits=hits,
                keyword_score=kw_score,
                prompt=prompt,
            )
        )

    return RowState(order_idx=order_idx, out=out, candidates=candidates)


def finalize_row(
    state: RowState,
    parsed_outputs: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    out = state.out
    candidates = state.candidates

    if not candidates:
        out["candidate_evaluations"] = []
        out["top_clean_query"] = None
        out["top_usable_query"] = None
        out["selected_query"] = None
        out["selected_candidate_id"] = None
        out["selected_scores"] = None
        out["selection_note"] = "no_candidates"
        out["selection_diagnostics"] = None
        out["all_good_queries"] = []  # [v2]
        return out

    evals: List[CandEval] = []
    for meta, parsed in zip(candidates, parsed_outputs):
        # [v2] Detect judge parse failure before extracting score
        judge_failed = is_judge_parse_failure(parsed)

        conn = coerce_int_0_100(parsed.get("connectivity_score", 0), default=0)
        conf = coerce_float_0_1(parsed.get("confidence", 0.5), default=0.5)

        label = parsed.get("label", "")
        if not isinstance(label, str):
            label = ""

        reason = parsed.get("reason", "")
        if not isinstance(reason, str) or not reason.strip():
            reason = ""

        final = args.kw_weight * meta.keyword_score + args.conn_weight * conn

        evals.append(
            CandEval(
                candidate_id=meta.candidate_id,
                query=meta.query,
                keyword_hits=meta.keyword_hits,
                keyword_score=meta.keyword_score,
                connectivity_score=conn,
                confidence=conf,
                label=label,
                reason=reason,
                final_score=final,
                clean=(len(meta.keyword_hits) == 0),
                judge_failed=judge_failed,  # [v2]
            )
        )

    # [v2] Exclude judge-failed candidates from selection to avoid unfair penalization.
    # They are still recorded in candidate_evaluations for transparency.
    evals_for_selection = [e for e in evals if not e.judge_failed]

    evals_by_final = sorted(
        evals,
        key=lambda e: (e.final_score, e.connectivity_score, e.clean),
        reverse=True,
    )
    best_clean = pick_best_clean(evals_for_selection)
    best_usable = pick_best_usable(evals_for_selection)
    selected, selection_note = choose_selected_candidate(
        best_clean=best_clean,
        best_usable=best_usable,
        min_select_threshold=args.min_select_threshold,
        clean_connectivity_threshold=args.clean_connectivity_threshold,
        clean_margin=args.clean_margin,
    )

    out["candidate_evaluations"] = [
        {
            "candidate_id": e.candidate_id,
            "query": e.query,
            "keyword_hits": e.keyword_hits,
            "keyword_score": e.keyword_score,
            "connectivity_score": e.connectivity_score,
            "confidence": e.confidence,
            "label": e.label,
            "reason": e.reason,
            "final_score": e.final_score,
            "clean": e.clean,
            "judge_failed": e.judge_failed,  # [v2]
            "is_good_connectivity": e.connectivity_score >= args.good_connectivity_threshold,
            "is_ok_connectivity": e.connectivity_score >= args.ok_connectivity_threshold,
            "passes_min_select_threshold": e.connectivity_score >= args.min_select_threshold,
            "passes_clean_connectivity_threshold": e.connectivity_score >= args.clean_connectivity_threshold,
        }
        for e in evals_by_final
    ]

    out["top_clean_query"] = None if best_clean is None else {
        "candidate_id": best_clean.candidate_id,
        "query": best_clean.query,
        "keyword_score": best_clean.keyword_score,
        "connectivity_score": best_clean.connectivity_score,
        "final_score": best_clean.final_score,
        "keyword_hits": best_clean.keyword_hits,
        "clean": best_clean.clean,
    }

    out["top_usable_query"] = None if best_usable is None else {
        "candidate_id": best_usable.candidate_id,
        "query": best_usable.query,
        "keyword_score": best_usable.keyword_score,
        "connectivity_score": best_usable.connectivity_score,
        "final_score": best_usable.final_score,
        "keyword_hits": best_usable.keyword_hits,
        "clean": best_usable.clean,
    }

    out["selected_query"] = None if selected is None else selected.query
    out["selected_candidate_id"] = None if selected is None else selected.candidate_id
    out["selected_scores"] = None if selected is None else {
        "keyword_score": selected.keyword_score,
        "connectivity_score": selected.connectivity_score,
        "final_score": selected.final_score,
        "keyword_hits": selected.keyword_hits,
        "clean": selected.clean,
    }
    out["selection_note"] = selection_note

    if best_clean is not None and best_usable is not None:
        out["selection_diagnostics"] = {
            "best_clean_connectivity": best_clean.connectivity_score,
            "best_usable_connectivity": best_usable.connectivity_score,
            "connectivity_gap_clean_vs_usable": best_usable.connectivity_score - best_clean.connectivity_score,
        }
    else:
        out["selection_diagnostics"] = None

    # [v2] Save all good queries for training data
    out["all_good_queries"] = collect_all_good_queries(evals, args.good_save_threshold)

    return out


# -----------------------------
# batched judging
# -----------------------------
def run_batched_judging(
    llm: LLM,
    sp: SamplingParams,
    row_states: List[RowState],
    prompt_cache: Dict[str, Dict[str, Any]],
) -> List[Tuple[RowState, List[Dict[str, Any]]]]:
    """
    Judge all candidates from many rows in one vLLM call.
    Uses prompt_cache when exact prompt repeats.
    Preserves the order of `row_states`.
    """
    flat_prompts: List[str] = []
    flat_meta: List[Tuple[int, int]] = []  # (row_idx, cand_idx)

    per_row_parsed: List[List[Optional[Dict[str, Any]]]] = [
        [None] * len(rs.candidates) for rs in row_states
    ]

    for row_idx, rs in enumerate(row_states):
        for cand_idx, cand in enumerate(rs.candidates):
            cached = prompt_cache.get(cand.prompt)
            if cached is not None:
                per_row_parsed[row_idx][cand_idx] = cached
            else:
                flat_prompts.append(cand.prompt)
                flat_meta.append((row_idx, cand_idx))

    if flat_prompts:
        outputs = llm.generate(flat_prompts, sp)
        for out_obj, (row_idx, cand_idx), prompt in zip(outputs, flat_meta, flat_prompts):
            raw_text = out_obj.outputs[0].text if out_obj.outputs else ""
            parsed = safe_parse_json(raw_text)
            prompt_cache[prompt] = parsed
            per_row_parsed[row_idx][cand_idx] = parsed

    finalized: List[Tuple[RowState, List[Dict[str, Any]]]] = []
    for rs, parsed_list in zip(row_states, per_row_parsed):
        parsed_clean = [p if p is not None else {} for p in parsed_list]
        finalized.append((rs, parsed_clean))

    return finalized


def run_judge_retry(
    llm: LLM,
    sp_retry: SamplingParams,
    row_states: List[RowState],
    parsed_results: List[List[Dict[str, Any]]],
    prompt_cache: Dict[str, Dict[str, Any]],
) -> List[List[Dict[str, Any]]]:
    """
    [v2 NEW] Retry judge for candidates where parsing failed (conn=0, reason="").

    For each row, identify candidates whose parsed output indicates a parse failure,
    and re-run the judge. Uses a slightly higher max_tokens to improve parse success.
    Returns updated parsed_results with retried values filled in.
    """
    flat_prompts: List[str] = []
    flat_meta: List[Tuple[int, int]] = []  # (row_idx, cand_idx)

    for row_idx, rs in enumerate(row_states):
        for cand_idx, cand in enumerate(rs.candidates):
            parsed = parsed_results[row_idx][cand_idx]
            if is_judge_parse_failure(parsed):
                # Don't use cache for retry — previous result was a failure
                flat_prompts.append(cand.prompt)
                flat_meta.append((row_idx, cand_idx))

    if not flat_prompts:
        return parsed_results

    outputs = llm.generate(flat_prompts, sp_retry)
    for out_obj, (row_idx, cand_idx), prompt in zip(outputs, flat_meta, flat_prompts):
        raw_text = out_obj.outputs[0].text if out_obj.outputs else ""
        parsed = safe_parse_json(raw_text)
        # Only accept retry result if it actually parsed successfully
        if not is_judge_parse_failure(parsed):
            parsed_results[row_idx][cand_idx] = parsed
            prompt_cache[prompt] = parsed  # Update cache with successful result

    return parsed_results


def drain_ready_rows_in_order(
    finalized_by_order: Dict[int, Dict[str, Any]],
    next_write_order: int,
    write_buffer: List[Dict[str, Any]],
    writer: JsonlAppender,
    save_every: int,
    written_rows: int,
) -> Tuple[int, int]:
    """
    Move only contiguous ready rows [next_write_order, next_write_order+1, ...]
    into write_buffer, and flush buffer as needed.
    """
    while next_write_order in finalized_by_order:
        write_buffer.append(finalized_by_order.pop(next_write_order))
        next_write_order += 1

        if len(write_buffer) >= save_every:
            writer.write_rows(write_buffer)
            written_rows += len(write_buffer)
            write_buffer.clear()

    return next_write_order, written_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_path", required=True)
    ap.add_argument("--output_path", required=True)

    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.9)

    ap.add_argument("--max_candidates", type=int, default=5)

    # number of candidate prompts judged together across rows
    ap.add_argument("--judge_batch_size", type=int, default=256)

    # output write interval in rows
    ap.add_argument("--save_every", type=int, default=200)

    # blended score is kept for reporting, but usable selection is connectivity-first
    ap.add_argument("--kw_weight", type=float, default=0.2)
    ap.add_argument("--conn_weight", type=float, default=0.8)

    # reporting thresholds
    ap.add_argument("--good_connectivity_threshold", type=int, default=70)
    ap.add_argument("--ok_connectivity_threshold", type=int, default=60)

    # selection thresholds
    ap.add_argument("--min_select_threshold", type=int, default=45)
    ap.add_argument("--clean_connectivity_threshold", type=int, default=65)
    ap.add_argument("--clean_margin", type=int, default=10)

    # [v2] threshold for all_good_queries collection (for training data)
    ap.add_argument("--good_save_threshold", type=int, default=65,
                    help="connectivity_score threshold for saving to all_good_queries")

    # Faster / more stable defaults
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--max_tokens", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)

    # [v2] retry settings for judge parse failures
    ap.add_argument("--retry_max_tokens", type=int, default=128,
                    help="max_tokens for judge retry (slightly higher than initial)")
    ap.add_argument("--no_judge_retry", action="store_true",
                    help="disable judge retry for parse failures")

    # optional toggles
    ap.add_argument("--fsync_each_save", action="store_true")
    ap.add_argument("--no_dedup_candidates", action="store_true")

    args = ap.parse_args()

    processed = load_processed_uids(args.output_path)

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        enable_prefix_caching=True,
    )
    sp = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
        stop=["\n"],
    )
    # [v2] Retry sampling params — slightly more tokens, same greedy decoding
    sp_retry = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.retry_max_tokens,
        seed=args.seed,
        stop=["\n"],
    )

    writer = JsonlAppender(args.output_path, do_fsync=args.fsync_each_save)

    prompt_cache: Dict[str, Dict[str, Any]] = {}
    write_buffer: List[Dict[str, Any]] = []
    pending_row_states: List[RowState] = []
    pending_prompt_count = 0

    # order-preserving state
    finalized_by_order: Dict[int, Dict[str, Any]] = {}
    next_write_order = 0
    order_counter = 0

    seen_rows = 0
    written_rows = 0

    try:
        for row in iter_jsonl(args.input_path):
            seen_rows += 1

            uid = row.get("_row_uid")
            if not isinstance(uid, str) or not uid:
                uid = f"row_{seen_rows}"

            if uid in processed:
                continue

            state = prepare_row_state(
                row=row,
                uid=uid,
                order_idx=order_counter,
                args=args,
            )
            order_counter += 1
            processed.add(uid)

            # Rows with no candidates finalize immediately,
            # but are NOT written immediately. They wait by order index.
            if not state.candidates:
                finalized = finalize_row(state, parsed_outputs=[], args=args)
                finalized_by_order[state.order_idx] = finalized
            else:
                pending_row_states.append(state)
                pending_prompt_count += len(state.candidates)

            # If enough candidate prompts accumulated, judge them in one batch.
            if pending_prompt_count >= args.judge_batch_size:
                judged = run_batched_judging(llm, sp, pending_row_states, prompt_cache)

                for rs, parsed_list in judged:
                    # [v2] Retry parse failures before finalizing
                    if not args.no_judge_retry:
                        updated = run_judge_retry(
                            llm, sp_retry,
                            [rs], [parsed_list],
                            prompt_cache,
                        )
                        parsed_list = updated[0]

                    finalized = finalize_row(rs, parsed_list, args)
                    finalized_by_order[rs.order_idx] = finalized

                pending_row_states = []
                pending_prompt_count = 0

            # Drain only rows that are ready in exact order.
            next_write_order, written_rows = drain_ready_rows_in_order(
                finalized_by_order=finalized_by_order,
                next_write_order=next_write_order,
                write_buffer=write_buffer,
                writer=writer,
                save_every=args.save_every,
                written_rows=written_rows,
            )

        # Flush remaining LLM batch.
        if pending_row_states:
            judged = run_batched_judging(llm, sp, pending_row_states, prompt_cache)

            for rs, parsed_list in judged:
                # [v2] Retry parse failures before finalizing
                if not args.no_judge_retry:
                    updated = run_judge_retry(
                        llm, sp_retry,
                        [rs], [parsed_list],
                        prompt_cache,
                    )
                    parsed_list = updated[0]

                finalized = finalize_row(rs, parsed_list, args)
                finalized_by_order[rs.order_idx] = finalized

            pending_row_states = []
            pending_prompt_count = 0

        # Final drain in exact order.
        next_write_order, written_rows = drain_ready_rows_in_order(
            finalized_by_order=finalized_by_order,
            next_write_order=next_write_order,
            write_buffer=write_buffer,
            writer=writer,
            save_every=args.save_every,
            written_rows=written_rows,
        )

        # Safety check: after final flush, there should be nothing unwritten.
        if finalized_by_order:
            missing = sorted(finalized_by_order.keys())[:10]
            raise RuntimeError(
                f"Unwritten finalized rows remain due to non-contiguous order issue. "
                f"next_write_order={next_write_order}, sample_remaining={missing}"
            )

        # Flush final writes.
        if write_buffer:
            writer.write_rows(write_buffer)
            written_rows += len(write_buffer)
            write_buffer.clear()

    finally:
        writer.close()

    print(f"[DONE] seen={seen_rows} written={written_rows} output={args.output_path}")


if __name__ == "__main__":
    main()
