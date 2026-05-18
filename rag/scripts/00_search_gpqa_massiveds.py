#!/usr/bin/env python
"""GPQA candidate pool 생성: massive-serve API를 통해 MassiveDS-140B에서 검색.

사전 준비:
    pip install massive-serve
    massive-serve serve --domain_name massiveds_140b   # 별도 터미널에서 실행

실행 예시 (smoke test):
    python scripts/00_search_gpqa_massiveds.py \
        --config gpqa_diamond \
        --output retrieval_files/gpqa/gpqa_diamond_pool.jsonl \
        --n_docs 100 \
        --limit 5

전체 실행:
    python scripts/00_search_gpqa_massiveds.py \
        --config gpqa_diamond \
        --output retrieval_files/gpqa/gpqa_diamond_pool.jsonl \
        --n_docs 100

출력 포맷:
    기존 mmlu_searched_results_from_massiveds와 동일한 구조:
    {"query": "...", "raw_query": "...", "ctxs": [{"retrieval text": ..., "id": ..., "source": ...}, ...]}
    → 이후 01/02 스크립트에 --dataset 대신 로컬 파일로 바로 투입 가능.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import requests
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mmlu_repro.utils import ensure_parent, write_jsonl_line


# ---------------------------------------------------------------------------
# GPQA 쿼리 포맷터
# ---------------------------------------------------------------------------

def format_gpqa_query(ex: dict, include_choices: bool = True) -> str:
    """GPQA 예제를 MMLU 스타일 multiple-choice 쿼리 문자열로 변환.

    include_choices=True: 선택지 포함 (검색 시 더 많은 context 제공).
    include_choices=False: 질문만 (짧고 clean한 쿼리).
    """
    question = ex["Question"]
    if not include_choices:
        return question

    choices = [
        ex.get("Correct Answer", ""),
        ex.get("Incorrect Answer 1", ""),
        ex.get("Incorrect Answer 2", ""),
        ex.get("Incorrect Answer 3", ""),
    ]
    # 알파벳 순서로 정렬 (셔플 없음 — 검색용이므로 순서 무관)
    labeled = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices) if c)
    return f"{question}\n{labeled}"


# ---------------------------------------------------------------------------
# massive-serve 클라이언트
# ---------------------------------------------------------------------------

def search_massiveds(
    query: str,
    n_docs: int,
    server_url: str,
    domains: str,
    retries: int = 3,
    retry_delay: float = 2.0,
) -> list:
    """massive-serve /search 엔드포인트 호출. ctxs 리스트 반환."""
    payload = {"query": query, "n_docs": n_docs, "domains": domains}
    headers = {"Content-Type": "application/json"}

    for attempt in range(retries):
        try:
            resp = requests.post(
                f"{server_url}/search",
                json=payload,
                headers=headers,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()

            # massive-serve 응답: {"results": [{"text": ..., "id": ..., "source": ...}, ...]}
            # → mmlu pool 포맷: {"retrieval text": ..., "id": ..., "source": ...}
            ctxs = []
            for item in data.get("results", []):
                ctx = {
                    "retrieval text": item.get("text", item.get("retrieval text", "")),
                    "id": str(item.get("id", "")),
                    "source": item.get("source", item.get("domain", "")),
                }
                if ctx["retrieval text"]:
                    ctxs.append(ctx)
            return ctxs

        except requests.exceptions.RequestException as e:
            if attempt < retries - 1:
                print(f"\n[warn] 검색 실패 (시도 {attempt+1}/{retries}): {e} — {retry_delay}초 후 재시도")
                time.sleep(retry_delay)
            else:
                print(f"\n[error] 최대 재시도 초과: {e}")
                return []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="GPQA → MassiveDS-140B 검색 → candidate pool JSONL 생성"
    )
    parser.add_argument(
        "--config",
        default="gpqa_diamond",
        choices=["gpqa_diamond", "gpqa_main", "gpqa_extended", "gpqa_experts"],
        help="GPQA config 선택 (기본: gpqa_diamond, 198문제)",
    )
    parser.add_argument(
        "--output",
        default="retrieval_files/gpqa/gpqa_diamond_pool.jsonl",
        help="출력 JSONL 경로",
    )
    parser.add_argument(
        "--server_url",
        default="http://localhost:8000",
        help="massive-serve 서버 URL (기본: http://localhost:8000)",
    )
    parser.add_argument(
        "--domains",
        default="massiveds_140b",
        help="massive-serve domains 파라미터 (서버 설정에 맞게 조정)",
    )
    parser.add_argument(
        "--n_docs",
        type=int,
        default=100,
        help="쿼리당 검색할 candidate 수 (기본: 100)",
    )
    parser.add_argument(
        "--include_choices",
        action="store_true",
        default=True,
        help="쿼리에 선택지 포함 여부 (기본: True)",
    )
    parser.add_argument(
        "--no_include_choices",
        dest="include_choices",
        action="store_false",
        help="쿼리를 질문만으로 구성",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="처리할 최대 예제 수 (-1 = 전체)",
    )
    parser.add_argument(
        "--request_delay",
        type=float,
        default=0.0,
        help="쿼리 간 대기 시간(초). 서버 부하 조절용 (기본: 0.0)",
    )
    args = parser.parse_args()

    ensure_parent(args.output)

    # GPQA 로드 (gated dataset — HF 로그인 필요)
    print(f"Loading Idavidrein/gpqa ({args.config})...")
    try:
        ds = load_dataset("Idavidrein/gpqa", args.config, split="train")
    except Exception as e:
        print(f"[error] GPQA 로드 실패: {e}")
        print("HuggingFace에서 데이터셋 접근 동의가 필요합니다:")
        print("  huggingface-cli login")
        print("  https://huggingface.co/datasets/Idavidrein/gpqa 에서 약관 동의")
        sys.exit(1)

    n = len(ds) if args.limit < 0 else min(args.limit, len(ds))
    print(f"  총 {len(ds)}개 중 {n}개 처리 예정")
    print(f"  서버: {args.server_url} | domains: {args.domains} | n_docs: {args.n_docs}")

    # 서버 health check
    try:
        resp = requests.get(args.server_url, timeout=5)
        print(f"  서버 응답: {resp.status_code}")
    except Exception:
        print(f"[warn] 서버({args.server_url})에 연결할 수 없음. massive-serve가 실행 중인지 확인하세요.")
        print("  massive-serve serve --domain_name massiveds_140b")

    skipped = 0
    with open(args.output, "w", encoding="utf-8") as fout:
        for i in tqdm(range(n), desc=f"Searching MassiveDS ({args.config})"):
            ex = ds[i]
            raw_query = format_gpqa_query(ex, include_choices=args.include_choices)
            query = raw_query  # original-query setting (MMLU 패키지와 동일)

            ctxs = search_massiveds(
                query=query,
                n_docs=args.n_docs,
                server_url=args.server_url,
                domains=args.domains,
            )

            if not ctxs:
                skipped += 1

            out = {
                "query": query,
                "raw_query": raw_query,
                # 원본 GPQA 필드도 보존 (필요 시 활용)
                "Question": ex["Question"],
                "Correct Answer": ex.get("Correct Answer", ""),
                "ctxs": ctxs,
            }
            write_jsonl_line(fout, out)

            if args.request_delay > 0:
                time.sleep(args.request_delay)

    print(f"\nDone.")
    print(f"  저장 경로:  {args.output}")
    print(f"  처리 완료:  {n}개")
    print(f"  ctxs 없음:  {skipped}개")
    print()
    print("다음 단계 — BM25 reranking:")
    print(f"  python scripts/01_make_gpqa_bm25.py --pool {args.output} --output retrieval_files/gpqa/bm25_original.jsonl")
    print()
    print("다음 단계 — Dense reranking:")
    print(f"  python scripts/02_make_gpqa_dense.py --pool {args.output} --backend hf_causal_eos \\")
    print(f"      --model_name_or_path Qwen/Qwen2.5-7B-Instruct --peft_adapter_path /path/to/lora \\")
    print(f"      --output retrieval_files/gpqa/ours_original.jsonl --append_eos --normalize")


if __name__ == "__main__":
    main()
