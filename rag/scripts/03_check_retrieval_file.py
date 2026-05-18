#!/usr/bin/env python
import argparse
import json
from collections import Counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval_file", required=True)
    parser.add_argument("--min_ctxs", type=int, default=1)
    parser.add_argument("--show", type=int, default=2)
    args = parser.parse_args()

    n = 0
    bad = 0
    ctx_lens = []
    missing = Counter()

    with open(args.retrieval_file, "r", encoding="utf-8") as f:
        for line in f:
            n += 1
            ex = json.loads(line)
            for key in ["raw_query", "query", "ctxs"]:
                if key not in ex:
                    missing[key] += 1
            ctxs = ex.get("ctxs", [])
            ctx_lens.append(len(ctxs))
            if len(ctxs) < args.min_ctxs:
                bad += 1
            for c in ctxs[: min(3, len(ctxs))]:
                if "retrieval text" not in c:
                    missing["ctx.retrieval text"] += 1

            if n <= args.show:
                print("=" * 80)
                print("row", n)
                print("query[:300]:", ex.get("query", "")[:300].replace("\n", " "))
                print("num_ctxs:", len(ctxs))
                if ctxs:
                    print("ctx0 keys:", sorted(ctxs[0].keys()))
                    print("ctx0 score/rank:", ctxs[0].get("score"), ctxs[0].get("rank"))
                    print("ctx0 text[:300]:", ctxs[0].get("retrieval text", "")[:300].replace("\n", " "))

    print("=" * 80)
    print("rows:", n)
    print("bad rows with ctxs < min_ctxs:", bad)
    print("missing:", dict(missing))
    if ctx_lens:
        print("ctx lens min/max/avg:", min(ctx_lens), max(ctx_lens), sum(ctx_lens) / len(ctx_lens))


if __name__ == "__main__":
    main()
