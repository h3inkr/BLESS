#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mmlu_repro.utils import get_ctxs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="rulins/mmlu_searched_results_from_massiveds")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    ds = load_dataset(args.dataset, split=args.split)
    print(ds)
    print("columns:", ds.column_names)

    n = min(args.limit, len(ds))
    lengths = []
    for i in range(n):
        ex = ds[i]
        ctxs = get_ctxs(ex)
        lengths.append(len(ctxs))
        print("=" * 80)
        print("row:", i)
        print("query[:300]:", ex.get("query", "")[:300].replace("\n", " "))
        print("raw_query[:300]:", ex.get("raw_query", "")[:300].replace("\n", " "))
        print("num_ctxs:", len(ctxs))
        if ctxs:
            print("ctx keys:", sorted(ctxs[0].keys()))
            print("ctx[0] text[:300]:", ctxs[0].get("retrieval text", "")[:300].replace("\n", " "))

    if lengths:
        print("=" * 80)
        print("ctx length stats over inspected rows:")
        print("min:", min(lengths), "max:", max(lengths), "avg:", sum(lengths) / len(lengths))


if __name__ == "__main__":
    main()
