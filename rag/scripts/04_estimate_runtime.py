#!/usr/bin/env python
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds_for_limit", type=float, required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--total_queries", type=int, default=33481)
    args = parser.parse_args()

    per_query = args.seconds_for_limit / args.limit
    total_sec = per_query * args.total_queries
    print(f"Per-query time: {per_query:.3f} sec")
    print(f"Estimated full time: {total_sec/3600:.2f} hours = {total_sec/86400:.2f} days")


if __name__ == "__main__":
    main()
