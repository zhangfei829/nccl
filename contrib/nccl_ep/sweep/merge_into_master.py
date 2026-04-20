#!/usr/bin/env python3
"""Merge one or more ep-sweep results.csv files into a long-lived master CSV.

Idempotent: rerunning the same sweep replaces older rows for the same
(ep_size, mode, tokens, dispatch_dtype_tag, algorithm) with the newer one
(based on the ``timestamp`` column).

Usage
-----
    merge_into_master.py  MASTER.csv  SRC1.csv [SRC2.csv ...]

The master file is created if it does not exist. All source columns
present in any input are preserved; missing cells stay empty.

Typical use:

    # Add one EP-size sweep to the master table
    python3 merge_into_master.py ~/fizhang/nccl_ep_master.csv \
            ~/fizhang/nccl-sweep-*/ep8/results.csv

    # Merge several historical sweeps into the master at once
    python3 merge_into_master.py ~/fizhang/nccl_ep_master.csv \
            ~/fizhang/nccl-sweep-20260420_141647/all_results.csv \
            ~/fizhang/nccl-sweep-20260420_143135/all_results.csv
"""
from __future__ import annotations

import csv
import os
import sys

DEDUP_KEY = ("ep_size", "mode", "tokens", "dispatch_dtype_tag", "algorithm")


def _load(path: str):
    if not path or not os.path.exists(path) or os.path.getsize(path) == 0:
        return [], []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    return fieldnames, rows


def _merge_fieldnames(all_fieldnames):
    """Preserve order of the first file's columns, then append any new ones."""
    merged: list[str] = []
    for fn in all_fieldnames:
        for col in fn:
            if col not in merged:
                merged.append(col)
    return merged


def _int_or_zero(s):
    try:
        return int(s)
    except (ValueError, TypeError):
        return 0


def merge(master_path: str, sources: list[str]) -> int:
    all_fieldnames = []
    all_rows = []

    # master is first so its column order wins
    master_fn, master_rows = _load(master_path)
    if master_fn:
        all_fieldnames.append(master_fn)
        all_rows.extend(master_rows)

    for p in sources:
        fn, rows = _load(p)
        if not fn:
            print(f"[merge] skip empty/missing: {p}", file=sys.stderr)
            continue
        all_fieldnames.append(fn)
        all_rows.extend(rows)

    if not all_rows:
        print("[merge] no rows to merge", file=sys.stderr)
        return 1

    fieldnames = _merge_fieldnames(all_fieldnames)
    # ensure dedup / sort columns exist
    for col in (*DEDUP_KEY, "timestamp"):
        if col not in fieldnames:
            fieldnames.append(col)

    latest: dict[tuple, dict] = {}
    for r in all_rows:
        key = tuple(str(r.get(k, "")) for k in DEDUP_KEY)
        ts = r.get("timestamp", "") or ""
        cur = latest.get(key)
        if cur is None or ts > (cur.get("timestamp", "") or ""):
            latest[key] = r

    ordered = sorted(
        latest.values(),
        key=lambda r: (
            _int_or_zero(r.get("ep_size")),
            (r.get("mode") or ""),
            _int_or_zero(r.get("tokens")),
            (r.get("dispatch_dtype_tag") or ""),
        ),
    )

    tmp = master_path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in ordered:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    os.replace(tmp, master_path)

    print(
        f"[merge] {len(all_rows)} input rows from "
        f"{1 + len(sources)} files -> {len(ordered)} unique rows "
        f"written to {master_path}"
    )
    return 0


def main() -> int:
    if len(sys.argv) < 3:
        print(
            "Usage: merge_into_master.py MASTER.csv SRC1.csv [SRC2.csv ...]",
            file=sys.stderr,
        )
        return 2
    return merge(sys.argv[1], sys.argv[2:])


if __name__ == "__main__":
    sys.exit(main())
