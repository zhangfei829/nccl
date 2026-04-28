#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Print HT vs FULLMESH bandwidth comparison from an EP-sweep CSV.

Reads ``all_results.csv`` produced by ``run_all_from_jumphost.sh`` and emits
two compact tables (Dispatch BW, Combine BW). Rows are ``tokens`` values,
column groups are ``ep_size`` values, each group has ``HT`` and ``FM``
sub-columns plus a ``Delta%`` showing FM relative to HT.

Usage:
    ep_summary.py <all_results.csv>
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict


def _to_float(s: str) -> float | None:
    if s is None or s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(f"Usage: {argv[0]} <all_results.csv>", file=sys.stderr)
        return 1
    csv_path = argv[1]

    # Map (mode, ep_size, tokens) -> {dispatch_bw_gbs, combine_bw_gbs}
    rows: dict[tuple[str, int, int], dict[str, float | None]] = {}
    ep_sizes_seen: set[int] = set()
    tokens_seen: set[int] = set()
    modes_seen: set[str] = set()

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mode = row.get("mode", "").strip()
            try:
                ep = int(row["ep_size"])
                tk = int(row["tokens"])
            except (KeyError, ValueError):
                continue
            modes_seen.add(mode)
            ep_sizes_seen.add(ep)
            tokens_seen.add(tk)
            rows[(mode, ep, tk)] = {
                "dispatch_bw_gbs": _to_float(row.get("dispatch_bw_gbs", "")),
                "combine_bw_gbs": _to_float(row.get("combine_bw_gbs", "")),
            }

    ht_modes = [m for m in sorted(modes_seen) if m.startswith("ht_")]
    fm_modes = [m for m in sorted(modes_seen) if m.startswith("fullmesh_")]
    if not ht_modes or not fm_modes:
        print(
            f"[ep_summary] no HT or FM rows found "
            f"(modes={sorted(modes_seen)}); skipping comparison.",
            file=sys.stderr,
        )
        return 0

    # Use the first ht_* and fullmesh_* mode for the comparison; if multiple
    # dtypes were swept, only the first one (typically bf16) is shown here.
    ht_mode = ht_modes[0]
    fm_mode = fm_modes[0]
    ep_sizes = sorted(ep_sizes_seen)
    tokens = sorted(tokens_seen)

    def render(metric_key: str, title: str) -> None:
        print(f"\n=== {title} ===")
        # Header row 1: EP groups
        head1 = f"{'tokens':>7s} "
        for ep in ep_sizes:
            head1 += f" {'EP=' + str(ep):>26s}"
        print(head1)
        # Header row 2: HT / FM / Delta% sub-columns
        head2 = " " * 8
        for _ in ep_sizes:
            head2 += f"   {'HT':>7s}  {'FM':>7s}  {'Delta':>7s}"
        print(head2)
        # Data rows
        for tk in tokens:
            line = f"{tk:>7d} "
            for ep in ep_sizes:
                ht = rows.get((ht_mode, ep, tk), {}).get(metric_key)
                fm = rows.get((fm_mode, ep, tk), {}).get(metric_key)
                ht_s = f"{ht:>7.1f}" if ht is not None else f"{'-':>7s}"
                fm_s = f"{fm:>7.1f}" if fm is not None else f"{'-':>7s}"
                if ht is not None and fm is not None and ht > 0:
                    delta = (fm - ht) / ht * 100.0
                    delta_s = f"{delta:+6.0f}%"
                else:
                    delta_s = f"{'-':>7s}"
                line += f"   {ht_s}  {fm_s}  {delta_s}"
            print(line)

    print("=" * 80)
    print(f"HT vs FULLMESH bandwidth summary  (csv: {csv_path})")
    print(f"  HT mode       : {ht_mode}")
    print(f"  FM mode       : {fm_mode}")
    print(f"  EP sizes seen : {ep_sizes}")
    print(f"  tokens seen   : {tokens}")
    print("=" * 80)
    render("dispatch_bw_gbs", "Dispatch BW (GB/s, kernel time, per-iter)")
    render("combine_bw_gbs",  "Combine  BW (GB/s, kernel time, per-iter)")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
