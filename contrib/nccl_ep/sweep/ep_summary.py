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

    # Single BW per phase (send_bw == recv_bw under per-rank avg by symmetry,
    # so showing both is redundant). Latency metrics carry avg/min/max from
    # the CSV; per-iter p99 is not yet collected by ep_bench (cudaEvent loop
    # only keeps min/max/avg, not the full per-iter array). Adding p99 means
    # changing runPairedBenchmark to keep the times array exposed to print.
    BW_METRICS = [
        ("dispatch_bw_gbs", "Dispatch BW (GB/s, kernel time, per-iter)"),
        ("combine_bw_gbs",  "Combine  BW (GB/s, kernel time, per-iter)"),
    ]
    LATENCY_METRICS = [
        ("dispatch", "Dispatch latency (us)  avg / min / max"),
        ("combine",  "Combine  latency (us)  avg / min / max"),
    ]
    # Columns we read out of the CSV per (mode, ep, tk)
    CSV_COLS = [
        "dispatch_bw_gbs", "combine_bw_gbs",
        "dispatch_avg_us", "dispatch_min_us", "dispatch_max_us",
        "combine_avg_us",  "combine_min_us",  "combine_max_us",
        "dispatch_kernel_us", "combine_kernel_us",
    ]

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
            rows[(mode, ep, tk)] = {key: _to_float(row.get(key, "")) for key in CSV_COLS}

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

    def fmt(x: float | None, w: int = 7, prec: int = 1) -> str:
        if x is None:
            return f"{'-':>{w}s}"
        return f"{x:>{w}.{prec}f}"

    def render_bw(metric_key: str, title: str) -> None:
        print(f"\n=== {title} ===")
        # Header row 1: EP groups
        head1 = f"{'tokens':>7s} "
        for ep in ep_sizes:
            head1 += f" {'EP=' + str(ep):>26s}"
        print(head1)
        head2 = " " * 8
        for _ in ep_sizes:
            head2 += f"   {'HT':>7s}  {'FM':>7s}  {'Delta':>7s}"
        print(head2)
        for tk in tokens:
            line = f"{tk:>7d} "
            for ep in ep_sizes:
                ht = rows.get((ht_mode, ep, tk), {}).get(metric_key)
                fm = rows.get((fm_mode, ep, tk), {}).get(metric_key)
                ht_s = fmt(ht)
                fm_s = fmt(fm)
                if ht is not None and fm is not None and ht > 0:
                    delta = (fm - ht) / ht * 100.0
                    delta_s = f"{delta:+6.0f}%"
                else:
                    delta_s = f"{'-':>7s}"
                line += f"   {ht_s}  {fm_s}  {delta_s}"
            print(line)

    def render_latency(phase: str, title: str) -> None:
        # phase = "dispatch" or "combine" -> reads <phase>_{avg,min,max}_us and
        # <phase>_kernel_us. Each (EP, tokens) cell shows HT and FM rows with
        # avg/min/max + kernel.
        avg_key = f"{phase}_avg_us"
        min_key = f"{phase}_min_us"
        max_key = f"{phase}_max_us"
        ker_key = f"{phase}_kernel_us"

        print(f"\n=== {title} (kernel = per-iter total kernel us) ===")
        # 1 row per (tokens, mode); 1 column group per EP (avg/min/max/kernel)
        head1 = f"{'tokens':>7s} {'mode':>4s} "
        for ep in ep_sizes:
            head1 += f" {'EP=' + str(ep):>33s}"
        print(head1)
        head2 = " " * 13
        for _ in ep_sizes:
            head2 += f"   {'avg':>7s}  {'min':>7s}  {'max':>7s}  {'ker':>6s}"
        print(head2)
        for tk in tokens:
            for label, mode in (("HT", ht_mode), ("FM", fm_mode)):
                line = f"{tk:>7d} {label:>4s} "
                for ep in ep_sizes:
                    r = rows.get((mode, ep, tk), {})
                    line += (f"   {fmt(r.get(avg_key))}"
                             f"  {fmt(r.get(min_key))}"
                             f"  {fmt(r.get(max_key))}"
                             f"  {fmt(r.get(ker_key), w=6, prec=0)}")
                print(line)

    print("=" * 80)
    print(f"HT vs FULLMESH summary  (csv: {csv_path})")
    print(f"  HT mode       : {ht_mode}")
    print(f"  FM mode       : {fm_mode}")
    print(f"  EP sizes seen : {ep_sizes}")
    print(f"  tokens seen   : {tokens}")
    print(f"  BW base       : kernel time, per-iter total")
    print(f"  BW direction  : single value per phase. send_bytes / recv_bytes")
    print(f"                  per-rank avg are equal under symmetric routing")
    print(f"                  (sum_send == sum_recv globally), so send_bw and")
    print(f"                  recv_bw share the same number; collapsed to one.")
    print(f"  Latency stats : avg / min / max from cudaEvent dispatch/combine")
    print(f"                  iter loop. p99 not yet captured -- ep_bench keeps")
    print(f"                  only avg/min/max, not the full per-iter array.")
    print("=" * 80)

    for key, label in BW_METRICS:
        render_bw(key, label)
    for phase, label in LATENCY_METRICS:
        render_latency(phase, label)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
