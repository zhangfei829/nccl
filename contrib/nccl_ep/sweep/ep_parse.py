#!/usr/bin/env python3
"""Parse a single ep_bench log file and append one row to a CSV.

Usage:
    ep_parse.py <log_file> <csv_file> [extra_k=v ...]

Extra k=v pairs are appended as columns (useful to tag run metadata like
``mode=ht_fp8`` or ``wall_s=12.3`` that the log itself does not record).

The parser tolerates:
  * LL vs HT output formats (both have the same Summary key words)
  * Missing kernel-time sections (CUPTI not captured)
  * Missing per-rank lines

Output columns (stable ordering, extra_k=v columns appended at the end):

    timestamp, algorithm, ranks, tokens, hidden, top_k, experts,
    dispatch_dtype, max_tokens_per_rank,
    dispatch_avg_us, dispatch_min_us, dispatch_max_us,
    dispatch_kernel_us,
    dispatch_bw_gbs, dispatch_recv_bw_gbs, dispatch_send_bw_gbs,
    combine_avg_us, combine_min_us, combine_max_us,
    combine_kernel_us,
    combine_bw_gbs, combine_send_bw_gbs, combine_recv_bw_gbs,
    total_dc_avg_us,
    setup_group_ms, setup_handle_ms,
    total_send_mb, total_recv_mb, nvl_send_mb, rdma_send_mb,
    log_path, <extras...>
"""
from __future__ import annotations

import csv
import os
import re
import sys
from datetime import datetime


def _grep(text: str, pattern: str, group: int = 1, cast=str, default=None):
    m = re.search(pattern, text)
    if not m:
        return default
    try:
        return cast(m.group(group))
    except (IndexError, ValueError):
        return default


def parse_log(log_path: str) -> dict:
    with open(log_path, "r", errors="replace") as f:
        text = f.read()

    r: dict = {"log_path": os.path.abspath(log_path)}

    # ---------- Config block ----------
    r["algorithm"] = _grep(text, r"Algorithm:\s+(\S+)")
    r["ranks"] = _grep(text, r"Ranks:\s+(\d+)", cast=int)
    r["tokens"] = _grep(text, r"Tokens:\s+(\d+)", cast=int)
    r["hidden"] = _grep(text, r"Hidden:\s+(\d+)", cast=int)
    r["top_k"] = _grep(text, r"Top-k:\s+(\d+)", cast=int)
    r["experts"] = _grep(text, r"Experts:\s+(\d+)", cast=int)
    r["dispatch_dtype"] = _grep(text, r"Dispatch dtype:\s+(\S+)")
    r["max_tokens_per_rank"] = r["tokens"]  # ep_bench: max==tokens unless dynamic

    # ---------- Summary split ----------
    ll_m = re.search(
        r"=== Summary \(Low Latency[^\n]*===\n(.+?)(?=\n===|\Z)",
        text, re.S)
    ht_m = re.search(
        r"=== Summary \(High Throughput[^\n]*===\n(.+?)(?=\n===|\Z)",
        text, re.S)

    if ll_m:
        s = ll_m.group(1)

        # Dispatch avg/min/max time + throughput
        r["dispatch_avg_us"] = _grep(s, r"Dispatch \([^)]+\):\s+avg=([\d.]+)", cast=float)
        r["dispatch_min_us"] = _grep(s, r"Dispatch \([^)]+\):\s+avg=[\d.]+ us, min=([\d.]+)", cast=float)
        r["dispatch_max_us"] = _grep(s, r"Dispatch \([^)]+\):\s+avg=[\d.]+ us, min=[\d.]+ us, max=([\d.]+)", cast=float)
        # "Dispatch (BF16):   ...\n                  throughput: avg=XX GB/s"
        m = re.search(r"Dispatch \([^)]+\):.*?throughput: avg=([\d.]+)\s+GB/s", s, re.S)
        r["dispatch_bw_gbs"] = float(m.group(1)) if m else None

        r["combine_avg_us"] = _grep(s, r"Combine \([^)]+\):\s+avg=([\d.]+)", cast=float)
        r["combine_min_us"] = _grep(s, r"Combine \([^)]+\):\s+avg=[\d.]+ us, min=([\d.]+)", cast=float)
        r["combine_max_us"] = _grep(s, r"Combine \([^)]+\):\s+avg=[\d.]+ us, min=[\d.]+ us, max=([\d.]+)", cast=float)
        m = re.search(r"Combine \([^)]+\):.*?throughput: avg=([\d.]+)\s+GB/s", s, re.S)
        r["combine_bw_gbs"] = float(m.group(1)) if m else None

        m = re.search(r"Total \(D\+C\):\s+avg=([\d.]+)\s+us", s)
        r["total_dc_avg_us"] = float(m.group(1)) if m else None

    elif ht_m:
        s = ht_m.group(1)

        # total-time block
        m = re.search(
            r"Dispatch:\s+total=([\d.]+)\s+us\s*\(min=([\d.]+),\s*max=([\d.]+)\)",
            s)
        if m:
            r["dispatch_avg_us"] = float(m.group(1))
            r["dispatch_min_us"] = float(m.group(2))
            r["dispatch_max_us"] = float(m.group(3))

        m = re.search(
            r"Combine:\s+total=([\d.]+)\s+us\s*\(min=([\d.]+),\s*max=([\d.]+)\)",
            s)
        if m:
            r["combine_avg_us"] = float(m.group(1))
            r["combine_min_us"] = float(m.group(2))
            r["combine_max_us"] = float(m.group(3))

        m = re.search(r"Total \(D\+C\):\s+avg=([\d.]+)\s+us", s)
        r["total_dc_avg_us"] = float(m.group(1)) if m else None

        # kernel-time block (preferred BW numbers)
        k = re.search(r"--- BW based on kernel time ---(.+?)$", s, re.S)
        if k:
            ks = k.group(1)
            r["dispatch_kernel_us"] = _grep(ks, r"Dispatch:\s+kernel=([\d.]+)\s+us", cast=float)
            r["combine_kernel_us"] = _grep(ks, r"Combine:\s+kernel=([\d.]+)\s+us", cast=float)

            # Dispatch recv/send bw
            d_block = re.search(r"Dispatch:.*?(?=Combine:|$)", ks, re.S)
            if d_block:
                r["dispatch_recv_bw_gbs"] = _grep(d_block.group(0), r"recv:\s+total_bw=([\d.]+)", cast=float)
                r["dispatch_send_bw_gbs"] = _grep(d_block.group(0), r"send:\s+total_bw=([\d.]+)", cast=float)
            # Combine send/recv bw
            c_block = re.search(r"Combine:.*$", ks, re.S)
            if c_block:
                r["combine_send_bw_gbs"] = _grep(c_block.group(0), r"send:\s+total_bw=([\d.]+)", cast=float)
                r["combine_recv_bw_gbs"] = _grep(c_block.group(0), r"recv:\s+total_bw=([\d.]+)", cast=float)

        # Convenience aggregate
        r["dispatch_bw_gbs"] = r.get("dispatch_recv_bw_gbs")
        r["combine_bw_gbs"] = r.get("combine_send_bw_gbs")

        # Byte counts (per-rank avg)
        bc = re.search(
            r"total_send=([\d.]+)\s+MB.*?rdma_send=([\d.]+)\s+MB.*?"
            r"rdma_recv=([\d.]+)\s+MB.*?total_recv=([\d.]+)\s+MB",
            text, re.S)
        if bc:
            r["total_send_mb"] = float(bc.group(1))
            r["rdma_send_mb"] = float(bc.group(2))
            r["rdma_recv_mb"] = float(bc.group(3))
            r["total_recv_mb"] = float(bc.group(4))
            r["nvl_send_mb"] = r["total_send_mb"] - r["rdma_send_mb"]
            r["nvl_recv_mb"] = r["total_recv_mb"] - r["rdma_recv_mb"]

    # ---------- Setup timing (both modes) ----------
    r["setup_group_ms"] = _grep(text, r"ncclEpCreateGroup:\s+avg=([\d.]+)\s+ms", cast=float)
    r["setup_handle_ms"] = _grep(text, r"ncclEpCreateHandle:\s+avg=([\d.]+)\s+ms", cast=float)

    return r


FIELDS = [
    "timestamp", "algorithm", "ranks", "tokens", "hidden", "top_k", "experts",
    "dispatch_dtype", "max_tokens_per_rank",
    "dispatch_avg_us", "dispatch_min_us", "dispatch_max_us",
    "dispatch_kernel_us",
    "dispatch_bw_gbs", "dispatch_recv_bw_gbs", "dispatch_send_bw_gbs",
    "combine_avg_us", "combine_min_us", "combine_max_us",
    "combine_kernel_us",
    "combine_bw_gbs", "combine_send_bw_gbs", "combine_recv_bw_gbs",
    "total_dc_avg_us",
    "setup_group_ms", "setup_handle_ms",
    "total_send_mb", "total_recv_mb",
    "nvl_send_mb", "nvl_recv_mb", "rdma_send_mb", "rdma_recv_mb",
    "log_path",
]


def main() -> int:
    if len(sys.argv) < 3:
        print("Usage: ep_parse.py <log_file> <csv_file> [key=value ...]",
              file=sys.stderr)
        return 1

    log_path = sys.argv[1]
    csv_path = sys.argv[2]
    extras = {}
    for kv in sys.argv[3:]:
        if "=" in kv:
            k, v = kv.split("=", 1)
            extras[k] = v

    try:
        r = parse_log(log_path)
    except FileNotFoundError:
        print(f"[parse] log not found: {log_path}", file=sys.stderr)
        return 2

    r["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    header = FIELDS + [k for k in extras if k not in FIELDS]
    exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0

    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(header)
        row = [r.get(k, "") if r.get(k) is not None else "" for k in FIELDS]
        row += [extras.get(k, "") for k in header[len(FIELDS):]]
        w.writerow(row)

    # one-line summary to stdout
    algo = r.get("algorithm") or "?"
    n = r.get("ranks") or "?"
    t = r.get("tokens") or "?"
    dt = r.get("dispatch_dtype") or "?"
    d_bw = r.get("dispatch_recv_bw_gbs") or r.get("dispatch_bw_gbs")
    c_bw = r.get("combine_send_bw_gbs") or r.get("combine_bw_gbs")
    d_us = r.get("dispatch_avg_us")
    c_us = r.get("combine_avg_us")
    bw_s = f"D={d_bw:.1f} C={c_bw:.1f} GB/s" if d_bw and c_bw else "BW=N/A"
    us_s = f"D={d_us:.1f} C={c_us:.1f} us" if d_us and c_us else "us=N/A"
    print(f"[parse] {algo:>4} EP={n} t={t:<5} {dt:>4}  {bw_s}  {us_s}  -> {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
