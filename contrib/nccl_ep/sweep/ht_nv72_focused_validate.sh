#!/usr/bin/env bash
# =============================================================================
# ht_nv72_focused_validate.sh
#
# Focused validation for EP16 HT NV72 tuning after the 9-cell sweep.
#
# Default test matrix:
#   tokens : 4096 8192
#   pairs  : (16,64) (32,128) (64,128)
#   repeats: 2
#   mode   : ht_bf16
#
# This script reuses ep_sweep.sh and therefore supports both Slurm and manual
# HOSTFILE_OVERRIDE mode. It collects per-run CSVs into:
#
#   $OUT/ht_nv72_focused_validate.csv
#
# and prints aggregate average dispatch/combine/total kernel time per
# (tokens, dtype, NUM_SMS, CHUNK), plus the best single config per token.
# =============================================================================
set -u
# Do not set -e: one failed cell should not kill the whole validation.

EP_SIZE="${EP_SIZE:-16}"
TOKENS="${TOKENS:-4096 8192}"
NV72_PAIR_LIST="${NV72_PAIR_LIST:-16:64 32:128 64:128}"
REPEATS="${REPEATS:-2}"
TS="$(date +%Y%m%d_%H%M%S)"
OUT="${OUT:-/home/fizhang/nccl-sweeps/nccl-focus-${TS}-ep16}"

mkdir -p "$OUT"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
EP_SWEEP="$SCRIPT_DIR/ep_sweep.sh"
if [[ ! -f "$EP_SWEEP" ]]; then
    echo "[ht_nv72_focused_validate] ERROR: $EP_SWEEP not found" >&2
    exit 2
fi

COMBINED_CSV="$OUT/ht_nv72_focused_validate.csv"
: > "$COMBINED_CSV"

cat <<EOF
===========================================================
HT NV72 focused validation
  EP_SIZE          : $EP_SIZE
  TOKENS           : $TOKENS
  NV72_PAIR_LIST   : $NV72_PAIR_LIST
  REPEATS          : $REPEATS
  OUT              : $OUT
  HOSTFILE_OVERRIDE: ${HOSTFILE_OVERRIDE:-<unset, will use SLURM_JOB_NODELIST>}
===========================================================
EOF

total_cells=$(( $(echo "$NV72_PAIR_LIST" | wc -w) * REPEATS ))
cell_idx=0

for rep in $(seq 1 "$REPEATS"); do
    for pair in $NV72_PAIR_LIST; do
        sms="${pair%%:*}"
        chunk="${pair##*:}"
        cell_idx=$((cell_idx + 1))
        cell_dir="$OUT/focus_rep${rep}_sms${sms}_chunk${chunk}"
        cell_csv="$cell_dir/results.csv"

        echo
        echo "##### [$cell_idx/$total_cells] rep=$rep NUM_SMS=$sms CHUNK=$chunk #####"
        mkdir -p "$cell_dir"

        NCCL_EP_HT_NV72_NUM_SMS="$sms" NCCL_EP_HT_NV72_CHUNK="$chunk" \
        EP_SIZE="$EP_SIZE" TOKENS="$TOKENS" MODES="ht_bf16" \
        LOG_DIR="$cell_dir" CSV_FILE="$cell_csv" \
        EXTRA_BENCH_ARGS="${EXTRA_BENCH_ARGS:-}" \
        HOSTFILE_OVERRIDE="${HOSTFILE_OVERRIDE:-}" \
            bash "$EP_SWEEP"
        rc=$?
        if [[ $rc -ne 0 ]]; then
            echo "[ht_nv72_focused_validate] WARN: rep=$rep sms=$sms chunk=$chunk exited rc=$rc; continuing"
        fi
    done
done

echo
echo "==========================================================="
echo "All focused cells done. Rebuilding combined CSV: $COMBINED_CSV"
echo "==========================================================="

if command -v python3 >/dev/null 2>&1; then
    python3 - "$OUT" "$COMBINED_CSV" <<'PY'
import csv, glob, os, re, statistics, sys

out, combined_path = sys.argv[1], sys.argv[2]

rows = []
fieldnames = None
for path in sorted(glob.glob(os.path.join(out, "focus_rep*_sms*_chunk*", "results.csv"))):
    m = re.search(r"focus_rep(\d+)_sms(\d+)_chunk(\d+)", path)
    if not m:
        continue
    rep, sms, chunk = m.group(1), m.group(2), m.group(3)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            continue
        if fieldnames is None:
            fieldnames = list(reader.fieldnames) + ["repeat", "nv72_num_sms", "nv72_chunk"]
        for row in reader:
            row["repeat"] = rep
            row["nv72_num_sms"] = sms
            row["nv72_chunk"] = chunk
            rows.append(row)

if fieldnames is None:
    print(f"[ht_nv72_focused_validate] WARN: no per-cell results.csv found under {out}", file=sys.stderr)
    sys.exit(0)

with open(combined_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)

print(f"[ht_nv72_focused_validate] wrote {len(rows)} rows -> {combined_path}")

groups = {}
for row in rows:
    try:
        tk = int(row["tokens"])
        dt = row.get("dispatch_dtype_tag") or row.get("dispatch_dtype") or "?"
        sms = row["nv72_num_sms"]
        chunk = row["nv72_chunk"]
        d = float(row.get("dispatch_kernel_us") or "nan")
        c = float(row.get("combine_kernel_us") or "nan")
    except Exception:
        continue
    if d != d or c != c:
        continue
    groups.setdefault((tk, dt, sms, chunk), []).append((d, c, d + c))

print()
print("=== Focused validation avg by config ===")
print(f"{'tokens':>6} {'dtype':>5} {'sms':>3} {'chunk':>5} | {'n':>2} {'dispatch_avg':>12} {'combine_avg':>11} {'total_avg':>10} | {'total_min':>9} {'total_max':>9}")
agg = {}
for key in sorted(groups):
    vals = groups[key]
    ds = [x[0] for x in vals]
    cs = [x[1] for x in vals]
    ts = [x[2] for x in vals]
    row = (statistics.mean(ds), statistics.mean(cs), statistics.mean(ts), min(ts), max(ts), len(vals))
    agg[key] = row
    tk, dt, sms, chunk = key
    print(f"{tk:>6} {dt:>5} {sms:>3} {chunk:>5} | {len(vals):>2} {row[0]:>12.1f} {row[1]:>11.1f} {row[2]:>10.1f} | {row[3]:>9.1f} {row[4]:>9.1f}")

print()
print("=== Best single config per (tokens, dtype), ranked by total_avg ===")
print(f"{'tokens':>6} {'dtype':>5} | {'sms':>3} {'chunk':>5} {'dispatch_avg':>12} {'combine_avg':>11} {'total_avg':>10}")
best = {}
for key, row in agg.items():
    tk, dt, sms, chunk = key
    bkey = (tk, dt)
    if bkey not in best or row[2] < best[bkey][0][2]:
        best[bkey] = (row, sms, chunk)
for bkey in sorted(best):
    row, sms, chunk = best[bkey]
    tk, dt = bkey
    print(f"{tk:>6} {dt:>5} | {sms:>3} {chunk:>5} {row[0]:>12.1f} {row[1]:>11.1f} {row[2]:>10.1f}")
PY
fi

echo
echo "Focused validation done: $COMBINED_CSV"
