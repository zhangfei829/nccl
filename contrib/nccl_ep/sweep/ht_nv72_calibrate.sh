#!/usr/bin/env bash
# =============================================================================
# ht_nv72_calibrate.sh
#
# Sweep the HT NV72 fabric-only template tuning knobs and pick the best
# (NUM_SMS, CHUNK) per token count.  Used to seed the P1 tuning JSON
# (contrib/nccl_ep/tuning_configs/sm103_gb300_HT_ep{N}.json) referenced in
# docs/MORI_VS_NCCL_EP.md.
#
# What it does:
#   For each (NUM_SMS in {16,32,64}) x (CHUNK in {64,128,256}) = 9 cells:
#     - run ep_sweep.sh with NCCL_EP_HT_NV72_NUM_SMS / _CHUNK env set
#     - per-cell LOG_DIR / CSV under $OUT/ht_nv72_sms${sms}_chunk${chunk}/
#   After all 9 cells finish:
#     - merge per-cell CSVs into $OUT/ht_nv72_calibrate.csv with extra
#       'nv72_num_sms,nv72_chunk' columns appended
#     - run a small python pass to print the best (NUM_SMS, CHUNK) per
#       token count, ranked by dispatch_kernel_us
#
# Usage (must already be inside an salloc shell, OR pass HOSTFILE_OVERRIDE):
#   # Slurm path (auto): inside a 4-node salloc
#   bash contrib/nccl_ep/sweep/ht_nv72_calibrate.sh
#
#   # Manual-reserved nodes (no Slurm): build a hostfile first, then run
#   cat > /tmp/hosts.ep16 <<EOF
#   pod4-gb300-2-tray01-f3 slots=4
#   pod4-gb300-2-tray02-f3 slots=4
#   pod4-gb300-2-tray03-f3 slots=4
#   pod4-gb300-2-tray04-f3 slots=4
#   EOF
#   HOSTFILE_OVERRIDE=/tmp/hosts.ep16 \
#     bash contrib/nccl_ep/sweep/ht_nv72_calibrate.sh
#
# Env overrides:
#   EP_SIZE             default 16
#   TOKENS              default "128 256 4096 8192"
#   NV72_NUM_SMS_LIST   default "16 32 64"
#   NV72_CHUNK_LIST     default "64 128 256"
#   OUT                 default $HOME/fizhang/nccl-sweep-<ts>-nv72cal
#   HOSTFILE_OVERRIDE   forwarded to ep_sweep.sh (no-Slurm path)
#   EXTRA_BENCH_ARGS    forwarded to ep_sweep.sh
#
# Notes:
#   - Only ht_bf16 is calibrated. HT NV72 knobs do NOT affect FP8 (FP8 path
#     skips the NV72 template switch since combine doesn't support FP8) so
#     fp8 sweep is wasted; if you really want FP8, pass MODES="ht_bf16 ht_fp8"
#     by editing the wrapper.
#   - Per-cell failure (e.g. NUM_SMS=64 + large token blowing smem) does NOT
#     kill the sweep; the cell's CSV will be missing rows for that mode/token
#     but the wrapper continues. Check $OUT/sweep.log for FAIL lines.
# =============================================================================
set -u
# DON'T set -e: single cell fail should not kill the whole 9-cell sweep.

EP_SIZE="${EP_SIZE:-16}"
TOKENS="${TOKENS:-128 256 4096 8192}"
NV72_NUM_SMS_LIST="${NV72_NUM_SMS_LIST:-16 32 64}"
NV72_CHUNK_LIST="${NV72_CHUNK_LIST:-64 128 256}"
TS="$(date +%Y%m%d_%H%M%S)"
OUT="${OUT:-$HOME/fizhang/nccl-sweep-${TS}-nv72cal}"

mkdir -p "$OUT"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
EP_SWEEP="$SCRIPT_DIR/ep_sweep.sh"
if [[ ! -f "$EP_SWEEP" ]]; then
    echo "[ht_nv72_calibrate] ERROR: $EP_SWEEP not found" >&2
    exit 2
fi

COMBINED_CSV="$OUT/ht_nv72_calibrate.csv"
: > "$COMBINED_CSV"

echo "==========================================================="
echo "HT NV72 calibrate sweep"
echo "  EP_SIZE          : $EP_SIZE"
echo "  TOKENS           : $TOKENS"
echo "  NV72_NUM_SMS_LIST: $NV72_NUM_SMS_LIST"
echo "  NV72_CHUNK_LIST  : $NV72_CHUNK_LIST"
echo "  OUT              : $OUT"
echo "  HOSTFILE_OVERRIDE: ${HOSTFILE_OVERRIDE:-<unset, will use SLURM_JOB_NODELIST>}"
echo "==========================================================="

cell_idx=0
total_cells=$(( $(echo $NV72_NUM_SMS_LIST | wc -w) * $(echo $NV72_CHUNK_LIST | wc -w) ))

for sms in $NV72_NUM_SMS_LIST; do
    for chunk in $NV72_CHUNK_LIST; do
        cell_idx=$((cell_idx + 1))
        cell_dir="$OUT/ht_nv72_sms${sms}_chunk${chunk}"
        cell_csv="$cell_dir/results.csv"
        echo
        echo "##### [$cell_idx/$total_cells] NV72 NUM_SMS=$sms CHUNK=$chunk #####"
        mkdir -p "$cell_dir"

        NCCL_EP_HT_NV72_NUM_SMS="$sms" NCCL_EP_HT_NV72_CHUNK="$chunk" \
        EP_SIZE="$EP_SIZE" TOKENS="$TOKENS" MODES="ht_bf16" \
        LOG_DIR="$cell_dir" CSV_FILE="$cell_csv" \
        EXTRA_BENCH_ARGS="${EXTRA_BENCH_ARGS:-}" \
        HOSTFILE_OVERRIDE="${HOSTFILE_OVERRIDE:-}" \
            bash "$EP_SWEEP"
        rc=$?
        if [[ $rc -ne 0 ]]; then
            echo "[ht_nv72_calibrate] WARN: cell sms=$sms chunk=$chunk exited rc=$rc; continuing"
        fi

        if [[ ! -f "$cell_csv" ]]; then
            echo "[ht_nv72_calibrate] WARN: $cell_csv missing; skip"
            continue
        fi
    done
done

echo
echo "==========================================================="
echo "All $total_cells cells done. Rebuilding combined CSV: $COMBINED_CSV"
echo "==========================================================="

if command -v python3 >/dev/null 2>&1; then
    python3 - "$OUT" "$COMBINED_CSV" <<'PY'
import csv, glob, os, re, sys

out, combined_path = sys.argv[1], sys.argv[2]

rows = []
fieldnames = None
for path in sorted(glob.glob(os.path.join(out, "ht_nv72_sms*_chunk*", "results.csv"))):
    m = re.search(r"ht_nv72_sms(\d+)_chunk(\d+)", path)
    if not m:
        continue
    sms, ch = m.group(1), m.group(2)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            continue
        if fieldnames is None:
            fieldnames = list(reader.fieldnames) + ["nv72_num_sms", "nv72_chunk"]
        for row in reader:
            row["nv72_num_sms"] = sms
            row["nv72_chunk"] = ch
            rows.append(row)

if fieldnames is None:
    print(f"[ht_nv72_calibrate] WARN: no per-cell results.csv found under {out}", file=sys.stderr)
    sys.exit(0)

with open(combined_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)

print(f"[ht_nv72_calibrate] wrote {len(rows)} rows -> {combined_path}")

# (tokens, dtype) -> (best_dispatch_us, best_combine_us, sms, chunk)
best_d = {}
best_c = {}
all_d = {}
all_c = {}

for row in rows:
    try:
        tk = int(row.get("tokens", "0"))
        dt = row.get("dispatch_dtype_tag") or row.get("dispatch_dtype") or "?"
        sms = row.get("nv72_num_sms", "?")
        ch = row.get("nv72_chunk", "?")
    except Exception:
        continue
    try:
        dus = float(row.get("dispatch_kernel_us") or "nan")
    except ValueError:
        dus = float("nan")
    try:
        cus = float(row.get("combine_kernel_us") or "nan")
    except ValueError:
        cus = float("nan")
    key = (tk, dt)
    all_d.setdefault(key, []).append((dus, sms, ch))
    all_c.setdefault(key, []).append((cus, sms, ch))
    if dus == dus:  # not nan
        cur = best_d.get(key)
        if (cur is None) or (dus < cur[0]):
            best_d[key] = (dus, sms, ch)
    if cus == cus:
        cur = best_c.get(key)
        if (cur is None) or (cus < cur[0]):
            best_c[key] = (cus, sms, ch)

print()
print("=== Best (NUM_SMS, CHUNK) per (tokens, dtype) ===")
print(f"{'tokens':>6} {'dtype':>6} | {'best_dispatch_us':>18} {'sms':>4} {'chunk':>5} | {'best_combine_us':>17} {'sms':>4} {'chunk':>5}")
for key in sorted(set(best_d.keys()) | set(best_c.keys())):
    tk, dt = key
    bd = best_d.get(key, (float("nan"), "-", "-"))
    bc = best_c.get(key, (float("nan"), "-", "-"))
    print(f"{tk:>6} {dt:>6} | {bd[0]:>18.1f} {bd[1]:>4} {bd[2]:>5} | {bc[0]:>17.1f} {bc[1]:>4} {bc[2]:>5}")
print()

print("=== Default (NUM_SMS=16, CHUNK=64) baseline vs best dispatch ===")
print(f"{'tokens':>6} {'dtype':>6} | {'default_us':>11} | {'best_us':>11} {'sms':>4} {'chunk':>5} | {'speedup':>8}")
for key in sorted(best_d.keys()):
    tk, dt = key
    default = next(((dus, sms, ch) for dus, sms, ch in all_d[key]
                    if str(sms) == "16" and str(ch) == "64"), None)
    if default is None:
        continue
    bus, bsms, bch = best_d[key]
    if default[0] == default[0] and bus == bus and bus > 0:
        speedup = default[0] / bus
        print(f"{tk:>6} {dt:>6} | {default[0]:>11.1f} | {bus:>11.1f} {bsms:>4} {bch:>5} | {speedup:>7.2f}x")
PY
fi

echo
echo "Next: copy the 'Best (NUM_SMS, CHUNK)' table above into a draft P1 JSON at"
echo "  contrib/nccl_ep/tuning_configs/sm103_gb300_HT_ep${EP_SIZE}.json"
echo "Format follows the MORI tuning_configs schema; see"
echo "  contrib/nccl_ep/docs/MORI_VS_NCCL_EP.md section 6 P1."
