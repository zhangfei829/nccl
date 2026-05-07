#!/usr/bin/env bash
# =============================================================================
# jumphost_ep16_4bay_auto.sh
#
# One command from the jumphost:
#   1. set up head-tray -> all-trays passwordless ssh
#   2. run EP16 HT BF16 NV72 tuning on the remote head tray
#   3. collect result CSVs + summary tables back to the jumphost
#
# Default scenario (2026-05-07):
#   head tray : pod4-gb300-2-tray05-f3
#   trays     : pod4-gb300-2-tray05-f3 .. pod4-gb300-2-tray08-f3
#   backend   : HT only
#   dtype     : BF16 only
#   tokens    : 128 256 4096 8192
#   tuning    : NUM_SMS={16,32,64} x CHUNK={64,128,256}
#
# Usage:
#   bash <(curl -sL https://raw.githubusercontent.com/zhangfei829/nccl/master/contrib/nccl_ep/sweep/jumphost_ep16_4bay_auto.sh)
#
# Env overrides:
#   SSH_KEY        default id_ed25519
#   USER_NAME      default fizhang
#   HEAD_TRAY      default first item of TRAYS
#   TRAYS          default tray05..08
#   TOKENS         default "128 256 4096 8192"
#   BASELINE_MODES default "ht_bf16"
#   LOCAL_OUT      default $HOME/nccl_ep_runs/ep16_4bay_<ts>
#
# Notes:
#   - This script intentionally does not expose FULLMESH. FULLMESH hung at
#     EP16 t=16 in this environment and is not part of the current test plan.
#   - The remote runbook is fetched from origin/master at runtime so the
#     jumphost does not need a local checkout.
# =============================================================================
set -euo pipefail

SSH_KEY="${SSH_KEY:-id_ed25519}"
USER_NAME="${USER_NAME:-fizhang}"
TRAYS="${TRAYS:-pod4-gb300-2-tray05-f3 pod4-gb300-2-tray06-f3 pod4-gb300-2-tray07-f3 pod4-gb300-2-tray08-f3}"
HEAD_TRAY="${HEAD_TRAY:-$(echo "$TRAYS" | awk '{print $1}')}"
TOKENS="${TOKENS:-128 256 4096 8192}"
BASELINE_MODES="${BASELINE_MODES:-ht_bf16}"
TS="$(date +%Y%m%d_%H%M%S)"
LOCAL_OUT="${LOCAL_OUT:-$HOME/nccl_ep_runs/ep16_4bay_${TS}}"
REMOTE_RUNBOOK_URL="${REMOTE_RUNBOOK_URL:-https://raw.githubusercontent.com/zhangfei829/nccl/master/contrib/nccl_ep/sweep/run_ep16_4bay.sh}"

SSH_BASE=(ssh -i "$SSH_KEY" -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new)

mkdir -p "$LOCAL_OUT"

cat <<EOF
===========================================================
Jumphost EP16 4-BAY automation
  SSH_KEY        : $SSH_KEY
  USER_NAME      : $USER_NAME
  HEAD_TRAY      : $HEAD_TRAY
  TRAYS          : $TRAYS
  TOKENS         : $TOKENS
  BASELINE_MODES : $BASELINE_MODES
  LOCAL_OUT      : $LOCAL_OUT
===========================================================
EOF

echo
echo "===== [1/5] Generate head-tray key and publish pubkey to shared /home ====="
"${SSH_BASE[@]}" "$USER_NAME@$HEAD_TRAY" bash -l <<'REMOTE'
set -euo pipefail
[ -f ~/.ssh/id_ed25519 ] || ssh-keygen -t ed25519 -N '' -f ~/.ssh/id_ed25519
cp ~/.ssh/id_ed25519.pub /home/fizhang/head_tray_pub.txt
chmod 644 /home/fizhang/head_tray_pub.txt
echo "head pub:"
cat /home/fizhang/head_tray_pub.txt
REMOTE

echo
echo "===== [2/5] Install head-tray pubkey on every tray local authorized_keys ====="
for tray in $TRAYS; do
    echo "=== install on $tray ==="
    "${SSH_BASE[@]}" "$USER_NAME@$tray" bash -l <<'REMOTE'
set -euo pipefail
mkdir -p ~/.ssh && chmod 700 ~/.ssh
touch ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys
grep -qxFf /home/fizhang/head_tray_pub.txt ~/.ssh/authorized_keys || cat /home/fizhang/head_tray_pub.txt >> ~/.ssh/authorized_keys
echo "$(hostname): $(wc -l < ~/.ssh/authorized_keys) authorized_keys lines"
REMOTE
done

echo
echo "===== [3/5] Verify head-tray -> all trays passwordless ssh ====="
"${SSH_BASE[@]}" "$USER_NAME@$HEAD_TRAY" bash -l <<REMOTE
set -euo pipefail
for h in $TRAYS; do
  printf "%-30s -> " "\$h"
  ssh -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new "\$h" hostname
done
REMOTE

echo
echo "===== [4/5] Run remote EP16 HT BF16 tuning ====="
"${SSH_BASE[@]}" "$USER_NAME@$HEAD_TRAY" bash -l <<REMOTE
set -euo pipefail
export MPI_HOME=/usr/mpi/gcc/openmpi-4.1.9a1
export PATH=\$MPI_HOME/bin:/usr/local/cuda/bin:\$PATH
export LD_LIBRARY_PATH=\$MPI_HOME/lib:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:\${LD_LIBRARY_PATH:-}
export CUDA_HOME=/usr/local/cuda

curl -sL "$REMOTE_RUNBOOK_URL" -o /tmp/run_ep16_4bay.sh

TRAYS="$TRAYS" \\
TOKENS="$TOKENS" \\
BASELINE_MODES="$BASELINE_MODES" \\
bash /tmp/run_ep16_4bay.sh 2>&1 | tee /home/fizhang/run_ep16_auto_${TS}.log
REMOTE

echo
echo "===== [5/5] Collect remote results ====="
REMOTE_INFO="$("${SSH_BASE[@]}" "$USER_NAME@$HEAD_TRAY" bash -l <<'REMOTE'
set -u
OUT=$(ls -td /home/fizhang/nccl-sweeps/*-ep16-4bay 2>/dev/null | head -1 || true)
LOG=$(ls -t /home/fizhang/run_ep16_auto_*.log \
            /home/fizhang/run_ep16_tray*_bf16_*.log \
            /home/fizhang/run_ep16_htonly_*.log \
            /home/fizhang/run_ep16_*.log 2>/dev/null | head -1 || true)
echo "OUT=$OUT"
echo "LOG=$LOG"
REMOTE
)"
echo "$REMOTE_INFO" | tee "$LOCAL_OUT/remote_paths.txt"
REMOTE_OUT="$(echo "$REMOTE_INFO" | awk -F= '/^OUT=/{print $2}')"
REMOTE_LOG="$(echo "$REMOTE_INFO" | awk -F= '/^LOG=/{print $2}')"

if [[ -z "$REMOTE_OUT" ]]; then
    echo "[jumphost_ep16_4bay_auto] ERROR: remote OUT not found" >&2
    exit 3
fi

"${SSH_BASE[@]}" "$USER_NAME@$HEAD_TRAY" "cat '$REMOTE_OUT/baseline/results.csv'" > "$LOCAL_OUT/baseline_results.csv" || true
"${SSH_BASE[@]}" "$USER_NAME@$HEAD_TRAY" "cat '$REMOTE_OUT/ht_nv72_calibrate.csv'" > "$LOCAL_OUT/ht_nv72_calibrate.csv" || true
if [[ -n "$REMOTE_LOG" ]]; then
    "${SSH_BASE[@]}" "$USER_NAME@$HEAD_TRAY" "cat '$REMOTE_LOG'" > "$LOCAL_OUT/run.log" || true
fi

"${SSH_BASE[@]}" "$USER_NAME@$HEAD_TRAY" bash -l <<REMOTE | tee "$LOCAL_OUT/summary.txt"
set -euo pipefail
OUT="$REMOTE_OUT"
echo "===== remote OUT: \$OUT ====="
echo
echo "===== latest run log tail ====="
if [[ -n "$REMOTE_LOG" && -f "$REMOTE_LOG" ]]; then
    tail -120 "$REMOTE_LOG"
fi
echo
echo "===== baseline/results.csv ====="
cat "\$OUT/baseline/results.csv" 2>/dev/null || true
echo
echo "===== phase-specific best from existing cell CSVs ====="
python3 - <<'PY'
import csv, glob, re
OUT="$REMOTE_OUT"
best_d, best_c = {}, {}
for path in glob.glob(f"{OUT}/ht_nv72_sms*_chunk*/results.csv"):
    m = re.search(r'ht_nv72_sms(\d+)_chunk(\d+)', path)
    if not m:
        continue
    sms, chunk = m.group(1), m.group(2)
    with open(path) as f:
        for r in csv.DictReader(f):
            t = int(r["tokens"])
            dt = r.get("dispatch_dtype_tag") or r.get("dispatch_dtype") or "?"
            d = float(r["dispatch_kernel_us"] or "nan")
            c = float(r["combine_kernel_us"] or "nan")
            key = (t, dt)
            if d == d and (key not in best_d or d < best_d[key][0]):
                best_d[key] = (d, sms, chunk)
            if c == c and (key not in best_c or c < best_c[key][0]):
                best_c[key] = (c, sms, chunk)
print("tokens dtype | best_dispatch_us sms chunk | best_combine_us sms chunk")
for key in sorted(set(best_d) | set(best_c)):
    bd = best_d.get(key, (float("nan"), "-", "-"))
    bc = best_c.get(key, (float("nan"), "-", "-"))
    print(f"{key[0]:>6} {key[1]:>5} | {bd[0]:>16.1f} {bd[1]:>3} {bd[2]:>5} | {bc[0]:>15.1f} {bc[1]:>3} {bc[2]:>5}")
PY
echo
echo "===== single-config D+C total best ====="
python3 - <<'PY'
import csv, glob, re
OUT="$REMOTE_OUT"
best = {}
default = {}
for path in glob.glob(f"{OUT}/ht_nv72_sms*_chunk*/results.csv"):
    m = re.search(r'ht_nv72_sms(\d+)_chunk(\d+)', path)
    if not m:
        continue
    sms, chunk = m.group(1), m.group(2)
    with open(path) as f:
        for r in csv.DictReader(f):
            t = int(r["tokens"])
            dt = r.get("dispatch_dtype_tag") or r.get("dispatch_dtype") or "?"
            d = float(r["dispatch_kernel_us"] or "nan")
            c = float(r["combine_kernel_us"] or "nan")
            if d != d or c != c:
                continue
            key = (t, dt)
            total = d + c
            if key not in best or total < best[key][0]:
                best[key] = (total, d, c, sms, chunk)
            if sms == "16" and chunk == "64":
                default[key] = (total, d, c)
print("tokens dtype | best_total_us dispatch_us combine_us sms chunk | default_total_us speedup")
for key in sorted(best):
    total, d, c, sms, chunk = best[key]
    if key in default:
        dft_total = default[key][0]
        print(f"{key[0]:>6} {key[1]:>5} | {total:>13.1f} {d:>11.1f} {c:>10.1f} {sms:>3} {chunk:>5} | {dft_total:>16.1f} {dft_total/total:>7.2f}x")
    else:
        print(f"{key[0]:>6} {key[1]:>5} | {total:>13.1f} {d:>11.1f} {c:>10.1f} {sms:>3} {chunk:>5} | {'-':>16} {'-':>7}")
PY
REMOTE

echo
echo "==========================================================="
echo "Done. Local collected files:"
find "$LOCAL_OUT" -maxdepth 1 -type f -print | sort
echo "==========================================================="
