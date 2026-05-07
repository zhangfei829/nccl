#!/usr/bin/env bash
# =============================================================================
# jumphost_ep16_focused_auto.sh
#
# One command from the jumphost to run the focused post-sweep validation:
#   - setup head-tray -> all-trays passwordless ssh
#   - ensure repo/build via run_ep16_4bay.sh with both sweeps skipped
#   - run ht_nv72_focused_validate.sh
#   - collect CSV/log/summary back to the jumphost
#
# Default scenario:
#   TRAYS=pod4-gb300-2-tray05-f3..08-f3
#   TOKENS="4096 8192"
#   NV72_PAIR_LIST="16:64 32:128 64:128"
#   REPEATS=2
# =============================================================================
set -euo pipefail

SSH_KEY="${SSH_KEY:-id_ed25519}"
USER_NAME="${USER_NAME:-fizhang}"
TRAYS="${TRAYS:-pod4-gb300-2-tray05-f3 pod4-gb300-2-tray06-f3 pod4-gb300-2-tray07-f3 pod4-gb300-2-tray08-f3}"
HEAD_TRAY="${HEAD_TRAY:-$(echo "$TRAYS" | awk '{print $1}')}"
TOKENS="${TOKENS:-4096 8192}"
NV72_PAIR_LIST="${NV72_PAIR_LIST:-16:64 32:128 64:128}"
REPEATS="${REPEATS:-2}"
TS="$(date +%Y%m%d_%H%M%S)"
LOCAL_OUT="${LOCAL_OUT:-$HOME/nccl_ep_runs/ep16_focused_${TS}}"
REMOTE_RUNBOOK_URL="${REMOTE_RUNBOOK_URL:-https://raw.githubusercontent.com/zhangfei829/nccl/master/contrib/nccl_ep/sweep/run_ep16_4bay.sh}"
REMOTE_FOCUS_URL="${REMOTE_FOCUS_URL:-https://raw.githubusercontent.com/zhangfei829/nccl/master/contrib/nccl_ep/sweep/ht_nv72_focused_validate.sh}"

SSH_BASE=(ssh -i "$SSH_KEY" -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new)

mkdir -p "$LOCAL_OUT"

cat <<EOF
===========================================================
Jumphost EP16 focused validation
  SSH_KEY       : $SSH_KEY
  USER_NAME     : $USER_NAME
  HEAD_TRAY     : $HEAD_TRAY
  TRAYS         : $TRAYS
  TOKENS        : $TOKENS
  NV72_PAIR_LIST: $NV72_PAIR_LIST
  REPEATS       : $REPEATS
  LOCAL_OUT     : $LOCAL_OUT
===========================================================
EOF

echo
echo "===== [1/5] Setup head-tray ssh key ====="
"${SSH_BASE[@]}" "$USER_NAME@$HEAD_TRAY" bash -l <<'REMOTE'
set -euo pipefail
[ -f ~/.ssh/id_ed25519 ] || ssh-keygen -t ed25519 -N '' -f ~/.ssh/id_ed25519
cp ~/.ssh/id_ed25519.pub /home/fizhang/head_tray_pub.txt
chmod 644 /home/fizhang/head_tray_pub.txt
cat /home/fizhang/head_tray_pub.txt
REMOTE

echo
echo "===== [2/5] Install head-tray pubkey on all trays and verify ====="
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

"${SSH_BASE[@]}" "$USER_NAME@$HEAD_TRAY" bash -l <<REMOTE
set -euo pipefail
for h in $TRAYS; do
  printf "%-30s -> " "\$h"
  ssh -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new "\$h" hostname
done
REMOTE

echo
echo "===== [3/5] Ensure repo/build exists (no benchmark sweeps) ====="
"${SSH_BASE[@]}" "$USER_NAME@$HEAD_TRAY" bash -l <<REMOTE
set -euo pipefail
export MPI_HOME=/usr/mpi/gcc/openmpi-4.1.9a1
export PATH=\$MPI_HOME/bin:/usr/local/cuda/bin:\$PATH
export LD_LIBRARY_PATH=\$MPI_HOME/lib:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:\${LD_LIBRARY_PATH:-}
export CUDA_HOME=/usr/local/cuda
curl -sL "$REMOTE_RUNBOOK_URL" -o /tmp/run_ep16_4bay.sh
TRAYS="$TRAYS" SKIP_BASELINE=1 SKIP_NV72_CALIBRATE=1 bash /tmp/run_ep16_4bay.sh
REMOTE

echo
echo "===== [4/5] Run focused validation ====="
"${SSH_BASE[@]}" "$USER_NAME@$HEAD_TRAY" bash -l <<REMOTE
set -euo pipefail
export MPI_HOME=/usr/mpi/gcc/openmpi-4.1.9a1
export PATH=\$MPI_HOME/bin:/usr/local/cuda/bin:\$PATH
export LD_LIBRARY_PATH=\$MPI_HOME/lib:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:\${LD_LIBRARY_PATH:-}
export CUDA_HOME=/usr/local/cuda

cd /home/fizhang/nccl
git fetch origin master && git reset --hard origin/master

FOCUS_OUT=/home/fizhang/nccl-sweeps/nccl-focused-${TS}-ep16
mkdir -p "\$FOCUS_OUT"
HOSTFILE="\$FOCUS_OUT/hosts.ep16"
: > "\$HOSTFILE"
for tray in $TRAYS; do
    echo "\$tray slots=4" >> "\$HOSTFILE"
done

OUT="\$FOCUS_OUT" \\
TRAYS="$TRAYS" \\
TOKENS="$TOKENS" \\
NV72_PAIR_LIST="$NV72_PAIR_LIST" \\
REPEATS="$REPEATS" \\
HOSTFILE_OVERRIDE="\$HOSTFILE" \\
bash contrib/nccl_ep/sweep/ht_nv72_focused_validate.sh 2>&1 | tee /home/fizhang/run_ep16_focused_${TS}.log
REMOTE

echo
echo "===== [5/5] Collect focused results ====="
REMOTE_INFO="$("${SSH_BASE[@]}" "$USER_NAME@$HEAD_TRAY" bash -l <<'REMOTE'
set -u
OUT=$(ls -td /home/fizhang/nccl-sweeps/nccl-focused-*-ep16 2>/dev/null | head -1 || true)
LOG=$(ls -t /home/fizhang/run_ep16_focused_*.log 2>/dev/null | head -1 || true)
echo "OUT=$OUT"
echo "LOG=$LOG"
REMOTE
)"
echo "$REMOTE_INFO" | tee "$LOCAL_OUT/remote_paths.txt"
REMOTE_OUT="$(echo "$REMOTE_INFO" | awk -F= '/^OUT=/{print $2}')"
REMOTE_LOG="$(echo "$REMOTE_INFO" | awk -F= '/^LOG=/{print $2}')"

if [[ -z "$REMOTE_OUT" ]]; then
    echo "[jumphost_ep16_focused_auto] ERROR: remote OUT not found" >&2
    exit 3
fi

"${SSH_BASE[@]}" "$USER_NAME@$HEAD_TRAY" "cat '$REMOTE_OUT/ht_nv72_focused_validate.csv'" > "$LOCAL_OUT/ht_nv72_focused_validate.csv" || true
if [[ -n "$REMOTE_LOG" ]]; then
    "${SSH_BASE[@]}" "$USER_NAME@$HEAD_TRAY" "cat '$REMOTE_LOG'" > "$LOCAL_OUT/run.log" || true
fi

"${SSH_BASE[@]}" "$USER_NAME@$HEAD_TRAY" bash -l <<REMOTE | tee "$LOCAL_OUT/summary.txt"
set -u
echo "===== remote OUT: $REMOTE_OUT ====="
echo "===== log tail ====="
if [[ -n "$REMOTE_LOG" && -f "$REMOTE_LOG" ]]; then tail -160 "$REMOTE_LOG"; fi
echo "===== focused csv ====="
cat "$REMOTE_OUT/ht_nv72_focused_validate.csv" 2>/dev/null || true
REMOTE

echo
echo "==========================================================="
echo "Done. Local collected files:"
find "$LOCAL_OUT" -maxdepth 1 -type f -print | sort
echo "==========================================================="
