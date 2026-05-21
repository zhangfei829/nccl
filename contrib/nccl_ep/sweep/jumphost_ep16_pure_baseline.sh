#!/usr/bin/env bash
# =============================================================================
# jumphost_ep16_pure_baseline.sh
#
# Pure-baseline build: NO extra macros at all.
#   - NOT defined: NCCL_EP_ENABLE_HT_TMA_COPY_OVERLAP
#   - NOT defined: NCCL_EP_ENABLE_HT_COMBINE_INPUT_COPY
#   - NOT defined: NCCL_EP_ENABLE_HT_COMBINE_INPUT_STANDALONE
#   - NOT defined: NCCL_EP_HT_NV72_FULL_MATRIX
#   - NOT set: any NCCL_EP_HT_* runtime env
#
# Purpose: confirm combine baseline BW is unaffected by any of the optimization
# macros / env vars added since the V3-V8 / NO-COPY work. Single-case baseline
# only (no 3-way comparison), EP16 BF16 default tokens.
# =============================================================================
set -euo pipefail

SSH_KEY="${SSH_KEY:-id_ed25519}"
USER_NAME="${USER_NAME:-fizhang}"
TRAYS="${TRAYS:-pod4-gb300-2-tray11-f3 pod4-gb300-2-tray12-f3 pod4-gb300-2-tray13-f3 pod4-gb300-2-tray14-f3}"
HEAD_TRAY="${HEAD_TRAY:-$(echo "$TRAYS" | awk '{print $1}')}"
EP_SIZES="${EP_SIZES:-16}"
TOKENS="${TOKENS:-4096}"
# SKIP_BUILD=1 reuses cached NCCL EP binaries if libnccl_ep.so + ep_bench exist.
# Set when no C++ changes since last build on this tray set.
SKIP_BUILD="${SKIP_BUILD:-0}"
TS="$(date +%Y%m%d_%H%M%S)"
LOCAL_OUT="${LOCAL_OUT:-$HOME/nccl_ep_runs/ep16_pure_baseline_${TS}}"
NCCL_GIT_URL="${NCCL_GIT_URL:-https://github.com/zhangfei829/nccl.git}"

SSH_BASE=(ssh -i "$SSH_KEY" -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new)
SSH_BASE_N=(ssh -n -i "$SSH_KEY" -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new)

mkdir -p "$LOCAL_OUT"

cat <<EOF
===========================================================
Jumphost EP16 PURE BASELINE (no optimization macros)
  SSH_KEY  : $SSH_KEY
  TRAYS    : $TRAYS
  HEAD_TRAY: $HEAD_TRAY
  EP_SIZES : $EP_SIZES
  TOKENS   : $TOKENS
  LOCAL_OUT: $LOCAL_OUT
===========================================================
EOF

echo
echo "===== [1/5] head ssh key setup ====="
"${SSH_BASE[@]}" "$USER_NAME@$HEAD_TRAY" bash -l <<'REMOTE'
set -euo pipefail
[ -f ~/.ssh/id_ed25519 ] || ssh-keygen -t ed25519 -N '' -f ~/.ssh/id_ed25519
cp ~/.ssh/id_ed25519.pub /home/fizhang/head_tray_pub.txt
chmod 644 /home/fizhang/head_tray_pub.txt
cat /home/fizhang/head_tray_pub.txt
REMOTE

echo
echo "===== [2/5] install head pubkey on all 4 trays ====="
for tray in $TRAYS; do
  "${SSH_BASE[@]}" "$USER_NAME@$tray" bash -l <<'REMOTE'
set -euo pipefail
mkdir -p ~/.ssh && chmod 700 ~/.ssh
touch ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys
grep -qxFf /home/fizhang/head_tray_pub.txt ~/.ssh/authorized_keys || cat /home/fizhang/head_tray_pub.txt >> ~/.ssh/authorized_keys
REMOTE
done

echo
echo "===== [3/5] build NCCL EP with NO extra macros (force rebuild) ====="
"${SSH_BASE[@]}" "$USER_NAME@$HEAD_TRAY" bash -l <<REMOTE
set -euo pipefail
export MPI_HOME=/usr/mpi/gcc/openmpi-4.1.9a1
export PATH=\$MPI_HOME/bin:/usr/local/cuda/bin:\$PATH
export LD_LIBRARY_PATH=\$MPI_HOME/lib:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:\${LD_LIBRARY_PATH:-}
export CUDA_HOME=/usr/local/cuda

if [[ ! -d /home/fizhang/nccl/.git ]]; then
  git clone "$NCCL_GIT_URL" /home/fizhang/nccl
fi
cd /home/fizhang/nccl
git fetch origin master && git reset --hard origin/master

if [[ ! -f /home/fizhang/nccl/build/lib/libnccl.so ]]; then
  time make -j src.build BUILDDIR=/home/fizhang/nccl/build \\
    NVCC_GENCODE="-gencode=arch=compute_103,code=sm_103"
fi

if [[ "$SKIP_BUILD" == "1" ]] \\
   && [[ -f /home/fizhang/nccl/build/lib/libnccl_ep.so ]] \\
   && [[ -f /home/fizhang/nccl/build/test/nccl_ep/ep_bench ]]; then
  echo "[SKIP_BUILD=1] reusing cached NCCL EP binaries (caller asserts no C++ change)"
  ls -l /home/fizhang/nccl/build/lib/libnccl_ep.so /home/fizhang/nccl/build/test/nccl_ep/ep_bench
else
  # Force rebuild EP with NO macros
  rm -f /home/fizhang/nccl/build/obj/nccl_ep/nccl_ep.o \\
        /home/fizhang/nccl/build/obj/nccl_ep/device/hybridep_adapter.o \\
        /home/fizhang/nccl/build/lib/libnccl_ep.so \\
        /home/fizhang/nccl/build/test/nccl_ep/ep_bench

  time make -j3 -C contrib/nccl_ep MPI=1 BUILDDIR=/home/fizhang/nccl/build \\
    NVCC_GENCODE="-gencode=arch=compute_103,code=sm_103" \\
    MPI_HOME=\$MPI_HOME

  echo "--- verify NO optimization macros in binary ---"
  strings /home/fizhang/nccl/build/lib/libnccl_ep.so | grep -E "NCCL_EP_ENABLE_HT_(TMA_COPY|COMBINE_INPUT)" | head -5 || echo "(none, as expected)"
  ls -l /home/fizhang/nccl/build/lib/libnccl_ep.so /home/fizhang/nccl/build/test/nccl_ep/ep_bench
fi
REMOTE

REMOTE_BASE="/home/fizhang/nccl-sweeps/ep16_pure_baseline-${TS}"

echo
echo "===== [4/5] run baseline (no env) ====="
"${SSH_BASE[@]}" "$USER_NAME@$HEAD_TRAY" bash -l <<REMOTE
set -uo pipefail
export MPI_HOME=/usr/mpi/gcc/openmpi-4.1.9a1
export PATH=\$MPI_HOME/bin:/usr/local/cuda/bin:\$PATH
export LD_LIBRARY_PATH=\$MPI_HOME/lib:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:\${LD_LIBRARY_PATH:-}
export CUDA_HOME=/usr/local/cuda
export NCCL_HOME=/home/fizhang/nccl/build
cd /home/fizhang/nccl

mkdir -p $REMOTE_BASE
HOSTFILE=$REMOTE_BASE/hosts.all
{
  for tray in $TRAYS; do
    echo "\$tray slots=4"
  done
} > "\$HOSTFILE"

unset NCCL_EP_HT_DISPATCH_TMA_COPY NCCL_EP_HT_NO_COPY NCCL_EP_HT_COMBINE_INPUT_COPY NCCL_EP_HT_COMBINE_INPUT_STANDALONE || true
unset NCCL_EP_HT_NV72_NUM_SMS NCCL_EP_HT_NV72_CHUNK || true

OUT=$REMOTE_BASE/baseline
mkdir -p "\$OUT"
EP_SIZE=$EP_SIZES \\
TOKENS="$TOKENS" \\
MODES="ht_bf16" \\
LOG_DIR="\$OUT" \\
CSV_FILE="\$OUT/results.csv" \\
HOSTFILE_OVERRIDE="\$HOSTFILE" \\
bash contrib/nccl_ep/sweep/ep_sweep.sh
REMOTE

echo
echo "===== [5/5] collect + summary ====="
mkdir -p "$LOCAL_OUT/baseline"
"${SSH_BASE_N[@]}" "$USER_NAME@$HEAD_TRAY" "cat '$REMOTE_BASE/baseline/results.csv'" > "$LOCAL_OUT/baseline/results.csv"

python3 - "$LOCAL_OUT/baseline/results.csv" <<'PY'
import csv, sys
with open(sys.argv[1]) as f:
    rows = list(csv.DictReader(f))
print()
print("=== EP16 BF16 pure baseline (NO macros, NO env) ===")
print(f"{'tokens':>7} | {'d_kernel_us':>11} {'d_avg_us':>10} | {'c_kernel_us':>11} {'c_avg_us':>10} | {'send_MB':>8} | {'d_kBW':>7} {'d_API':>7} {'c_kBW':>7} {'c_API':>7}")
print("-" * 120)
for r in rows:
    t = int(r['tokens'])
    dk = float(r['dispatch_kernel_us']); da = float(r['dispatch_avg_us'])
    ck = float(r['combine_kernel_us']);  ca = float(r['combine_avg_us'])
    mb = float(r['total_send_mb'])
    d_kbw = mb/dk*1000; d_abw = mb/da*1000
    c_kbw = mb/ck*1000; c_abw = mb/ca*1000
    print(f"{t:>7} | {dk:>11.1f} {da:>10.1f} | {ck:>11.1f} {ca:>10.1f} | {mb:>8.2f} | {d_kbw:>7.1f} {d_abw:>7.1f} {c_kbw:>7.1f} {c_abw:>7.1f}")
print()
print("Compare with nocopy_compare (TMA_COPY_OVERLAP macro on, baseline case):")
print("  combine kernel ~652.6, combine API ~440.6 GB/s")
PY
