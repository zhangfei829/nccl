#!/usr/bin/env bash
# =============================================================================
# jumphost_dispatch_compare.sh
#
# One-command dispatch baseline vs TMA-copy-overlap comparison across multiple
# EP sizes (default EP4/EP8/EP16) and token counts (default 4096/8192).
# BF16 only.
#
# Steps from jumphost:
#   1. setup head-tray -> all-trays passwordless ssh
#   2. clone/pull repo on /home/fizhang/nccl
#   3. ensure main NCCL is built
#   4. rebuild NCCL EP with -DNCCL_EP_ENABLE_HT_TMA_COPY_OVERLAP
#   5. for each EP in {4,8,16}: run baseline + dispatch_overlap
#   6. collect CSV/log
#   7. print kernel BW + API BW table with ratios
# =============================================================================
set -euo pipefail

SSH_KEY="${SSH_KEY:-id_ed25519}"
USER_NAME="${USER_NAME:-fizhang}"
TRAYS="${TRAYS:-pod4-gb300-2-tray01-f3 pod4-gb300-2-tray02-f3 pod4-gb300-2-tray03-f3 pod4-gb300-2-tray04-f3}"
HEAD_TRAY="${HEAD_TRAY:-$(echo "$TRAYS" | awk '{print $1}')}"
EP_SIZES="${EP_SIZES:-4 8 16}"
TOKENS="${TOKENS:-4096 8192}"
EXTRA_BUILD_FLAGS="${EXTRA_BUILD_FLAGS:-}"
TS="$(date +%Y%m%d_%H%M%S)"
LOCAL_OUT="${LOCAL_OUT:-$HOME/nccl_ep_runs/dispatch_compare_${TS}}"
NCCL_GIT_URL="${NCCL_GIT_URL:-https://github.com/zhangfei829/nccl.git}"

SSH_BASE=(ssh -i "$SSH_KEY" -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new)
SSH_BASE_N=(ssh -n -i "$SSH_KEY" -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new)

mkdir -p "$LOCAL_OUT"

cat <<EOF
===========================================================
Jumphost dispatch baseline vs TMA-copy-overlap comparison
  SSH_KEY          : $SSH_KEY
  USER_NAME        : $USER_NAME
  HEAD_TRAY        : $HEAD_TRAY
  TRAYS            : $TRAYS
  EP_SIZES         : $EP_SIZES
  TOKENS           : $TOKENS
  EXTRA_BUILD_FLAGS: ${EXTRA_BUILD_FLAGS:-<none>}
  LOCAL_OUT        : $LOCAL_OUT
===========================================================
EOF

echo
echo "===== [1/6] Setup head-tray ssh key ====="
"${SSH_BASE[@]}" "$USER_NAME@$HEAD_TRAY" bash -l <<'REMOTE'
set -euo pipefail
[ -f ~/.ssh/id_ed25519 ] || ssh-keygen -t ed25519 -N '' -f ~/.ssh/id_ed25519
cp ~/.ssh/id_ed25519.pub /home/fizhang/head_tray_pub.txt
chmod 644 /home/fizhang/head_tray_pub.txt
cat /home/fizhang/head_tray_pub.txt
REMOTE

echo
echo "===== [2/6] Install head pubkey on all trays and verify ====="
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
echo "===== [3/6] Build NCCL EP with TMA overlap macro ====="
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

rm -f /home/fizhang/nccl/build/obj/nccl_ep/nccl_ep.o \\
      /home/fizhang/nccl/build/obj/nccl_ep/device/hybridep_adapter.o \\
      /home/fizhang/nccl/build/lib/libnccl_ep.so \\
      /home/fizhang/nccl/build/test/nccl_ep/ep_bench

time make -j3 -C contrib/nccl_ep MPI=1 BUILDDIR=/home/fizhang/nccl/build \\
  NVCC_GENCODE="-gencode=arch=compute_103,code=sm_103" \\
  EXTRA_CXXFLAGS="-DNCCL_EP_ENABLE_HT_TMA_COPY_OVERLAP $EXTRA_BUILD_FLAGS" \\
  EXTRA_NVCCFLAGS="-DNCCL_EP_ENABLE_HT_TMA_COPY_OVERLAP $EXTRA_BUILD_FLAGS" \\
  MPI_HOME=\$MPI_HOME

ls -l /home/fizhang/nccl/build/lib/libnccl_ep.so /home/fizhang/nccl/build/test/nccl_ep/ep_bench
REMOTE

REMOTE_BASE="/home/fizhang/nccl-sweeps/dispatch-${TS}"

echo
echo "===== [4/6] Run baseline + dispatch_overlap for EP=$EP_SIZES tokens=$TOKENS ====="
"${SSH_BASE[@]}" "$USER_NAME@$HEAD_TRAY" bash -l <<REMOTE
set -uo pipefail
export MPI_HOME=/usr/mpi/gcc/openmpi-4.1.9a1
export PATH=\$MPI_HOME/bin:/usr/local/cuda/bin:\$PATH
export LD_LIBRARY_PATH=\$MPI_HOME/lib:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:\${LD_LIBRARY_PATH:-}
export CUDA_HOME=/usr/local/cuda
export NCCL_HOME=/home/fizhang/nccl/build
cd /home/fizhang/nccl

# Build hostfile (all trays, slots=4 per tray).  mpirun --np EP picks first EP slots.
mkdir -p $REMOTE_BASE
HOSTFILE=$REMOTE_BASE/hosts.all
{
  for tray in $TRAYS; do
    echo "\$tray slots=4"
  done
} > "\$HOSTFILE"
echo "[hostfile]"
sed 's/^/  /' "\$HOSTFILE"

for ep in $EP_SIZES; do
  for case in baseline dispatch_overlap; do
    if [[ "\$case" == "dispatch_overlap" ]]; then
      export NCCL_EP_HT_DISPATCH_TMA_COPY=1
    else
      unset NCCL_EP_HT_DISPATCH_TMA_COPY || true
    fi
    OUT=$REMOTE_BASE/ep\${ep}/\${case}
    mkdir -p "\$OUT"
    echo
    echo "===== EP=\$ep case=\$case (NCCL_EP_HT_DISPATCH_TMA_COPY=\${NCCL_EP_HT_DISPATCH_TMA_COPY:-unset}) ====="
    EP_SIZE=\$ep \\
    TOKENS="$TOKENS" \\
    MODES="ht_bf16" \\
    LOG_DIR="\$OUT" \\
    CSV_FILE="\$OUT/results.csv" \\
    HOSTFILE_OVERRIDE="\$HOSTFILE" \\
    bash contrib/nccl_ep/sweep/ep_sweep.sh
  done
done
REMOTE

echo
echo "===== [5/6] Collect remote outputs ====="
mkdir -p "$LOCAL_OUT"
for ep in $EP_SIZES; do
  for case in baseline dispatch_overlap; do
    mkdir -p "$LOCAL_OUT/ep${ep}/${case}"
    "${SSH_BASE_N[@]}" "$USER_NAME@$HEAD_TRAY" "cat '$REMOTE_BASE/ep${ep}/${case}/results.csv'" \
      > "$LOCAL_OUT/ep${ep}/${case}/results.csv" 2>/dev/null || true
  done
done

echo
echo "===== [6/6] Comparison Tables ====="
python3 - "$LOCAL_OUT" <<'PY' | tee "$LOCAL_OUT/summary.txt"
import csv, os, sys

root = sys.argv[1]
data = {}  # data[(ep, tokens)] = {case: {dispatch_kernel_us, dispatch_avg_us, send_mb}}

for ep_dir in sorted(os.listdir(root)):
    if not ep_dir.startswith('ep'):
        continue
    ep = int(ep_dir[2:])
    for case in ('baseline', 'dispatch_overlap'):
        csvp = os.path.join(root, ep_dir, case, 'results.csv')
        if not os.path.exists(csvp):
            continue
        with open(csvp) as f:
            for r in csv.DictReader(f):
                t = int(r['tokens'])
                data.setdefault((ep, t), {})[case] = {
                    'kernel_us': float(r['dispatch_kernel_us']),
                    'avg_us':    float(r['dispatch_avg_us']),
                    'send_mb':   float(r['total_send_mb']),
                }

def bw_gbs(send_mb, time_us):
    return send_mb / time_us * 1000.0 if time_us > 0 else 0.0

# Table 1: dispatch kernel BW (GPU only)
print()
print("=== Dispatch Kernel BW (GPU-only, GB/s) ===")
print(f"{'EP':>4} {'tokens':>7} | {'baseline':>10} | {'optim':>10} | {'ratio':>8}")
print("-" * 60)
for (ep, t) in sorted(data.keys()):
    d = data[(ep, t)]
    if 'baseline' in d and 'dispatch_overlap' in d:
        b_bw = bw_gbs(d['baseline']['send_mb'], d['baseline']['kernel_us'])
        o_bw = bw_gbs(d['dispatch_overlap']['send_mb'], d['dispatch_overlap']['kernel_us'])
        r = o_bw / b_bw if b_bw > 0 else 0
        print(f"{ep:>4} {t:>7} | {b_bw:>10.1f} | {o_bw:>10.1f} | {r:>7.3f}x")

# Table 2: dispatch API BW (wall-clock incl. host op)
print()
print("=== Dispatch API BW (wall-clock incl host, GB/s) ===")
print(f"{'EP':>4} {'tokens':>7} | {'baseline':>10} | {'optim':>10} | {'ratio':>8}")
print("-" * 60)
for (ep, t) in sorted(data.keys()):
    d = data[(ep, t)]
    if 'baseline' in d and 'dispatch_overlap' in d:
        b_bw = bw_gbs(d['baseline']['send_mb'], d['baseline']['avg_us'])
        o_bw = bw_gbs(d['dispatch_overlap']['send_mb'], d['dispatch_overlap']['avg_us'])
        r = o_bw / b_bw if b_bw > 0 else 0
        print(f"{ep:>4} {t:>7} | {b_bw:>10.1f} | {o_bw:>10.1f} | {r:>7.3f}x")

# Combined table for at-a-glance
print()
print("=== Combined (kernel BW / API BW) ===")
print(f"{'EP':>4} {'tokens':>7} | {'base kernel':>11} {'opt kernel':>11} {'k ratio':>8} | {'base API':>10} {'opt API':>10} {'API ratio':>10}")
print("-" * 100)
for (ep, t) in sorted(data.keys()):
    d = data[(ep, t)]
    if 'baseline' in d and 'dispatch_overlap' in d:
        bk = bw_gbs(d['baseline']['send_mb'], d['baseline']['kernel_us'])
        ok = bw_gbs(d['dispatch_overlap']['send_mb'], d['dispatch_overlap']['kernel_us'])
        ba = bw_gbs(d['baseline']['send_mb'], d['baseline']['avg_us'])
        oa = bw_gbs(d['dispatch_overlap']['send_mb'], d['dispatch_overlap']['avg_us'])
        kr = ok / bk if bk > 0 else 0
        ar = oa / ba if ba > 0 else 0
        print(f"{ep:>4} {t:>7} | {bk:>10.1f} {ok:>10.1f} {kr:>7.3f}x | {ba:>9.1f} {oa:>9.1f} {ar:>9.3f}x")
PY

echo
echo "==========================================================="
echo "Done. Local collected files:"
find "$LOCAL_OUT" -maxdepth 3 -type f -print | sort
echo "==========================================================="
