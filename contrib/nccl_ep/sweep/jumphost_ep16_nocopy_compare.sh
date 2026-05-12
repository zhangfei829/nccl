#!/usr/bin/env bash
# =============================================================================
# jumphost_ep16_nocopy_compare.sh
#
# Three-way comparison: baseline vs dispatch_overlap vs no_copy.
#
# baseline        : default HT path (host cudaMemcpyAsync for dispatch input
#                   (multi-node only), dispatch output, combine input).
# dispatch_overlap: NCCL_EP_HT_DISPATCH_TMA_COPY=1 -> skip dispatch output
#                   cudaMemcpyAsync; in-kernel TMA copy writes recv_x.
# no_copy         : NCCL_EP_HT_NO_COPY=1 -> skip ALL host cudaMemcpyAsync
#                   (dispatch input + dispatch output + combine input). Kernel
#                   does no in-kernel COPY warps either (tma_overlap forced
#                   off). Caller is expected to read/write NCCL's internal
#                   staging buffers directly via:
#                     ncclEpHtGetDispatchOutputBuffer(handle, &ptr, &maxsz)
#                     ncclEpHtGetCombineInputBuffer  (handle, &ptr, &maxsz)
#                   ep_bench uses those APIs in its --validate path to bridge
#                   STAGE<->user buffers ONLY for the one-shot validation
#                   iteration (NOT in the timing loop), so calc_diff is
#                   meaningful while the timing data still reflects true
#                   no-copy cost.
#
# Default: EP=16, BF16, tokens 4096/8192, all 3 cases.
# Same binary handles all 3 cases (env-only switching, no rebuild).
#
# Usage from jumphost:
#   bash contrib/nccl_ep/sweep/jumphost_ep16_nocopy_compare.sh
# Or via curl:
#   curl -fsSL https://raw.githubusercontent.com/zhangfei829/nccl/master/\
#     contrib/nccl_ep/sweep/jumphost_ep16_nocopy_compare.sh \
#     | TRAYS="..." TOKENS="4096 8192" bash
# =============================================================================
set -euo pipefail

SSH_KEY="${SSH_KEY:-id_ed25519}"
USER_NAME="${USER_NAME:-fizhang}"
TRAYS="${TRAYS:-pod4-gb300-2-tray01-f3 pod4-gb300-2-tray02-f3 pod4-gb300-2-tray03-f3 pod4-gb300-2-tray04-f3}"
HEAD_TRAY="${HEAD_TRAY:-$(echo "$TRAYS" | awk '{print $1}')}"
EP_SIZES="${EP_SIZES:-16}"
TOKENS="${TOKENS:-4096 8192}"
EXTRA_BUILD_FLAGS="${EXTRA_BUILD_FLAGS:-}"
TS="$(date +%Y%m%d_%H%M%S)"
LOCAL_OUT="${LOCAL_OUT:-$HOME/nccl_ep_runs/nocopy_compare_${TS}}"
NCCL_GIT_URL="${NCCL_GIT_URL:-https://github.com/zhangfei829/nccl.git}"

SSH_BASE=(ssh -i "$SSH_KEY" -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new)
SSH_BASE_N=(ssh -n -i "$SSH_KEY" -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new)

mkdir -p "$LOCAL_OUT"

cat <<EOF
===========================================================
Jumphost dispatch+combine NO-COPY 3-way comparison
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
echo "===== [3/6] Build NCCL EP (TMA overlap macro on so dispatch_overlap can also be tested) ====="
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

REMOTE_BASE="/home/fizhang/nccl-sweeps/nocopy-${TS}"

echo
echo "===== [4/6] Run baseline + dispatch_overlap + no_copy for EP=$EP_SIZES tokens=$TOKENS ====="
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
echo "[hostfile]"
sed 's/^/  /' "\$HOSTFILE"

for ep in $EP_SIZES; do
  for case in baseline dispatch_overlap no_copy; do
    unset NCCL_EP_HT_DISPATCH_TMA_COPY NCCL_EP_HT_NO_COPY || true
    case "\$case" in
      baseline)         : ;;
      dispatch_overlap) export NCCL_EP_HT_DISPATCH_TMA_COPY=1 ;;
      no_copy)          export NCCL_EP_HT_NO_COPY=1 ;;
    esac
    OUT=$REMOTE_BASE/ep\${ep}/\${case}
    mkdir -p "\$OUT"
    # Validation: enable for all 3 cases. baseline confirms validator infra
    # works, dispatch_overlap exercises in-kernel TMA write to recv_x,
    # no_copy exercises the bench-side STAGE<->user bridging copies that
    # ep_bench performs only in the validation path when NCCL_EP_HT_NO_COPY=1.
    BENCH_EXTRA="--validate"
    echo
    echo "===== EP=\$ep case=\$case (DISPATCH_TMA=\${NCCL_EP_HT_DISPATCH_TMA_COPY:-unset} NO_COPY=\${NCCL_EP_HT_NO_COPY:-unset}) ====="
    EP_SIZE=\$ep \\
    TOKENS="$TOKENS" \\
    MODES="ht_bf16" \\
    LOG_DIR="\$OUT" \\
    CSV_FILE="\$OUT/results.csv" \\
    HOSTFILE_OVERRIDE="\$HOSTFILE" \\
    EXTRA_BENCH_ARGS="\$BENCH_EXTRA" \\
    bash contrib/nccl_ep/sweep/ep_sweep.sh
  done
done
REMOTE

echo
echo "===== [5/6] Collect remote outputs ====="
mkdir -p "$LOCAL_OUT"
for ep in $EP_SIZES; do
  for case in baseline dispatch_overlap no_copy; do
    mkdir -p "$LOCAL_OUT/ep${ep}/${case}"
    "${SSH_BASE_N[@]}" "$USER_NAME@$HEAD_TRAY" "cat '$REMOTE_BASE/ep${ep}/${case}/results.csv'" \
      > "$LOCAL_OUT/ep${ep}/${case}/results.csv" 2>/dev/null || true
    # Pull all token-size logs for this case so we can grep validation lines.
    "${SSH_BASE_N[@]}" "$USER_NAME@$HEAD_TRAY" \
      "tar -C '$REMOTE_BASE/ep${ep}/${case}' -cf - . 2>/dev/null" \
      | tar -C "$LOCAL_OUT/ep${ep}/${case}" -xf - 2>/dev/null || true
  done
done

echo
echo "===== [5b/6] Validation summary (grep --validate output) ====="
for ep in $EP_SIZES; do
  for case in baseline dispatch_overlap no_copy; do
    LOG_DIR_LOCAL="$LOCAL_OUT/ep${ep}/${case}"
    [ -d "$LOG_DIR_LOCAL" ] || continue
    for log in "$LOG_DIR_LOCAL"/*.log; do
      [ -f "$log" ] || continue
      tok=$(basename "$log" .log)
      d_line=$(grep -E '^Dispatch validation' "$log" | head -1 || true)
      c_line=$(grep -E '^Combine validation'  "$log" | head -1 || true)
      g_line=$(grep -E '^Global validation'   "$log" | head -1 || true)
      printf "[VAL] EP=%s case=%-16s %-40s\n" "$ep" "$case" "$tok"
      printf "      %s\n      %s\n      %s\n" "${d_line:-<no Dispatch line>}" "${c_line:-<no Combine line>}" "${g_line:-<no Global line>}"
    done
  done
done | tee "$LOCAL_OUT/validation_summary.txt"

echo
echo "===== [6/6] Comparison Tables ====="
python3 - "$LOCAL_OUT" <<'PY' | tee "$LOCAL_OUT/summary.txt"
import csv, os, sys

root = sys.argv[1]
# data[(ep, tokens)][case] = {'d_kernel_us', 'd_avg_us', 'd_send_mb',
#                             'c_kernel_us', 'c_avg_us', 'c_send_mb'}
data = {}

CASES = ('baseline', 'dispatch_overlap', 'no_copy')

for ep_dir in sorted(os.listdir(root)):
    if not ep_dir.startswith('ep'):
        continue
    ep = int(ep_dir[2:])
    for case in CASES:
        csvp = os.path.join(root, ep_dir, case, 'results.csv')
        if not os.path.exists(csvp):
            continue
        with open(csvp) as f:
            for r in csv.DictReader(f):
                t = int(r['tokens'])
                data.setdefault((ep, t), {})[case] = {
                    'd_kernel_us': float(r['dispatch_kernel_us']),
                    'd_avg_us':    float(r['dispatch_avg_us']),
                    'd_send_mb':   float(r['total_send_mb']),
                    'c_kernel_us': float(r['combine_kernel_us']),
                    'c_avg_us':    float(r['combine_avg_us']),
                    'c_send_mb':   float(r['total_send_mb']),  # combine reuses send_mb (symmetric)
                }

def bw(send_mb, time_us):
    return send_mb / time_us * 1000.0 if time_us > 0 else 0.0

def fmt_row(label, ep, t, base, opt, key, send_key='d_send_mb'):
    if base is None or opt is None:
        return None
    bv = bw(base[send_key], base[key])
    ov = bw(opt[send_key],  opt[key])
    r  = ov / bv if bv > 0 else 0.0
    return f"{label:<18} {ep:>4} {t:>7} | {bv:>10.1f} | {ov:>10.1f} | {r:>7.3f}x"

# === DISPATCH ===
print()
print("=== Dispatch Kernel BW (GPU-only, GB/s) ===")
print(f"{'comparison':<18} {'EP':>4} {'tokens':>7} | {'base':>10} | {'optim':>10} | {'ratio':>8}")
print("-" * 80)
for (ep, t) in sorted(data.keys()):
    d = data[(ep, t)]
    b = d.get('baseline')
    for opt_case in ('dispatch_overlap', 'no_copy'):
        line = fmt_row(opt_case, ep, t, b, d.get(opt_case), 'd_kernel_us', 'd_send_mb')
        if line: print(line)

print()
print("=== Dispatch API BW (wall-clock incl host, GB/s) ===")
print(f"{'comparison':<18} {'EP':>4} {'tokens':>7} | {'base':>10} | {'optim':>10} | {'ratio':>8}")
print("-" * 80)
for (ep, t) in sorted(data.keys()):
    d = data[(ep, t)]
    b = d.get('baseline')
    for opt_case in ('dispatch_overlap', 'no_copy'):
        line = fmt_row(opt_case, ep, t, b, d.get(opt_case), 'd_avg_us', 'd_send_mb')
        if line: print(line)

# === COMBINE ===
print()
print("=== Combine Kernel BW (GPU-only, GB/s) ===")
print(f"{'comparison':<18} {'EP':>4} {'tokens':>7} | {'base':>10} | {'optim':>10} | {'ratio':>8}")
print("-" * 80)
for (ep, t) in sorted(data.keys()):
    d = data[(ep, t)]
    b = d.get('baseline')
    for opt_case in ('dispatch_overlap', 'no_copy'):
        line = fmt_row(opt_case, ep, t, b, d.get(opt_case), 'c_kernel_us', 'c_send_mb')
        if line: print(line)

print()
print("=== Combine API BW (wall-clock incl host, GB/s) ===")
print(f"{'comparison':<18} {'EP':>4} {'tokens':>7} | {'base':>10} | {'optim':>10} | {'ratio':>8}")
print("-" * 80)
for (ep, t) in sorted(data.keys()):
    d = data[(ep, t)]
    b = d.get('baseline')
    for opt_case in ('dispatch_overlap', 'no_copy'):
        line = fmt_row(opt_case, ep, t, b, d.get(opt_case), 'c_avg_us', 'c_send_mb')
        if line: print(line)

# === Combined view: 3 cases side-by-side, dispatch + combine ===
print()
print("=== Combined: 3 cases side-by-side ===")
print(f"{'EP':>4} {'tokens':>7} | {'phase':<8} | "
      f"{'baseline kernel/API':>22} | "
      f"{'overlap kernel/API':>22} | "
      f"{'no_copy kernel/API':>22}")
print("-" * 110)
for (ep, t) in sorted(data.keys()):
    d = data[(ep, t)]
    for phase, kk, ka in (('dispatch','d_kernel_us','d_avg_us'),
                          ('combine', 'c_kernel_us','c_avg_us')):
        cells = []
        for case in CASES:
            r = d.get(case)
            if r is None:
                cells.append(f"{'-':>22}")
                continue
            kbw = bw(r['d_send_mb'], r[kk])
            abw = bw(r['d_send_mb'], r[ka])
            cells.append(f"{kbw:>10.1f}/{abw:>10.1f}")
        print(f"{ep:>4} {t:>7} | {phase:<8} | {cells[0]:>22} | {cells[1]:>22} | {cells[2]:>22}")
PY

echo
echo "==========================================================="
echo "Done. Local collected files:"
find "$LOCAL_OUT" -maxdepth 3 -type f -print | sort
echo "==========================================================="
