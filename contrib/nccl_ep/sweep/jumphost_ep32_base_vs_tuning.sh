#!/usr/bin/env bash
# =============================================================================
# jumphost_ep32_base_vs_tuning.sh
#
# Two-case comparison for EP32 BF16:
#   baseline: no NV72 env (group->ht_nv72_env_override = false)
#   tunning : NCCL_EP_HT_NV72_NUM_SMS=$NV72_NUM_SMS_TUNING +
#             NCCL_EP_HT_NV72_CHUNK=$NV72_CHUNK_TUNING (env override)
#
# IMPORTANT: per hybridep_adapter.cu:611-755 (dispatch_impl) and 873-997
# (combine_impl), the multi-node (NUM_LSA_TEAMS != 1) branches hard-code
# HYBRIDEP_DISPATCH_NUM_OF_BLOCKS / HYBRIDEP_COMBINE_NUM_OF_BLOCKS = 16
# regardless of the num_blocks runtime argument. So on EP32 (NUM_LSA_TEAMS = 8)
# the env vars above do NOT change the actual kernel grid size; both cases
# launch with 16 SM. This sweep empirically verifies that fact.
#
# Defaults: EP=32, BF16, tokens 16/256/512/4096/8192, 8 trays.
#
# Usage from jumphost:
#   bash contrib/nccl_ep/sweep/jumphost_ep32_base_vs_tuning.sh
# Or via curl:
#   curl -fsSL https://raw.githubusercontent.com/zhangfei829/nccl/master/\
#     contrib/nccl_ep/sweep/jumphost_ep32_base_vs_tuning.sh \
#     | TRAYS="..." TOKENS="..." bash
# =============================================================================
set -euo pipefail

SSH_KEY="${SSH_KEY:-id_ed25519}"
USER_NAME="${USER_NAME:-fizhang}"
TRAYS="${TRAYS:-pod4-gb300-2-tray05-f3 pod4-gb300-2-tray06-f3 pod4-gb300-2-tray07-f3 pod4-gb300-2-tray08-f3 pod4-gb300-2-tray01-f3 pod4-gb300-2-tray02-f3 pod4-gb300-2-tray03-f3 pod4-gb300-2-tray04-f3}"
HEAD_TRAY="${HEAD_TRAY:-$(echo "$TRAYS" | awk '{print $1}')}"
EP_SIZES="${EP_SIZES:-32}"
TOKENS="${TOKENS:-16 256 512 4096 8192}"
EXTRA_BUILD_FLAGS="${EXTRA_BUILD_FLAGS:-}"
NV72_NUM_SMS_TUNING="${NV72_NUM_SMS_TUNING:-32}"
NV72_CHUNK_TUNING="${NV72_CHUNK_TUNING:-128}"
TS="$(date +%Y%m%d_%H%M%S)"
LOCAL_OUT="${LOCAL_OUT:-$HOME/nccl_ep_runs/ep32_base_vs_tuning_${TS}}"
NCCL_GIT_URL="${NCCL_GIT_URL:-https://github.com/zhangfei829/nccl.git}"

SSH_BASE=(ssh -i "$SSH_KEY" -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new)
SSH_BASE_N=(ssh -n -i "$SSH_KEY" -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new)

mkdir -p "$LOCAL_OUT"

cat <<EOF
===========================================================
Jumphost EP32 baseline vs tunning sweep
  SSH_KEY              : $SSH_KEY
  USER_NAME            : $USER_NAME
  HEAD_TRAY            : $HEAD_TRAY
  TRAYS                : $TRAYS
  EP_SIZES             : $EP_SIZES
  TOKENS               : $TOKENS
  NV72_NUM_SMS_TUNING  : $NV72_NUM_SMS_TUNING
  NV72_CHUNK_TUNING    : $NV72_CHUNK_TUNING
  EXTRA_BUILD_FLAGS    : ${EXTRA_BUILD_FLAGS:-<none>}
  LOCAL_OUT            : $LOCAL_OUT
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
echo "===== [2/6] Install head pubkey on all 8 trays and verify ====="
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
echo "===== [3/6] Build NCCL EP (no extra macros, just default + EXTRA_BUILD_FLAGS) ====="
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
  EXTRA_CXXFLAGS="$EXTRA_BUILD_FLAGS" \\
  EXTRA_NVCCFLAGS="$EXTRA_BUILD_FLAGS" \\
  MPI_HOME=\$MPI_HOME

ls -l /home/fizhang/nccl/build/lib/libnccl_ep.so /home/fizhang/nccl/build/test/nccl_ep/ep_bench
REMOTE

REMOTE_BASE="/home/fizhang/nccl-sweeps/ep32_base_vs_tuning-${TS}"

echo
echo "===== [4/6] Run baseline + tunning for EP=$EP_SIZES tokens=$TOKENS ====="
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
  for case in baseline tunning; do
    unset NCCL_EP_HT_NV72_NUM_SMS NCCL_EP_HT_NV72_CHUNK || true
    case "\$case" in
      baseline) : ;;
      tunning)
        export NCCL_EP_HT_NV72_NUM_SMS=$NV72_NUM_SMS_TUNING
        export NCCL_EP_HT_NV72_CHUNK=$NV72_CHUNK_TUNING
        ;;
    esac
    OUT=$REMOTE_BASE/ep\${ep}/\${case}
    mkdir -p "\$OUT"
    echo
    echo "===== EP=\$ep case=\$case (NV72_NUM_SMS=\${NCCL_EP_HT_NV72_NUM_SMS:-unset} NV72_CHUNK=\${NCCL_EP_HT_NV72_CHUNK:-unset}) ====="
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
  for case in baseline tunning; do
    mkdir -p "$LOCAL_OUT/ep${ep}/${case}"
    "${SSH_BASE_N[@]}" "$USER_NAME@$HEAD_TRAY" "cat '$REMOTE_BASE/ep${ep}/${case}/results.csv'" \
      > "$LOCAL_OUT/ep${ep}/${case}/results.csv" 2>/dev/null || true
    "${SSH_BASE_N[@]}" "$USER_NAME@$HEAD_TRAY" \
      "tar -C '$REMOTE_BASE/ep${ep}/${case}' -cf - . 2>/dev/null" \
      | tar -C "$LOCAL_OUT/ep${ep}/${case}" -xf - 2>/dev/null || true
  done
done

echo
echo "===== [5b/6] NV72 env header check (verify env actually reached the run) ====="
for ep in $EP_SIZES; do
  for case in baseline tunning; do
    LOG_DIR_LOCAL="$LOCAL_OUT/ep${ep}/${case}"
    [ -d "$LOG_DIR_LOCAL" ] || continue
    for log in "$LOG_DIR_LOCAL"/*.log; do
      [ -f "$log" ] || continue
      tok=$(basename "$log" .log)
      # NV72-ADAPT header line (printed by nccl_ep.cc:2004-2013 on rank 0)
      nv_line=$(grep -E '\[NV72-ADAPT\] HT NV72 tuning' "$log" | head -1 || true)
      printf "[NV72] EP=%s case=%-9s %-40s\n" "$ep" "$case" "$tok"
      printf "       %s\n" "${nv_line:-<no NV72-ADAPT line>}"
    done
  done
done | tee "$LOCAL_OUT/nv72_env_check.txt"

echo
echo "===== [6/6] Comparison Tables ====="
python3 - "$LOCAL_OUT" <<'PY' | tee "$LOCAL_OUT/summary.txt"
import csv, os, sys

root = sys.argv[1]
# data[(ep, tokens)][case] = {'d_kernel_us', 'd_avg_us', 'd_send_mb',
#                             'c_kernel_us', 'c_avg_us'}
data = {}

CASES = ('baseline', 'tunning')

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
                }

def bw(send_mb, time_us):
    return send_mb / time_us * 1000.0 if time_us > 0 else 0.0

def fmt_row(ep, t, base, opt, key):
    if base is None or opt is None:
        return None
    bv = bw(base['d_send_mb'], base[key])
    ov = bw(opt['d_send_mb'],  opt[key])
    r  = ov / bv if bv > 0 else 0.0
    return f"{ep:>4} {t:>7} | {bv:>10.1f} | {ov:>10.1f} | {r:>7.3f}x"

# === DISPATCH ===
print()
print("=== Dispatch Kernel BW (GPU-only, GB/s) ===")
print(f"{'EP':>4} {'tokens':>7} | {'baseline':>10} | {'tunning':>10} | {'ratio':>8}")
print("-" * 60)
for (ep, t) in sorted(data.keys()):
    d = data[(ep, t)]
    line = fmt_row(ep, t, d.get('baseline'), d.get('tunning'), 'd_kernel_us')
    if line: print(line)

print()
print("=== Dispatch API BW (wall-clock incl host, GB/s) ===")
print(f"{'EP':>4} {'tokens':>7} | {'baseline':>10} | {'tunning':>10} | {'ratio':>8}")
print("-" * 60)
for (ep, t) in sorted(data.keys()):
    d = data[(ep, t)]
    line = fmt_row(ep, t, d.get('baseline'), d.get('tunning'), 'd_avg_us')
    if line: print(line)

# === COMBINE ===
print()
print("=== Combine Kernel BW (GPU-only, GB/s) ===")
print(f"{'EP':>4} {'tokens':>7} | {'baseline':>10} | {'tunning':>10} | {'ratio':>8}")
print("-" * 60)
for (ep, t) in sorted(data.keys()):
    d = data[(ep, t)]
    line = fmt_row(ep, t, d.get('baseline'), d.get('tunning'), 'c_kernel_us')
    if line: print(line)

print()
print("=== Combine API BW (wall-clock incl host, GB/s) ===")
print(f"{'EP':>4} {'tokens':>7} | {'baseline':>10} | {'tunning':>10} | {'ratio':>8}")
print("-" * 60)
for (ep, t) in sorted(data.keys()):
    d = data[(ep, t)]
    line = fmt_row(ep, t, d.get('baseline'), d.get('tunning'), 'c_avg_us')
    if line: print(line)
PY

echo
echo "==========================================================="
echo "Done. Local collected files:"
find "$LOCAL_OUT" -maxdepth 3 -type f -print | sort
echo "==========================================================="
