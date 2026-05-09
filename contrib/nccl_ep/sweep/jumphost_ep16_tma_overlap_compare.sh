#!/usr/bin/env bash
# =============================================================================
# jumphost_ep16_tma_overlap_compare.sh
#
# One-command comparison for the gated HT dispatch TMA-copy overlap prototype.
#
# It runs from the jumphost and does all of this:
#   1. setup head-tray -> all-trays passwordless ssh
#   2. clone/pull repo on /home/fizhang/nccl
#   3. ensure main NCCL is built
#   4. rebuild NCCL EP with NCCL_EP_ENABLE_HT_TMA_COPY_OVERLAP
#   5. run baseline (env unset) and overlap candidate
#   6. collect CSV/log/HT-PROFILE markers
#   7. print a comparison table
#
# Default:
#   TRAYS=pod4-gb300-2-tray05-f3..08-f3
#   TOKENS="4096 8192"
#   BASELINE_MODES="ht_bf16"
# =============================================================================
set -euo pipefail

SSH_KEY="${SSH_KEY:-id_ed25519}"
USER_NAME="${USER_NAME:-fizhang}"
TRAYS="${TRAYS:-pod4-gb300-2-tray05-f3 pod4-gb300-2-tray06-f3 pod4-gb300-2-tray07-f3 pod4-gb300-2-tray08-f3}"
HEAD_TRAY="${HEAD_TRAY:-$(echo "$TRAYS" | awk '{print $1}')}"
TOKENS="${TOKENS:-4096 8192}"
BASELINE_MODES="${BASELINE_MODES:-ht_bf16}"
OVERLAP_CASES="${OVERLAP_CASES:-32:128 64:128}"
TS="$(date +%Y%m%d_%H%M%S)"
LOCAL_OUT="${LOCAL_OUT:-$HOME/nccl_ep_runs/ep16_tma_overlap_${TS}}"
NCCL_GIT_URL="${NCCL_GIT_URL:-https://github.com/zhangfei829/nccl.git}"

SSH_BASE=(ssh -i "$SSH_KEY" -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new)
# Variant for non-stdin ssh (cat / grep collection in step 5).  Critical:
# under `curl ... | bash` invocation the parent bash reads the script body
# from stdin; an interactive ssh without -n would consume that stdin and
# eat the rest of the script (esp. step 6 python heredoc).  Steps using
# heredoc (bash -l <<REMOTE) MUST stay on the no-`-n` form.
SSH_BASE_N=(ssh -n -i "$SSH_KEY" -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new)

mkdir -p "$LOCAL_OUT"

cat <<EOF
===========================================================
Jumphost EP16 TMA-overlap comparison
  SSH_KEY       : $SSH_KEY
  USER_NAME     : $USER_NAME
  HEAD_TRAY     : $HEAD_TRAY
  TRAYS         : $TRAYS
  TOKENS        : $TOKENS
  BASELINE_MODES: $BASELINE_MODES
  OVERLAP_CASES : $OVERLAP_CASES
  LOCAL_OUT     : $LOCAL_OUT
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
echo "===== [3/6] Build NCCL EP with overlap prototype macro ====="
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
  EXTRA_CXXFLAGS="-DNCCL_EP_ENABLE_HT_TMA_COPY_OVERLAP -DNCCL_EP_ENABLE_HT_COMBINE_INPUT_COPY" \\
  EXTRA_NVCCFLAGS="-DNCCL_EP_ENABLE_HT_TMA_COPY_OVERLAP -DNCCL_EP_ENABLE_HT_COMBINE_INPUT_COPY" \\
  MPI_HOME=\$MPI_HOME
REMOTE

REMOTE_BASE="/home/fizhang/nccl-sweeps/tma-overlap-${TS}-ep16"

run_case() {
  local tag="$1"
  local enable_copy="$2"
  echo
  echo "===== [4/6] Run case: $tag ====="
  "${SSH_BASE[@]}" "$USER_NAME@$HEAD_TRAY" bash -l <<REMOTE
set -euo pipefail
export MPI_HOME=/usr/mpi/gcc/openmpi-4.1.9a1
export PATH=\$MPI_HOME/bin:/usr/local/cuda/bin:\$PATH
export LD_LIBRARY_PATH=\$MPI_HOME/lib:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:\${LD_LIBRARY_PATH:-}
export CUDA_HOME=/usr/local/cuda
export NCCL_EP_HT_PROFILE=1
if [[ "$enable_copy" == "1" ]]; then
  export NCCL_EP_HT_DISPATCH_TMA_COPY=1
  export NCCL_EP_HT_COMBINE_INPUT_COPY=1
else
  unset NCCL_EP_HT_DISPATCH_TMA_COPY || true
  unset NCCL_EP_HT_COMBINE_INPUT_COPY || true
fi
cd /home/fizhang/nccl
OUT="$REMOTE_BASE/$tag" \\
TRAYS="$TRAYS" \\
TOKENS="$TOKENS" \\
NV72_PAIR_LIST="$OVERLAP_CASES" \\
BASELINE_MODES="$BASELINE_MODES" \\
SKIP_NV72_CALIBRATE=1 \\
NCCL_EP_HT_PROFILE=1 \\
bash contrib/nccl_ep/sweep/run_ep16_4bay.sh 2>&1 | tee "/home/fizhang/run_ep16_${tag}_${TS}.log"
REMOTE
}

run_case baseline 0
run_case tma_overlap 1

echo
echo "===== [5/6] Collect remote outputs ====="
for tag in baseline tma_overlap; do
  mkdir -p "$LOCAL_OUT/$tag"
  # Use SSH_BASE_N (with -n) here: under `curl|bash` the parent bash reads
  # the rest of this script from stdin; an interactive ssh would eat that
  # stdin and silently drop the python heredoc in step 6 below.
  "${SSH_BASE_N[@]}" "$USER_NAME@$HEAD_TRAY" "cat '$REMOTE_BASE/$tag/baseline/results.csv'" > "$LOCAL_OUT/$tag/results.csv" || true
  "${SSH_BASE_N[@]}" "$USER_NAME@$HEAD_TRAY" "cat '/home/fizhang/run_ep16_${tag}_${TS}.log'" > "$LOCAL_OUT/$tag/run.log" || true
  "${SSH_BASE_N[@]}" "$USER_NAME@$HEAD_TRAY" "grep -h '\\[HT-PROFILE\\]' '$REMOTE_BASE/$tag'/baseline/*.log 2>/dev/null || true" > "$LOCAL_OUT/$tag/ht_profile.txt" || true
done

echo
echo "===== [6/6] Comparison ====="
python3 - "$LOCAL_OUT" <<'PY' | tee "$LOCAL_OUT/summary.txt"
import csv, os, re, sys

root = sys.argv[1]
data = {}
profiles = {}
for tag in ("baseline", "tma_overlap"):
    csvp = os.path.join(root, tag, "results.csv")
    if os.path.exists(csvp):
        with open(csvp) as f:
            for r in csv.DictReader(f):
                data[(tag, int(r["tokens"]))] = (
                    float(r["dispatch_avg_us"]),
                    float(r["dispatch_kernel_us"]),
                    float(r["combine_avg_us"]),
                    float(r["combine_kernel_us"]),
                )
    profp = os.path.join(root, tag, "ht_profile.txt")
    vals = []
    if os.path.exists(profp):
        with open(profp, errors="replace") as f:
            for line in f:
                if "[HT-PROFILE] dispatch" in line:
                    m = re.search(r"t=(\d+).*output_copy=([\d.]+).*total_stream=([\d.]+)", line)
                    if m:
                        vals.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
    profiles[tag] = vals

for tag, vals in profiles.items():
    print(f"===== {tag} HT-PROFILE tail =====")
    for t, copy, total in vals[-8:]:
        print(f"PROFILE {tag:>11} t={t} output_copy={copy:.1f} total_stream={total:.1f}")
    if not vals:
        print(f"NO HT-PROFILE lines for {tag}")

print()
print("tokens | baseline dispatch_avg/kernel | overlap dispatch_avg/kernel | dispatch_avg speedup")
for t in sorted({k[1] for k in data}):
    if ("baseline", t) in data and ("tma_overlap", t) in data:
        b = data[("baseline", t)]
        x = data[("tma_overlap", t)]
        print(f"{t:>6} | {b[0]:>8.1f}/{b[1]:>8.1f} | {x[0]:>8.1f}/{x[1]:>8.1f} | {b[0]/x[0]:>6.2f}x")

print()
print("tokens | baseline combine_avg/kernel  | overlap combine_avg/kernel  | combine_avg speedup")
for t in sorted({k[1] for k in data}):
    if ("baseline", t) in data and ("tma_overlap", t) in data:
        b = data[("baseline", t)]
        x = data[("tma_overlap", t)]
        print(f"{t:>6} | {b[2]:>8.1f}/{b[3]:>8.1f} | {x[2]:>8.1f}/{x[3]:>8.1f} | {b[2]/x[2]:>6.2f}x")

print()
print("tokens | baseline (D+C avg) | overlap (D+C avg) | total speedup")
for t in sorted({k[1] for k in data}):
    if ("baseline", t) in data and ("tma_overlap", t) in data:
        b = data[("baseline", t)]
        x = data[("tma_overlap", t)]
        b_dc = b[0] + b[2]
        x_dc = x[0] + x[2]
        print(f"{t:>6} | {b_dc:>14.1f}us  | {x_dc:>14.1f}us  | {b_dc/x_dc:>6.2f}x")
PY

echo
echo "==========================================================="
echo "Done. Local collected files:"
find "$LOCAL_OUT" -maxdepth 2 -type f -print | sort
echo "==========================================================="
