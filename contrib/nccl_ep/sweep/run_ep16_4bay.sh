#!/usr/bin/env bash
# =============================================================================
# run_ep16_4bay.sh
#
# One-shot driver for "I have 4 manually-reserved GB300 BAYs, no Slurm".
# Runs from the head node (e.g. pod4-gb300-2-tray01-f3) after you ssh into
# it from the jumphost. It will:
#
#   1. cd to NCCL_REPO (default $HOME/fizhang/nccl), git fetch + hard reset
#      to origin/master so we run exactly what was pushed.
#   2. Write /tmp/hosts.ep16 with the 4 default trays
#      (override TRAYS env var to use a different set).
#   3. Autobuild: if any contrib/nccl_ep/*.{cu,cuh,cc,h} is newer than the
#      ep_bench binary (or binary missing), run `make -j3`.
#   4. Run baseline sweep: EP=16, MODES="ht_bf16 ht_fp8 fullmesh_bf16",
#      TOKENS="16 32 64 128 256 4096 8192", into $OUT/baseline.
#   5. Run HT NV72 9-cell calibrate via ht_nv72_calibrate.sh, into $OUT.
#   6. Merge baseline results into ~/fizhang/nccl_ep_master.csv.
#   7. Print HT vs FULLMESH summary with ep_summary.py.
#
# Usage from jumphost:
#   ssh -i ~/Desktop/my/id_ed25519 fizhang@pod4-gb300-2-tray01-f3 \
#       bash -lc 'cd ~/fizhang/nccl && \
#                 git fetch origin master && git reset --hard origin/master && \
#                 bash contrib/nccl_ep/sweep/run_ep16_4bay.sh'
#
# Env overrides:
#   NCCL_REPO   default $HOME/fizhang/nccl
#   NCCL_HOME   default $NCCL_REPO/build
#   CUDA_HOME   default /usr/local/cuda
#   NVCC_GENCODE default "-gencode=arch=compute_103,code=sm_103"
#   TRAYS       default "pod4-gb300-2-tray01-f3 pod4-gb300-2-tray02-f3 \
#                        pod4-gb300-2-tray03-f3 pod4-gb300-2-tray04-f3"
#   OUT         default $HOME/fizhang/nccl-sweep-<ts>-ep16-4bay
#   SKIP_BASELINE       set to 1 to skip the baseline sweep
#   SKIP_NV72_CALIBRATE set to 1 to skip the 9-cell HT NV72 sweep
#   EXTRA_BENCH_ARGS    forwarded to ep_bench (e.g. "--validate")
#
# Wall time on a healthy 4-BAY GB300:
#   - baseline (3 modes x 7 tokens): ~30-40 min
#   - NV72 9 cells x 7 tokens:        ~60-90 min
#   Total ~2h. Plan your reservation accordingly.
# =============================================================================
set -u
# DO NOT set -e: per-cell failures should not kill the whole run.

NCCL_REPO="${NCCL_REPO:-$HOME/fizhang/nccl}"
NCCL_HOME="${NCCL_HOME:-$NCCL_REPO/build}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
NVCC_GENCODE="${NVCC_GENCODE:--gencode=arch=compute_103,code=sm_103}"
TRAYS="${TRAYS:-pod4-gb300-2-tray01-f3 pod4-gb300-2-tray02-f3 pod4-gb300-2-tray03-f3 pod4-gb300-2-tray04-f3}"
TS="$(date +%Y%m%d_%H%M%S)"
OUT="${OUT:-$HOME/fizhang/nccl-sweep-${TS}-ep16-4bay}"
SKIP_BASELINE="${SKIP_BASELINE:-0}"
SKIP_NV72_CALIBRATE="${SKIP_NV72_CALIBRATE:-0}"
EXTRA_BENCH_ARGS="${EXTRA_BENCH_ARGS:-}"

mkdir -p "$OUT"
HOSTFILE="$OUT/hosts.ep16"
{
    for tray in $TRAYS; do
        echo "$tray slots=4"
    done
} > "$HOSTFILE"

# Sanity: 4 trays * 4 GPUs = 16 ranks for EP=16
NTRAYS=$(echo $TRAYS | wc -w)
EP_SIZE=$(( NTRAYS * 4 ))
if [[ $EP_SIZE -ne 16 ]]; then
    echo "[run_ep16_4bay] WARN: TRAYS produces EP_SIZE=$EP_SIZE != 16; downstream wrappers will use $EP_SIZE."
fi

cat <<EOF
===========================================================
EP16 4-BAY runbook
  NCCL_REPO   : $NCCL_REPO
  NCCL_HOME   : $NCCL_HOME
  CUDA_HOME   : $CUDA_HOME
  NVCC_GENCODE: $NVCC_GENCODE
  TRAYS       : $TRAYS
  EP_SIZE     : $EP_SIZE
  HOSTFILE    : $HOSTFILE
  OUT         : $OUT
  SKIP_BASELINE       : $SKIP_BASELINE
  SKIP_NV72_CALIBRATE : $SKIP_NV72_CALIBRATE
===========================================================
EOF
sed 's/^/  /' "$HOSTFILE"
echo

# Sanity: every tray must be ssh-reachable without password from this head
# node, otherwise mpirun --mca plm slurm or rsh/ssh launchers will hang.
echo "[run_ep16_4bay] Verifying ssh reachability of each tray..."
for tray in $TRAYS; do
    if ! ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new "$tray" "echo ok-from-\$(hostname)" 2>/dev/null | grep -q ok-from-; then
        echo "[run_ep16_4bay] ERROR: cannot ssh $tray (no passwordless ssh?). Aborting." >&2
        exit 2
    fi
done
echo "[run_ep16_4bay] All trays reachable."
echo

cd "$NCCL_REPO" || { echo "[run_ep16_4bay] ERROR: NCCL_REPO=$NCCL_REPO missing"; exit 2; }

export NCCL_HOME CUDA_HOME NVCC_GENCODE
which mpirun >/dev/null || { echo "[run_ep16_4bay] ERROR: mpirun not in PATH"; exit 2; }
export MPI_HOME="$(dirname $(dirname $(readlink -f $(which mpirun))))"
[[ -f "$MPI_HOME/include/mpi.h" ]] || { echo "[run_ep16_4bay] ERROR: mpi.h not at $MPI_HOME/include/mpi.h"; exit 2; }
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64:$NCCL_HOME/lib:${LD_LIBRARY_PATH:-}"

# Autobuild
BIN="$NCCL_HOME/test/nccl_ep/ep_bench"
LIB="$NCCL_HOME/lib/libnccl_ep.so"
NEED_BUILD=0
if [[ ! -x "$BIN" || ! -f "$LIB" ]]; then
    echo "[autobuild] $BIN or $LIB missing, will build"
    NEED_BUILD=1
else
    NEWER=$(find contrib/nccl_ep \
        \( -name '*.cu' -o -name '*.cuh' -o -name '*.cc' \
           -o -name '*.h' -o -name '*.hpp' -o -name 'Makefile' \) \
        -newer "$BIN" 2>/dev/null | head -5)
    if [[ -n "$NEWER" ]]; then
        echo "[autobuild] sources newer than $BIN:"
        echo "$NEWER" | sed 's/^/  /'
        NEED_BUILD=1
    else
        echo "[autobuild] $BIN up to date"
    fi
fi
if [[ $NEED_BUILD -eq 1 ]]; then
    time make -j3 -C contrib/nccl_ep MPI=1 BUILDDIR="$NCCL_HOME" \
              NVCC_GENCODE="$NVCC_GENCODE" MPI_HOME="$MPI_HOME"
    rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "[autobuild] FAILED rc=$rc" >&2
        exit 3
    fi
    ls -l "$BIN" "$LIB"
fi

# Baseline sweep
if [[ "$SKIP_BASELINE" != "1" ]]; then
    echo
    echo "===== Baseline sweep (HT bf16/fp8 + FULLMESH bf16) ====="
    HOSTFILE_OVERRIDE="$HOSTFILE" \
        EP_SIZE="$EP_SIZE" \
        TOKENS="16 32 64 128 256 4096 8192" \
        MODES="ht_bf16 ht_fp8 fullmesh_bf16" \
        LOG_DIR="$OUT/baseline" \
        CSV_FILE="$OUT/baseline/results.csv" \
        EXTRA_BENCH_ARGS="$EXTRA_BENCH_ARGS" \
        bash contrib/nccl_ep/sweep/ep_sweep.sh
else
    echo "[run_ep16_4bay] SKIP_BASELINE=1, skipping baseline sweep"
fi

# NV72 9-cell calibrate (only ht_bf16 -- combine doesn't support FP8)
if [[ "$SKIP_NV72_CALIBRATE" != "1" ]]; then
    echo
    echo "===== HT NV72 9-cell calibrate (NUM_SMS x CHUNK) ====="
    HOSTFILE_OVERRIDE="$HOSTFILE" \
        EP_SIZE="$EP_SIZE" \
        OUT="$OUT" \
        EXTRA_BENCH_ARGS="$EXTRA_BENCH_ARGS" \
        bash contrib/nccl_ep/sweep/ht_nv72_calibrate.sh
else
    echo "[run_ep16_4bay] SKIP_NV72_CALIBRATE=1, skipping NV72 calibrate"
fi

# Merge baseline into long-lived master csv
MASTER_CSV="${MASTER_CSV:-$HOME/fizhang/nccl_ep_master.csv}"
if [[ "$SKIP_BASELINE" != "1" && -f "$OUT/baseline/results.csv" ]]; then
    if python3 contrib/nccl_ep/sweep/merge_into_master.py "$MASTER_CSV" "$OUT/baseline/results.csv"; then
        echo "[run_ep16_4bay] merged baseline into $MASTER_CSV"
    else
        echo "[run_ep16_4bay] WARN: merge into master failed"
    fi
fi

# Summary
echo
echo "===== HT vs FULLMESH baseline summary ====="
if [[ -f "$OUT/baseline/results.csv" ]]; then
    python3 contrib/nccl_ep/sweep/ep_summary.py "$OUT/baseline/results.csv" || true
fi

echo
echo "==========================================================="
echo "EP16 4-BAY runbook done."
echo "  Results dir : $OUT"
echo "  Baseline csv: $OUT/baseline/results.csv"
echo "  NV72 csv    : $OUT/ht_nv72_calibrate.csv"
echo "  Master csv  : $MASTER_CSV"
echo "==========================================================="
