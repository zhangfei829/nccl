#!/usr/bin/env bash
# =============================================================================
# ep_sweep.sh
#
# Run ep_bench across a sweep of (mode x tokens) for an EP size that has
# ALREADY been allocated via salloc. Designed for NCCL EP + MNNVL on NVL72.
#
# Usage (inside the interactive shell of the head node):
#     EP_SIZE=8  ./ep_sweep.sh              # auto tokens/modes
#     EP_SIZE=8 TOKENS="128 4096" ./ep_sweep.sh
#     EP_SIZE=16 MODES="ht_fp8 ht_bf16" ./ep_sweep.sh
#
# Environment:
#     EP_SIZE          Number of MPI ranks / GPUs in the EP group (required)
#     TOKENS           Space-separated list of per-rank token counts
#                      Default: "16 32 64 128 256 4096 8192"
#     MODES            Which (algorithm,dtype) combos to run.
#                      Valid tokens in MODES: ll ht_bf16 ht_fp8
#                      Default: "ll ht_bf16 ht_fp8"
#     HIDDEN           hidden dim (default 7168)
#     TOPK             top-k per token (default 8)
#     EXPERTS          total experts (default 256)
#     WARMUP_LL / ITERS_LL       default 20 / 50
#     WARMUP_HT / ITERS_HT       default 20 / 30
#     NCCL_HOME        default $HOME/fizhang/nccl/build
#     CUDA_HOME        default /usr/local/cuda
#     LOG_DIR          default ./sweep_<ts>_ep<size>
#     CSV_FILE         default $LOG_DIR/results.csv
#     EXTRA_BENCH_ARGS extra args appended to every ep_bench invocation
#                      (e.g. EXTRA_BENCH_ARGS="--validate" for Phase 0 checks)
#
# Output:
#     <LOG_DIR>/ep{N}_{mode}_t{tokens}.log   raw ep_bench output
#     <LOG_DIR>/results.csv                  one row per config, Excel-ready
#
# Assumptions:
#     * Current shell has $SLURM_JOB_ID / $SLURM_JOB_NODELIST set
#     * SLURM_TRES_PER_TASK is already unset (see README pitfalls)
#     * nccl_ep libraries + ep_bench binary available under $NCCL_HOME
# =============================================================================

set -u
# NOTE: don't "set -e" -- a single bench fail should not kill the sweep.

: "${EP_SIZE:?EP_SIZE is required (e.g. EP_SIZE=8)}"

TOKENS_DEFAULT="16 32 64 128 256 4096 8192"
MODES_DEFAULT="ll ht_bf16 ht_fp8"

TOKENS="${TOKENS:-$TOKENS_DEFAULT}"
MODES="${MODES:-$MODES_DEFAULT}"
HIDDEN="${HIDDEN:-7168}"
TOPK="${TOPK:-8}"
EXPERTS="${EXPERTS:-256}"
WARMUP_LL="${WARMUP_LL:-20}"
ITERS_LL="${ITERS_LL:-50}"
WARMUP_HT="${WARMUP_HT:-20}"
ITERS_HT="${ITERS_HT:-30}"
EXTRA_BENCH_ARGS="${EXTRA_BENCH_ARGS:-}"

NCCL_HOME="${NCCL_HOME:-$HOME/fizhang/nccl/build}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
EP_BENCH="${NCCL_HOME}/test/nccl_ep/ep_bench"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-$PWD/sweep_${TS}_ep${EP_SIZE}}"
CSV_FILE="${CSV_FILE:-$LOG_DIR/results.csv}"

mkdir -p "$LOG_DIR"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
EP_PARSE="${EP_PARSE:-$SCRIPT_DIR/ep_parse.py}"

# LD_LIBRARY_PATH for nccl_ep + CUPTI + MPI
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/extras/CUPTI/lib64:${NCCL_HOME}/lib:${LD_LIBRARY_PATH:-}"
export NCCL_GIN_TYPE="${NCCL_GIN_TYPE:-3}"

# Must unset to avoid srun's --cpus-per-gpu / --tres-per-task mutex when mpirun
# launches orted internally; harmless if already unset.
unset SLURM_TRES_PER_TASK || true

# Generate hostfile from current allocation
HOSTFILE="$LOG_DIR/hosts.ep${EP_SIZE}"
if [[ -z "${SLURM_JOB_NODELIST:-}" ]]; then
    echo "[ep_sweep] ERROR: SLURM_JOB_NODELIST is not set; are you inside an salloc?" >&2
    exit 2
fi
scontrol show hostnames "$SLURM_JOB_NODELIST" \
    | awk -v s=4 '{print $1" slots="s}' > "$HOSTFILE"

echo "==========================================================="
echo "EP Sweep"
echo "  EP_SIZE     : $EP_SIZE"
echo "  tokens      : $TOKENS"
echo "  modes       : $MODES"
echo "  hidden      : $HIDDEN"
echo "  topk/experts: $TOPK / $EXPERTS"
echo "  log dir     : $LOG_DIR"
echo "  csv         : $CSV_FILE"
echo "  extra bench : ${EXTRA_BENCH_ARGS:-<none>}"
echo "  hostfile    :"
sed 's/^/    /' "$HOSTFILE"
echo "  LD_LIBRARY_PATH : $LD_LIBRARY_PATH"
echo "  NCCL_GIN_TYPE   : $NCCL_GIN_TYPE"
echo "==========================================================="

# Verify binaries and libs
if [[ ! -x "$EP_BENCH" ]]; then
    echo "[ep_sweep] ERROR: $EP_BENCH not found/executable" >&2
    exit 2
fi
if ! command -v mpirun >/dev/null 2>&1; then
    echo "[ep_sweep] ERROR: mpirun not found in PATH" >&2
    exit 2
fi
if ! command -v python3 >/dev/null 2>&1; then
    echo "[ep_sweep] ERROR: python3 not found (required for ep_parse.py)" >&2
    exit 2
fi
if [[ ! -f "$EP_PARSE" ]]; then
    echo "[ep_sweep] ERROR: ep_parse.py not found at $EP_PARSE" >&2
    exit 2
fi

# ------------------------ helpers ------------------------
should_skip() {
    local mode="$1" tokens="$2"
    case "$mode" in
        ll)
            # LL output buffer = [L, max*nRanks, H], grows as O(ranks * tokens).
            # Empirical safe upper bound on GB300 96GB: tokens*nRanks <= 2048
            # i.e. LL skips very large tokens at large EP size.
            if (( tokens * EP_SIZE > 2048 )); then return 0; fi
            ;;
        ht_bf16|ht_fp8)
            # HT: MAX_SUPPORTED_TOKENS_PER_RANK is compiled at 8192.
            if (( tokens > 8192 )); then return 0; fi
            ;;
    esac
    return 1
}

mode_to_algo_args() {
    # sets global vars: ALGO WARMUP ITERS EXTRA_ARGS DTYPE_TAG
    local mode="$1"
    case "$mode" in
        ll)
            ALGO="ll"; WARMUP="$WARMUP_LL"; ITERS="$ITERS_LL"
            EXTRA_ARGS=""; DTYPE_TAG="bf16"
            ;;
        ht_bf16)
            ALGO="ht"; WARMUP="$WARMUP_HT"; ITERS="$ITERS_HT"
            EXTRA_ARGS=""; DTYPE_TAG="bf16"
            ;;
        ht_fp8)
            ALGO="ht"; WARMUP="$WARMUP_HT"; ITERS="$ITERS_HT"
            EXTRA_ARGS="--use-fp8"; DTYPE_TAG="fp8"
            ;;
        *)
            echo "[ep_sweep] ERROR: unknown mode '$mode'" >&2
            return 1
            ;;
    esac
    return 0
}

# ------------------------ run loop ------------------------
total=0; ran=0; skipped=0; failed=0
for mode in $MODES; do
    mode_to_algo_args "$mode" || continue
    for t in $TOKENS; do
        total=$((total+1))
        if should_skip "$mode" "$t"; then
            printf "[skip] mode=%-7s tokens=%-5s (size limit)\n" "$mode" "$t" | tee -a "$LOG_DIR/sweep.log"
            skipped=$((skipped+1))
            continue
        fi
        name="ep${EP_SIZE}_${mode}_t${t}"
        logf="$LOG_DIR/${name}.log"
        printf "\n[run ] %-30s ... " "$name" | tee -a "$LOG_DIR/sweep.log"
        start=$(date +%s)
        mpirun --mca plm slurm -np "$EP_SIZE" --hostfile "$HOSTFILE" \
               --oversubscribe --bind-to none \
               -x LD_LIBRARY_PATH -x NCCL_GIN_TYPE \
            "$EP_BENCH" --algorithm "$ALGO" \
                        --tokens "$t" --hidden "$HIDDEN" \
                        --top-k "$TOPK" --experts "$EXPERTS" \
                        --warmup "$WARMUP" --iters "$ITERS" $EXTRA_ARGS $EXTRA_BENCH_ARGS \
            > "$logf" 2>&1
        rc=$?
        wall=$(( $(date +%s) - start ))
        if [[ $rc -eq 0 ]]; then
            echo "OK (${wall}s)" | tee -a "$LOG_DIR/sweep.log"
            python3 "$EP_PARSE" "$logf" "$CSV_FILE" \
                    "mode=$mode" "dispatch_dtype_tag=$DTYPE_TAG" \
                    "ep_size=$EP_SIZE" "wall_s=$wall" \
                | tee -a "$LOG_DIR/sweep.log"
            ran=$((ran+1))
        else
            echo "FAIL rc=$rc (${wall}s) -> $logf" | tee -a "$LOG_DIR/sweep.log"
            failed=$((failed+1))
        fi
    done
done

echo | tee -a "$LOG_DIR/sweep.log"
echo "================== sweep done ==================" | tee -a "$LOG_DIR/sweep.log"
echo "total=$total  ran=$ran  skipped=$skipped  failed=$failed" | tee -a "$LOG_DIR/sweep.log"
echo "CSV : $CSV_FILE" | tee -a "$LOG_DIR/sweep.log"
echo "LOGS: $LOG_DIR" | tee -a "$LOG_DIR/sweep.log"

# quick table view on stdout (best effort; requires column)
if command -v column >/dev/null 2>&1 && [[ -s "$CSV_FILE" ]]; then
    echo
    echo "--- first rows ---"
    head -15 "$CSV_FILE" | column -t -s,
fi
