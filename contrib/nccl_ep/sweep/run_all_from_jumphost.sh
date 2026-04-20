#!/usr/bin/env bash
# =============================================================================
# run_all_from_jumphost.sh
#
# Drive the full EP sweep across EP sizes {4, 8, 16, 32, 64} from a jumphost
# that can submit to the gb300 partition. Each EP size is allocated
# independently (so larger sizes can wait in queue without blocking smaller
# ones), the inner sweep (ep_sweep.sh) is launched once nodes are ready, and
# the CSV is accumulated into a single file.
#
# Usage (from jumphost, repo checked out under $HOME/fizhang/nccl):
#     ./contrib/nccl_ep/sweep/run_all_from_jumphost.sh
#
# Env overrides:
#     EP_SIZES       default "4 8 16 32 64"
#     TOKENS         default "16 32 64 128 256 4096 8192"
#     MODES          default "ll ht_bf16 ht_fp8"
#     PARTITION      default gb300
#     TIME_LIMIT     default 02:00:00
#     OUT_ROOT       default $HOME/fizhang/nccl-sweep-<ts>
#     NCCL_REPO      default $HOME/fizhang/nccl
#
# Each EP_SIZE N is mapped to a topology:
#     4 -> -N1 (single bay, 4 GPU)
#     others -> -N<ceil(N/4)>, --gres=gpu:4 per node
# All sizes fit inside a single NVL72 pod (72 GPU) and will use MNNVL fabric
# memory path that we patched into nccl_ep.
# =============================================================================
set -u

EP_SIZES="${EP_SIZES:-4 8 16 32 64}"
TOKENS="${TOKENS:-16 32 64 128 256 4096 8192}"
MODES="${MODES:-ll ht_bf16 ht_fp8}"
PARTITION="${PARTITION:-gb300}"
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
NCCL_REPO="${NCCL_REPO:-$HOME/fizhang/nccl}"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-$HOME/fizhang/nccl-sweep-${TS}}"
AGG_CSV="$OUT_ROOT/all_results.csv"

SWEEP_SH="${NCCL_REPO}/contrib/nccl_ep/sweep/ep_sweep.sh"
PARSE_PY="${NCCL_REPO}/contrib/nccl_ep/sweep/ep_parse.py"

mkdir -p "$OUT_ROOT"
echo "==========================================================="
echo "EP Sweep Driver (jumphost)"
echo "  repo       : $NCCL_REPO"
echo "  ep_sizes   : $EP_SIZES"
echo "  tokens     : $TOKENS"
echo "  modes      : $MODES"
echo "  partition  : $PARTITION"
echo "  time limit : $TIME_LIMIT"
echo "  out_root   : $OUT_ROOT"
echo "==========================================================="

for sh in "$SWEEP_SH" "$PARSE_PY"; do
    if [[ ! -f "$sh" ]]; then
        echo "ERROR: missing $sh (checked out?)" >&2
        exit 2
    fi
done

run_one_size() {
    local ep="$1"
    local nnodes=$(( (ep + 3) / 4 ))   # 4 GPU per node
    local outdir="$OUT_ROOT/ep${ep}"
    local csv="$outdir/results.csv"

    mkdir -p "$outdir"
    echo
    echo "######  EP=$ep  (nodes=$nnodes)  ######"

    # NOTE on allocation model (why no inner srun):
    # We want mpirun's `--mca plm slurm` to launch orted with full GPU
    # visibility on every allocated node. If we wrap ep_sweep.sh inside an
    # inner `srun -N1 -w head_node` step, that inner step grabs the head
    # node's GPU resource as a Slurm step, and the orted steps that
    # mpirun starts on OTHER nodes end up with no GPU visibility
    # ("cudaSetDevice: invalid device ordinal" at ep_bench.cu:1862).
    # The fix is to run ep_sweep.sh directly in the salloc shell
    # (which executes on the submit host). mpirun then starts fresh
    # Slurm steps for orted on every node and they inherit the job's
    # full GRES allocation.

    salloc -p "$PARTITION" -N "$nnodes" --ntasks-per-node=1 \
           --gres=gpu:4 --cpus-per-gpu=8 --time="$TIME_LIMIT" \
           bash -lc "
              set -u
              unset SLURM_TRES_PER_TASK || true
              cd ${NCCL_REPO}/contrib/nccl_ep/sweep
              export NCCL_HOME=\"${NCCL_HOME:-\$HOME/fizhang/nccl/build}\"
              export CUDA_HOME=\"${CUDA_HOME:-/usr/local/cuda}\"
              EP_SIZE=${ep} TOKENS='${TOKENS}' MODES='${MODES}' \
                  LOG_DIR='${outdir}' CSV_FILE='${csv}' \
                  bash ep_sweep.sh
           "
    rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "EP=$ep salloc/run failed (rc=$rc), continuing next size"
    fi

    # Aggregate CSV
    if [[ -f "$csv" ]]; then
        if [[ ! -f "$AGG_CSV" ]]; then
            cp "$csv" "$AGG_CSV"
        else
            tail -n +2 "$csv" >> "$AGG_CSV"
        fi
        echo "EP=$ep -> appended $(wc -l <"$csv") rows to $AGG_CSV"
    else
        echo "EP=$ep produced no csv"
    fi
}

for ep in $EP_SIZES; do
    run_one_size "$ep"
done

echo
echo "==========================================================="
echo "ALL DONE. Combined CSV: $AGG_CSV"
ls -l "$AGG_CSV" 2>/dev/null || echo "(no CSV produced)"
echo "==========================================================="
