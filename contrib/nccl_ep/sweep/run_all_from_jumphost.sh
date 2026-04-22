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
#     EXTRA_BENCH_ARGS  default ""
#                       Extra args passed through to every ep_bench call;
#                       e.g. EXTRA_BENCH_ARGS="--validate" for Phase-0
#                       correctness checks (pairs with [NV72-ADAPT] logs).
#     NCCL_EP_SKIP_BUILD  default 0
#                       Set to 1 to skip the auto-build step at start. By
#                       default the driver checks whether any source under
#                       contrib/nccl_ep is newer than the ep_bench binary
#                       and, if so, allocates -N1 to run `make` before any
#                       per-EP sweep salloc (cursor rule: C++/CUDA commits
#                       must always be matched with a make).
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
EXTRA_BENCH_ARGS="${EXTRA_BENCH_ARGS:-}"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-$HOME/fizhang/nccl-sweep-${TS}}"
AGG_CSV="$OUT_ROOT/all_results.csv"
AGG_SLOT_CSV="$OUT_ROOT/all_slot_results.csv"   # Phase 2 Stage 1 slot-alloc rows

SWEEP_SH="${NCCL_REPO}/contrib/nccl_ep/sweep/ep_sweep.sh"
PARSE_PY="${NCCL_REPO}/contrib/nccl_ep/sweep/ep_parse.py"
MERGE_PY="${NCCL_REPO}/contrib/nccl_ep/sweep/merge_into_master.py"

# Long-lived master CSV: every EP-size result is merged here, deduped by
# (ep_size, mode, tokens, dispatch_dtype_tag, algorithm) keeping the latest
# timestamp. Override with MASTER_CSV=/some/path if you want a different
# location.
MASTER_CSV="${MASTER_CSV:-$HOME/fizhang/nccl_ep_master.csv}"

mkdir -p "$OUT_ROOT"
mkdir -p "$(dirname "$MASTER_CSV")"
echo "==========================================================="
echo "EP Sweep Driver (jumphost)"
echo "  repo       : $NCCL_REPO"
echo "  ep_sizes   : $EP_SIZES"
echo "  tokens     : $TOKENS"
echo "  modes      : $MODES"
echo "  partition  : $PARTITION"
echo "  time limit : $TIME_LIMIT"
echo "  out_root   : $OUT_ROOT"
echo "  master csv : $MASTER_CSV"
echo "  extra args : ${EXTRA_BENCH_ARGS:-<none>}"
echo "==========================================================="

for sh in "$SWEEP_SH" "$PARSE_PY" "$MERGE_PY"; do
    if [[ ! -f "$sh" ]]; then
        echo "ERROR: missing $sh (checked out?)" >&2
        exit 2
    fi
done

# ---------------------------------------------------------------------------
# Auto-build: if any source under contrib/nccl_ep/ is newer than the ep_bench
# binary (or the binary is missing), trigger a small salloc -N1 to rebuild
# before the per-EP sweeps. Keeps the invariant from the cursor rule that
# C++/CUDA commits must always be matched with a `make` before running.
# Skip with NCCL_EP_SKIP_BUILD=1 if you know the binary is already current.
# ---------------------------------------------------------------------------
_autobuild_nccl_ep() {
    local build_dir="${NCCL_HOME:-$HOME/fizhang/nccl/build}"
    local bin="$build_dir/test/nccl_ep/ep_bench"

    if [[ "${NCCL_EP_SKIP_BUILD:-0}" == "1" ]]; then
        echo "[autobuild] NCCL_EP_SKIP_BUILD=1, skipping make"
        return 0
    fi

    local need_build=0
    if [[ ! -x "$bin" ]]; then
        echo "[autobuild] ep_bench binary missing, will build"
        need_build=1
    else
        # Any .cu/.cuh/.cc/.h/.hpp or Makefile newer than the binary triggers a build.
        local newer
        newer=$(find "$NCCL_REPO/contrib/nccl_ep" \
                      -type f \( -name '*.cu' -o -name '*.cuh' \
                               -o -name '*.cc' -o -name '*.h' \
                               -o -name '*.hpp' -o -name 'Makefile' \) \
                      -newer "$bin" 2>/dev/null | head -5)
        if [[ -n "$newer" ]]; then
            echo "[autobuild] sources newer than $bin:"
            echo "$newer" | sed 's/^/  /'
            need_build=1
        else
            echo "[autobuild] ep_bench up to date, skipping make"
        fi
    fi

    if [[ $need_build -eq 0 ]]; then return 0; fi

    echo "[autobuild] allocating -N1 to run make..."
    salloc -p "$PARTITION" -N 1 --gres=gpu:4 --cpus-per-gpu=8 --time=00:15:00 \
      srun --overlap -n 1 bash -lc "
        set -e
        cd ${NCCL_REPO}
        export NCCL_HOME='${build_dir}'
        export CUDA_HOME=\"\${CUDA_HOME:-/usr/local/cuda}\"
        export NVCC_GENCODE=\"\${NVCC_GENCODE:--gencode=arch=compute_103,code=sm_103}\"
        export MPI_HOME=\"\$(dirname \$(dirname \$(readlink -f \$(which mpirun))))\"
        export LD_LIBRARY_PATH=\"\${CUDA_HOME}/lib64:\${CUDA_HOME}/extras/CUPTI/lib64:\${NCCL_HOME}/lib:\${LD_LIBRARY_PATH:-}\"
        echo '[autobuild] building on '\$(hostname)
        time make -j3 -C contrib/nccl_ep MPI=1 BUILDDIR=\"\${NCCL_HOME}\" \
                  NVCC_GENCODE=\"\${NVCC_GENCODE}\" MPI_HOME=\"\${MPI_HOME}\"
        ls -l \"\${NCCL_HOME}/lib/libnccl_ep.so\" \"\${NCCL_HOME}/test/nccl_ep/ep_bench\"
      "
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "[autobuild] FAILED rc=$rc" >&2
        exit 3
    fi
    echo "[autobuild] done"
}

_autobuild_nccl_ep

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
                  EXTRA_BENCH_ARGS='${EXTRA_BENCH_ARGS}' \
                  bash ep_sweep.sh
           "
    rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "EP=$ep salloc/run failed (rc=$rc), continuing next size"
    fi

    # Aggregate CSV for this sweep run (under OUT_ROOT, keeps historical).
    # Two CSVs may be produced depending on which modes were run:
    #   results.csv       -- LL / HT production dispatch (ep_parse.py schema)
    #   slot_results.csv  -- Phase 2 Stage 1 slot-alloc microbench (inline schema)
    # Each has its own aggregator path; they are never merged together because
    # the column schemas are different.
    local slot_csv="$outdir/slot_results.csv"
    local any_produced=0

    if [[ -f "$csv" ]]; then
        if [[ ! -f "$AGG_CSV" ]]; then
            cp "$csv" "$AGG_CSV"
        else
            tail -n +2 "$csv" >> "$AGG_CSV"
        fi
        echo "EP=$ep -> appended $(wc -l <"$csv") rows to $AGG_CSV"

        # Also merge into the long-lived master CSV (dedup by
        # ep_size/mode/tokens/dispatch_dtype_tag/algorithm). This makes the
        # master grow smoothly as we run more EP sizes over time, and
        # reruns of the same config replace older rows.
        if python3 "$MERGE_PY" "$MASTER_CSV" "$csv"; then
            echo "EP=$ep -> merged into $MASTER_CSV"
        else
            echo "EP=$ep -> WARN: merge into master failed (csv kept in $csv)"
        fi
        any_produced=1
    fi

    if [[ -f "$slot_csv" ]]; then
        if [[ ! -f "$AGG_SLOT_CSV" ]]; then
            cp "$slot_csv" "$AGG_SLOT_CSV"
        else
            tail -n +2 "$slot_csv" >> "$AGG_SLOT_CSV"
        fi
        echo "EP=$ep -> appended $(wc -l <"$slot_csv") slot rows to $AGG_SLOT_CSV"
        any_produced=1
    fi

    if [[ $any_produced -eq 0 ]]; then
        echo "EP=$ep produced no csv"
    fi
}

for ep in $EP_SIZES; do
    run_one_size "$ep"
done

echo
echo "==========================================================="
echo "ALL DONE."
echo "  Combined HT/LL CSV   : $AGG_CSV"
ls -l "$AGG_CSV" 2>/dev/null || echo "    (no HT/LL CSV produced)"
echo "  Combined slot CSV    : $AGG_SLOT_CSV"
ls -l "$AGG_SLOT_CSV" 2>/dev/null || echo "    (no slot CSV produced)"
echo "==========================================================="
