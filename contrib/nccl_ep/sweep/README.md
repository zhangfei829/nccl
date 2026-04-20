# NCCL EP Sweep (DeepSeek-V3 shape, NVL72)

Automates sweeping EP size x tokens-per-rank x mode against `ep_bench`
and collects results into a single CSV ready for Excel pivoting.

## Files

| file                        | role                                           |
|-----------------------------|------------------------------------------------|
| `ep_parse.py`               | Parse one `ep_bench` log, append one CSV row.  |
| `ep_sweep.sh`               | Inner driver: run all (mode x tokens) for one already-allocated EP size. |
| `run_all_from_jumphost.sh`  | Outer driver: loop EP sizes, allocate + run inner. |

## Sweep axes (defaults)

- **EP sizes**   : 4, 8, 16, 32, 64
- **tokens/rank**: 16, 32, 64, 128, 256, 4096, 8192
- **modes**      :
  - `ll`       -- Low-latency + BF16 dispatch + BF16 combine
  - `ht_bf16`  -- High-throughput + BF16 dispatch + BF16 combine
  - `ht_fp8`   -- High-throughput + FP8 dispatch + BF16 combine

- Fixed: `hidden=7168`, `top_k=8`, `experts=256` (DeepSeek-V3).
- LL runs skip configs where `tokens*EP > 2048` (LL buffer scales as
  `num_local_experts * max_tokens_per_rank * nRanks` and blows up past that).
- HT caps at `tokens<=8192` (compile-time `MAX_SUPPORTED_TOKENS_PER_RANK`).
- LL does not support FP8 dispatch (README), so `ll_fp8` is not in the default set.

## One-shot usage (jumphost)

Assumes repo is checked out at `$HOME/fizhang/nccl` and NCCL+nccl_ep are
already built under `$HOME/fizhang/nccl/build` (including the fabric-memory
patch for NVL72 cross-bay HT).

```bash
cd ~/fizhang/nccl
bash contrib/nccl_ep/sweep/run_all_from_jumphost.sh
```

Outputs under `$OUT_ROOT` (default `$HOME/fizhang/nccl-sweep-<ts>/`):

```
nccl-sweep-<ts>/
  all_results.csv           <-- single file for Excel
  ep4/
    results.csv
    ep4_ll_t128.log
    ep4_ht_bf16_t4096.log
    sweep.log
    hosts.ep4
  ep8/
    ...
```

## Running a single EP size manually

If you already have an `salloc` open and are sitting on the head node:

```bash
cd ~/fizhang/nccl/contrib/nccl_ep/sweep

# required env (see "Pitfalls")
unset SLURM_TRES_PER_TASK

# pick what to run
EP_SIZE=8 TOKENS="128 4096 8192" MODES="ht_bf16 ht_fp8" \
  bash ep_sweep.sh
```

Results go under `./sweep_<ts>_ep8/results.csv`.

## CSV columns (Excel-ready)

```
timestamp, algorithm, ranks, tokens, hidden, top_k, experts,
dispatch_dtype, max_tokens_per_rank,
dispatch_avg_us, dispatch_min_us, dispatch_max_us,
dispatch_kernel_us,
dispatch_bw_gbs, dispatch_recv_bw_gbs, dispatch_send_bw_gbs,
combine_avg_us, combine_min_us, combine_max_us,
combine_kernel_us,
combine_bw_gbs, combine_send_bw_gbs, combine_recv_bw_gbs,
total_dc_avg_us,
setup_group_ms, setup_handle_ms,
total_send_mb, total_recv_mb,
nvl_send_mb, nvl_recv_mb, rdma_send_mb, rdma_recv_mb,
log_path, mode, dispatch_dtype_tag, ep_size, wall_s
```

Recommended Excel pivots:

- Dispatch BW (`dispatch_bw_gbs`) vs. `tokens`, one line per `(ep_size, mode)`
- Dispatch latency (`dispatch_avg_us`) vs. `tokens`, one line per `(ep_size, mode)`
- Combine BW (`combine_bw_gbs`) vs. `tokens`, one line per `(ep_size, mode)`

## Pitfalls

- **`srun --pty bash -l`** (how you usually enter the compute node) injects
  `SLURM_TRES_PER_TASK=cpu=32` into the shell. Open MPI's `--mca plm slurm`
  then spawns `srun` for `orted` which fails with:
    > `--cpus-per-task, --tres-per-task, --cpus-per-gpu are mutually exclusive`
  `ep_sweep.sh` does `unset SLURM_TRES_PER_TASK` internally, but if you run
  `mpirun` by hand you must do it yourself.
- **Do not batch-unset `SLURM_*`**. OMPI's slurm plm needs a handful of
  SLURM state variables; clearing them yields `orte_plm_base_select failed
  (Not found)`.
- **NVL72 cross-bay HT** requires the fabric-memory patch in
  `nccl_ep.cc` (commit `defc3f9`). Without it, HT on nNodes>1 tries to
  initialize GIN/RDMA and fails when RoCE GID isn't fully configured.
- **EP=64 with top_k=8, experts=256** puts `num_local_experts=4`, less than
  `top_k`. `ep_bench` doesn't refuse this, but load-imbalance grows because
  most token top-k choices land off-rank; numbers are still meaningful, but
  interpret as "high fan-out" not "saturated NVLink".
