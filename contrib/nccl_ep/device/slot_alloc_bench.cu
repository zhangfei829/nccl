/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */
// Phase 2 Stage 1: slot allocation micro-benchmark.
//
// commit 2: scan-based lean implementation lands here. atomic and cumsum
// remain stub and will be filled in commits 3/4. The scan path here is
// intentionally NOT call_metadata_preprocessing - that function does a pile
// of HT-specific preprocessing (rdma_to_attn_map, local_expert_routing_map,
// per-expert counts, ...) that a hypothetical Phase 2 "scan" backend would
// not ship. We only compute the sparse_to_dense_map absolute-slot table,
// which is the minimum this algorithm needs to produce my_slot_at_dest.
//
// Output slot semantics: absolute slot in a layout-3 compact packing of
// the dest recv buffer, where tokens are ordered by (src_rank, local_t).
// Value -1 means (t, k) is masked or does not target that dest.

#include "slot_alloc_bench.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/device/device_scan.cuh>
#include <mpi.h>

// NOTE: hybridep_adapter.cuh's convert_topk_to_routing_map lives in
// libnccl_ep.so but is compiled with -fvisibility=hidden, so the symbol is
// not exported. Rather than poke visibility on a product library for a
// benchmark, we keep slot_alloc_bench self-contained and reimplement the
// same 30-line bitmap kernel locally in an anonymous namespace below.

#define SLOT_BENCH_CUDA_CHECK(expr) do {                                    \
    cudaError_t _e = (expr);                                                \
    if (_e != cudaSuccess) {                                                \
        fprintf(stderr, "[slot_bench] CUDA error %s:%d: %s\n",              \
                __FILE__, __LINE__, cudaGetErrorString(_e));                \
        std::abort();                                                       \
    }                                                                       \
} while (0)

#define SLOT_BENCH_NCCL_CHECK(expr) do {                                    \
    ncclResult_t _r = (expr);                                               \
    if (_r != ncclSuccess) {                                                \
        fprintf(stderr, "[slot_bench] NCCL error %s:%d: %s\n",              \
                __FILE__, __LINE__, ncclGetErrorString(_r));                \
        std::abort();                                                       \
    }                                                                       \
} while (0)

#define SLOT_BENCH_CU_CHECK(expr) do {                                      \
    CUresult _r = (expr);                                                   \
    if (_r != CUDA_SUCCESS) {                                               \
        const char* _msg = nullptr;                                         \
        cuGetErrorString(_r, &_msg);                                        \
        fprintf(stderr, "[slot_bench] CU error %s:%d: %s\n",                \
                __FILE__, __LINE__, _msg ? _msg : "?");                     \
        std::abort();                                                       \
    }                                                                       \
} while (0)

#define SLOT_BENCH_MPI_CHECK(expr) do {                                     \
    int _r = (expr);                                                        \
    if (_r != MPI_SUCCESS) {                                                \
        fprintf(stderr, "[slot_bench] MPI error %s:%d: %d\n",               \
                __FILE__, __LINE__, _r);                                    \
        std::abort();                                                       \
    }                                                                       \
} while (0)

namespace nccl_ep { namespace slot_bench {

// ============================================================================
// scan-based lean implementation
// ============================================================================

namespace {

// Implementation-private state cached across iterations. Re-allocates if the
// shape changes between calls.
struct ScanImpl {
    bool     initialized       = false;
    int      tokens_cached     = 0;
    int      top_k_cached      = 0;
    int      num_experts_cached= 0;
    int      nRanks_cached     = 0;

    uint8_t* d_local_bitmap    = nullptr;  // [tokens, experts_packed]
    uint8_t* d_global_bitmap   = nullptr;  // [nRanks * tokens, experts_packed]
    uint8_t* d_routed_T        = nullptr;  // [nRanks][N] bool (1 per (dest, global_token))
    int32_t* d_s2d_map_T       = nullptr;  // [nRanks][N] exclusive-sum result per dest
    int32_t* d_s2d_map         = nullptr;  // [N][nRanks] final absolute slot (or -1)

    void*    d_cub_tmp         = nullptr;
    size_t   cub_tmp_bytes     = 0;

    cudaEvent_t ev_start       = nullptr;
    cudaEvent_t ev_local_done  = nullptr;
    cudaEvent_t ev_ag_done     = nullptr;
    cudaEvent_t ev_scan_done   = nullptr;
    cudaEvent_t ev_extract_done= nullptr;
};

static ScanImpl g_scan;

static void destroy_scan_state(ScanImpl& s) {
    if (!s.initialized) return;
    cudaFree(s.d_local_bitmap);
    cudaFree(s.d_global_bitmap);
    cudaFree(s.d_routed_T);
    cudaFree(s.d_s2d_map_T);
    cudaFree(s.d_s2d_map);
    cudaFree(s.d_cub_tmp);
    cudaEventDestroy(s.ev_start);
    cudaEventDestroy(s.ev_local_done);
    cudaEventDestroy(s.ev_ag_done);
    cudaEventDestroy(s.ev_scan_done);
    cudaEventDestroy(s.ev_extract_done);
    s = ScanImpl{};
}

static void ensure_scan_state(const SlotAllocParams& p) {
    bool same_shape = g_scan.initialized
                   && g_scan.tokens_cached      == p.tokens
                   && g_scan.top_k_cached       == p.top_k
                   && g_scan.num_experts_cached == p.num_experts
                   && g_scan.nRanks_cached      == p.nRanks;
    if (same_shape) return;
    if (g_scan.initialized) destroy_scan_state(g_scan);

    const int experts_packed = (p.num_experts + 7) / 8;
    const size_t N = static_cast<size_t>(p.nRanks) * p.tokens;

    SLOT_BENCH_CUDA_CHECK(cudaMalloc(&g_scan.d_local_bitmap,
                                     static_cast<size_t>(p.tokens) * experts_packed));
    SLOT_BENCH_CUDA_CHECK(cudaMalloc(&g_scan.d_global_bitmap,
                                     N * experts_packed));
    SLOT_BENCH_CUDA_CHECK(cudaMalloc(&g_scan.d_routed_T,
                                     static_cast<size_t>(p.nRanks) * N));
    SLOT_BENCH_CUDA_CHECK(cudaMalloc(&g_scan.d_s2d_map_T,
                                     static_cast<size_t>(p.nRanks) * N * sizeof(int32_t)));
    SLOT_BENCH_CUDA_CHECK(cudaMalloc(&g_scan.d_s2d_map,
                                     N * p.nRanks * sizeof(int32_t)));

    // Determine CUB temp size for one ExclusiveSum of length N (same for all dest columns)
    size_t bytes = 0;
    cub::DeviceScan::ExclusiveSum(
        nullptr, bytes,
        static_cast<uint8_t*>(nullptr),
        static_cast<int32_t*>(nullptr),
        static_cast<int>(N));
    SLOT_BENCH_CUDA_CHECK(cudaMalloc(&g_scan.d_cub_tmp, bytes));
    g_scan.cub_tmp_bytes = bytes;

    SLOT_BENCH_CUDA_CHECK(cudaEventCreate(&g_scan.ev_start));
    SLOT_BENCH_CUDA_CHECK(cudaEventCreate(&g_scan.ev_local_done));
    SLOT_BENCH_CUDA_CHECK(cudaEventCreate(&g_scan.ev_ag_done));
    SLOT_BENCH_CUDA_CHECK(cudaEventCreate(&g_scan.ev_scan_done));
    SLOT_BENCH_CUDA_CHECK(cudaEventCreate(&g_scan.ev_extract_done));

    g_scan.tokens_cached      = p.tokens;
    g_scan.top_k_cached       = p.top_k;
    g_scan.num_experts_cached = p.num_experts;
    g_scan.nRanks_cached      = p.nRanks;
    g_scan.initialized        = true;
}

// Kernel: local topk_idx -> local bitmap (mirror of
// nccl_ep::hybridep::convert_topk_to_routing_map_kernel, copied here to avoid
// depending on a -fvisibility=hidden symbol in libnccl_ep.so).
// Caller must zero routing_bitmap before launch since we OR into it.
__global__ void topk_to_bitmap_local_kernel(
    const int64_t* __restrict__ topk_idx,
    uint8_t*       __restrict__ routing_bitmap,
    int tokens,
    int top_k,
    int experts_packed)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= tokens) return;

    uint8_t* row = routing_bitmap + static_cast<size_t>(t) * experts_packed;
    for (int k = 0; k < top_k; k++) {
        int64_t eid = topk_idx[static_cast<size_t>(t) * top_k + k];
        if (eid >= 0) {
            row[eid >> 3] |= static_cast<uint8_t>(1u << (eid & 7));
        }
    }
}

static inline void topk_to_bitmap_local(
    const int64_t* d_topk_idx,
    uint8_t* d_bitmap,
    int tokens, int top_k, int experts_packed,
    cudaStream_t stream)
{
    int block = 256;
    int grid  = (tokens + block - 1) / block;
    topk_to_bitmap_local_kernel<<<grid, block, 0, stream>>>(
        d_topk_idx, d_bitmap, tokens, top_k, experts_packed);
}

// Kernel: global_bitmap[N, experts_packed] -> routed_T[nRanks][N] bool.
// routed_T[d][t_g] = 1 if any expert in [d*L, (d+1)*L) is set in bitmap row t_g.
__global__ void bitmap_to_routed_T_kernel(
    const uint8_t* __restrict__ global_bitmap,
    uint8_t* __restrict__ routed_T,
    int N,
    int nRanks,
    int num_local_experts,
    int experts_packed)
{
    int t_g = blockIdx.x * blockDim.x + threadIdx.x;
    int d   = blockIdx.y;
    if (t_g >= N || d >= nRanks) return;

    const uint8_t* row = global_bitmap + static_cast<size_t>(t_g) * experts_packed;
    const int eid_lo = d * num_local_experts;
    const int eid_hi = eid_lo + num_local_experts;

    uint8_t routed = 0;
    for (int e = eid_lo; e < eid_hi; e++) {
        if (row[e >> 3] & (1u << (e & 7))) { routed = 1; break; }
    }
    routed_T[static_cast<size_t>(d) * N + t_g] = routed;
}

// Kernel: combine transposed scan back and flip into s2d_map[t_g][d].
// Untransposes [nRanks][N] int32 into [N][nRanks] int32 at the same time.
__global__ void combine_s2d_kernel(
    const uint8_t* __restrict__ routed_T,
    const int32_t* __restrict__ s2d_map_T,
    int32_t* __restrict__ s2d_map,
    int N,
    int nRanks)
{
    int t_g = blockIdx.x * blockDim.x + threadIdx.x;
    int d   = blockIdx.y;
    if (t_g >= N || d >= nRanks) return;

    size_t idx_T = static_cast<size_t>(d) * N + t_g;
    uint8_t r    = routed_T[idx_T];
    int32_t s    = s2d_map_T[idx_T];
    s2d_map[static_cast<size_t>(t_g) * nRanks + d] = r ? s : -1;
}

// Kernel: per (t, k) lookup my_slot_at_dest using the my-rank segment of s2d_map.
// my_slot_at_dest[t, k] = s2d_map[myRank * tokens + t, topk_idx[t,k] / num_local_experts]
// or -1 if topk_idx[t,k] < 0.
__global__ void extract_my_slot_kernel(
    const int64_t* __restrict__ topk_idx,
    const int32_t* __restrict__ s2d_map,
    int32_t* __restrict__ my_slot_at_dest,
    int tokens,
    int top_k,
    int nRanks,
    int myRank,
    int num_local_experts)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y;
    if (t >= tokens || k >= top_k) return;

    int64_t eid = topk_idx[static_cast<size_t>(t) * top_k + k];
    int32_t slot = -1;
    if (eid >= 0) {
        int dest = static_cast<int>(eid / static_cast<int64_t>(num_local_experts));
        int t_g  = myRank * tokens + t;
        slot     = s2d_map[static_cast<size_t>(t_g) * nRanks + dest];
    }
    my_slot_at_dest[static_cast<size_t>(t) * top_k + k] = slot;
}

}  // anonymous namespace

SlotAllocTiming run_scan_based(const SlotAllocParams& p) {
    ensure_scan_state(p);

    const int experts_packed = (p.num_experts + 7) / 8;
    const size_t N = static_cast<size_t>(p.nRanks) * p.tokens;

    SLOT_BENCH_CUDA_CHECK(cudaEventRecord(g_scan.ev_start, p.stream));

    // Step 1: local topk_idx -> bitmap. topk_to_bitmap_local uses bitwise OR
    // so the row must be zeroed first (mirror of nccl_ep.cc:2043 pattern).
    SLOT_BENCH_CUDA_CHECK(cudaMemsetAsync(
        g_scan.d_local_bitmap, 0,
        static_cast<size_t>(p.tokens) * experts_packed, p.stream));
    topk_to_bitmap_local(
        p.d_topk_idx, g_scan.d_local_bitmap,
        p.tokens, p.top_k, experts_packed, p.stream);
    SLOT_BENCH_CUDA_CHECK(cudaEventRecord(g_scan.ev_local_done, p.stream));

    // Step 2: allgather the per-rank bitmap into a concatenated global bitmap.
    // Layout: [nRanks * tokens, experts_packed], so dest row for global token
    // t_g = src_rank * tokens + t_local is at byte offset t_g * experts_packed.
    SLOT_BENCH_NCCL_CHECK(ncclAllGather(
        g_scan.d_local_bitmap, g_scan.d_global_bitmap,
        static_cast<size_t>(p.tokens) * experts_packed, ncclUint8,
        p.comm, p.stream));
    SLOT_BENCH_CUDA_CHECK(cudaEventRecord(g_scan.ev_ag_done, p.stream));

    // Step 3: expand bitmap -> per-(dest, global_token) bool, transposed.
    {
        dim3 block(256);
        dim3 grid(static_cast<unsigned>((N + block.x - 1) / block.x),
                  static_cast<unsigned>(p.nRanks));
        bitmap_to_routed_T_kernel<<<grid, block, 0, p.stream>>>(
            g_scan.d_global_bitmap, g_scan.d_routed_T,
            static_cast<int>(N), p.nRanks, p.num_local_experts, experts_packed);
    }

    // Step 4: per-dest ExclusiveSum over the [N] bool column -> absolute slot
    // before this token. nRanks launches; each launch is independent so the
    // driver pipelines them on the same stream.
    for (int d = 0; d < p.nRanks; d++) {
        size_t bytes = g_scan.cub_tmp_bytes;
        cub::DeviceScan::ExclusiveSum(
            g_scan.d_cub_tmp, bytes,
            g_scan.d_routed_T  + static_cast<size_t>(d) * N,
            g_scan.d_s2d_map_T + static_cast<size_t>(d) * N,
            static_cast<int>(N), p.stream);
    }

    // Step 5: combine scan result + routed mask into [N, nRanks] s2d_map
    // with -1 for non-routed entries. Also untransposes in one pass.
    {
        dim3 block(256);
        dim3 grid(static_cast<unsigned>((N + block.x - 1) / block.x),
                  static_cast<unsigned>(p.nRanks));
        combine_s2d_kernel<<<grid, block, 0, p.stream>>>(
            g_scan.d_routed_T, g_scan.d_s2d_map_T, g_scan.d_s2d_map,
            static_cast<int>(N), p.nRanks);
    }
    SLOT_BENCH_CUDA_CHECK(cudaEventRecord(g_scan.ev_scan_done, p.stream));

    // Step 6: extract my_slot_at_dest from the my-rank segment of s2d_map.
    {
        dim3 block(256);
        dim3 grid(static_cast<unsigned>((p.tokens + block.x - 1) / block.x),
                  static_cast<unsigned>(p.top_k));
        extract_my_slot_kernel<<<grid, block, 0, p.stream>>>(
            p.d_topk_idx, g_scan.d_s2d_map, p.d_my_slot_at_dest,
            p.tokens, p.top_k, p.nRanks, p.myRank, p.num_local_experts);
    }
    SLOT_BENCH_CUDA_CHECK(cudaEventRecord(g_scan.ev_extract_done, p.stream));

    SLOT_BENCH_CUDA_CHECK(cudaEventSynchronize(g_scan.ev_extract_done));

    float local_ms = 0.f, ag_ms = 0.f, scan_ms = 0.f, extract_ms = 0.f;
    SLOT_BENCH_CUDA_CHECK(cudaEventElapsedTime(&local_ms,
        g_scan.ev_start,      g_scan.ev_local_done));
    SLOT_BENCH_CUDA_CHECK(cudaEventElapsedTime(&ag_ms,
        g_scan.ev_local_done, g_scan.ev_ag_done));
    SLOT_BENCH_CUDA_CHECK(cudaEventElapsedTime(&scan_ms,
        g_scan.ev_ag_done,    g_scan.ev_scan_done));
    SLOT_BENCH_CUDA_CHECK(cudaEventElapsedTime(&extract_ms,
        g_scan.ev_scan_done,  g_scan.ev_extract_done));

    SlotAllocTiming t;
    t.kernel_us         = (local_ms + scan_ms + extract_ms) * 1000.0f;
    t.collective_us     = ag_ms * 1000.0f;
    t.total_us          = t.kernel_us + t.collective_us;
    // allgather moves (nRanks-1)*local_bytes into each rank's recv buffer; we
    // report total bytes *received* to keep the metric rank-local-comparable.
    t.bytes_collective  = static_cast<size_t>(p.tokens) * experts_packed
                          * static_cast<size_t>(p.nRanks - 1);
    return t;
}

// ============================================================================
// atomic and cumsum: still stubs; real implementations land in commits 3/4
// ============================================================================

namespace {

void fill_output_minus_one(const SlotAllocParams& p) {
    SLOT_BENCH_CUDA_CHECK(cudaMemsetAsync(
        p.d_my_slot_at_dest, 0xFF,
        static_cast<size_t>(p.tokens) * p.top_k * sizeof(int32_t),
        p.stream));
    SLOT_BENCH_CUDA_CHECK(cudaStreamSynchronize(p.stream));
}

void announce_stub_once(const char* name, int myRank) {
    static int announced[3] = {0, 0, 0};
    int idx = (name[0] == 'a') ? 1 : (name[0] == 'c' ? 2 : 0);
    if (myRank == 0 && !announced[idx]) {
        announced[idx] = 1;
        fprintf(stderr, "[slot_bench] %s: stub, not implemented yet\n", name);
        fflush(stderr);
    }
}

}  // anonymous namespace

// ============================================================================
// atomic-based: dest-side remote atomicAdd on a fabric-memory counter matrix
// ============================================================================
//
// Each rank owns `per_rank_bytes` of fabric memory. The first nRanks*4 bytes
// are an int32 count_row where count_row[src] is the number of tokens src
// has pushed to ME so far this iteration. Source ranks increment count_row
// on the DEST rank's allocation via remote atomicAdd through the fabric
// mapping (peer_bufs[dest]), each returning the pre-add value as the
// absolute slot of that (token, topk) pair.
//
// Output slot semantics: "layout 2" - absolute slot in the dest recv buffer
// assigned by atomic arrival order. Different (non-deterministic across
// iterations wrt scan/cumsum), but still a valid absolute slot and still
// comparable on per-iter timing.

namespace {

struct AtomicImpl {
    bool        initialized = false;
    cudaEvent_t ev_start    = nullptr;
    cudaEvent_t ev_kernel_done = nullptr;
};

static AtomicImpl g_atomic;

static void ensure_atomic_state() {
    if (g_atomic.initialized) return;
    SLOT_BENCH_CUDA_CHECK(cudaEventCreate(&g_atomic.ev_start));
    SLOT_BENCH_CUDA_CHECK(cudaEventCreate(&g_atomic.ev_kernel_done));
    g_atomic.initialized = true;
}

// Kernel: for each (t, k) pair, compute dest rank from topk_idx and do a
// remote atomicAdd on dest's fabric-memory counter. my_slot_at_dest[t,k]
// receives the pre-add value (absolute slot in dest's recv buffer).
//
// peer_bufs[d] must point at rank d's fabric allocation (per-rank base VA).
// The first nRanks * int32 of each allocation is the count_row for that
// rank; count_row[my_rank] is the counter ME bumps when sending to peer d.
__global__ void atomic_slot_kernel(
    const int64_t* __restrict__ topk_idx,
    int32_t*       __restrict__ my_slot_at_dest,
    void* const*   __restrict__ peer_bufs,   // [nRanks], device array
    int tokens,
    int top_k,
    int nRanks,
    int myRank,
    int num_local_experts)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y;
    if (t >= tokens || k >= top_k) return;

    int64_t eid = topk_idx[static_cast<size_t>(t) * top_k + k];
    int32_t slot = -1;
    if (eid >= 0) {
        int dest = static_cast<int>(eid / static_cast<int64_t>(num_local_experts));
        int32_t* dest_counter = reinterpret_cast<int32_t*>(peer_bufs[dest]);
        // dest_counter[my_rank] is the per-(dest, src) counter inside dest's
        // block. atomicAdd returns the pre-increment value.
        slot = atomicAdd(&dest_counter[myRank], 1);
    }
    my_slot_at_dest[static_cast<size_t>(t) * top_k + k] = slot;
}

}  // anonymous namespace

SlotAllocTiming run_atomic_based(const SlotAllocParams& p) {
    ensure_atomic_state();

    if (!p.fabric.initialized || p.fabric.d_peer_bufs_dev == nullptr) {
        if (p.myRank == 0) {
            fprintf(stderr,
                "[slot_bench] atomic: fabric buffers not initialized; "
                "ep_bench must call init_slot_fabric_buffers before running "
                "--slot-alloc=atomic\n");
        }
        fill_output_minus_one(p);
        return {0.f, 0.f, 0.f, 0};
    }

    // Reset the count_row in MY block to zero. Each rank only resets its own
    // block; a host-side MPI_Barrier ensures all resets complete before any
    // rank starts incrementing peer counters.
    SLOT_BENCH_CUDA_CHECK(cudaMemsetAsync(
        p.fabric.local_buf, 0,
        static_cast<size_t>(p.nRanks) * sizeof(int32_t), p.stream));
    SLOT_BENCH_CUDA_CHECK(cudaStreamSynchronize(p.stream));
    SLOT_BENCH_MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    SLOT_BENCH_CUDA_CHECK(cudaEventRecord(g_atomic.ev_start, p.stream));
    {
        dim3 block(256);
        dim3 grid(static_cast<unsigned>((p.tokens + block.x - 1) / block.x),
                  static_cast<unsigned>(p.top_k));
        atomic_slot_kernel<<<grid, block, 0, p.stream>>>(
            p.d_topk_idx, p.d_my_slot_at_dest,
            p.fabric.d_peer_bufs_dev,
            p.tokens, p.top_k, p.nRanks, p.myRank, p.num_local_experts);
    }
    SLOT_BENCH_CUDA_CHECK(cudaEventRecord(g_atomic.ev_kernel_done, p.stream));
    SLOT_BENCH_CUDA_CHECK(cudaEventSynchronize(g_atomic.ev_kernel_done));

    // A host MPI_Barrier AFTER the kernel keeps iters isolated from each
    // other (so the next iter's memset does not race a peer still finishing
    // its atomicAdd). This barrier is NOT counted in kernel_us.
    SLOT_BENCH_MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    float kernel_ms = 0.f;
    SLOT_BENCH_CUDA_CHECK(cudaEventElapsedTime(&kernel_ms,
        g_atomic.ev_start, g_atomic.ev_kernel_done));

    SlotAllocTiming t;
    t.kernel_us        = kernel_ms * 1000.0f;
    t.collective_us    = 0.f;        // no NCCL/alltoall, atomicAdd overlaps with compute
    t.total_us         = t.kernel_us;
    t.bytes_collective = 0;          // no collective payload; remote atomics only
    return t;
}

SlotAllocTiming run_cumsum_based(const SlotAllocParams& p) {
    announce_stub_once("cumsum", p.myRank);
    fill_output_minus_one(p);
    return {0.f, 0.f, 0.f, 0};
}

SlotAllocTiming run(SlotAllocAlgorithm algo, const SlotAllocParams& p) {
    switch (algo) {
        case SLOT_SCAN:   return run_scan_based(p);
        case SLOT_ATOMIC: return run_atomic_based(p);
        case SLOT_CUMSUM: return run_cumsum_based(p);
    }
    if (p.myRank == 0) {
        fprintf(stderr, "[slot_bench] run: unknown algorithm %d\n", (int)algo);
    }
    fill_output_minus_one(p);
    return {0.f, 0.f, 0.f, 0};
}

// ============================================================================
// Fabric-memory symmetric allocation helper used by atomic and cumsum paths.
// ============================================================================
//
// Pattern mirrors nccl_ep.cc:init_hybridep_intranode_fabric but is stripped
// to what the slot-alloc microbench needs:
//   - one per-rank CUmemCreate(FABRIC) allocation of `per_rank_bytes`
//   - export + MPI_Allgather of CUmemFabricHandle (64 B each)
//   - each rank imports every peer's handle and maps it into a fresh VA
//   - d_peer_bufs_dev is a device array [nRanks] of void* (VAs in THIS
//     rank's address space) so kernels can do `peer_bufs[dest_rank]` to
//     find dest's buffer.
//
// Lifecycle assumption: ep_bench creates at most one fabric buffer per
// process (one --slot-only run per ep_bench invocation). The raw CU handles
// and VAs used for cleanup are parked in file-scope static vectors; init
// writes them, destroy reads and releases.

namespace {

static std::vector<CUmemGenericAllocationHandle>& slot_fabric_allocs_storage() {
    static std::vector<CUmemGenericAllocationHandle> v;
    return v;
}
static std::vector<CUdeviceptr>& slot_fabric_vas_storage() {
    static std::vector<CUdeviceptr> v;
    return v;
}

static size_t slot_fabric_round_up(size_t sz, size_t gran) {
    if (gran == 0) gran = 1;
    return (sz + gran - 1) & ~(gran - 1);
}

}  // anonymous namespace

void init_slot_fabric_buffers(
    SlotAllocFabricBuffers& buf,
    int nRanks,
    int myRank,
    int cuda_device_id,
    size_t per_rank_bytes_hint,
    cudaStream_t stream)
{
    if (buf.initialized) {
        destroy_slot_fabric_buffers(buf);
    }
    std::memset(&buf, 0, sizeof(buf));
    buf.nRanks = nRanks;

    CUmemAllocationProp prop = {};
    prop.type                 = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type        = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id          = cuda_device_id;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;

    size_t gran = 0;
    SLOT_BENCH_CU_CHECK(cuMemGetAllocationGranularity(
        &gran, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    size_t aligned = slot_fabric_round_up(per_rank_bytes_hint, gran);
    buf.per_rank_bytes = aligned;

    // Phase 1: allocate local fabric block and map into our own VA, zero it.
    CUmemGenericAllocationHandle local_alloc = 0;
    SLOT_BENCH_CU_CHECK(cuMemCreate(&local_alloc, aligned, &prop, 0));
    CUdeviceptr local_va = 0;
    SLOT_BENCH_CU_CHECK(cuMemAddressReserve(&local_va, aligned, gran, 0, 0));
    SLOT_BENCH_CU_CHECK(cuMemMap(local_va, aligned, 0, local_alloc, 0));
    CUmemAccessDesc access = {};
    access.location = prop.location;
    access.flags    = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    SLOT_BENCH_CU_CHECK(cuMemSetAccess(local_va, aligned, &access, 1));
    SLOT_BENCH_CUDA_CHECK(cudaMemsetAsync(
        reinterpret_cast<void*>(local_va), 0, aligned, stream));
    SLOT_BENCH_CUDA_CHECK(cudaStreamSynchronize(stream));

    // Phase 2: export local handle, allgather all ranks' handles via MPI.
    CUmemFabricHandle local_fh = {};
    SLOT_BENCH_CU_CHECK(cuMemExportToShareableHandle(
        &local_fh, local_alloc, CU_MEM_HANDLE_TYPE_FABRIC, 0));

    static_assert(sizeof(CUmemFabricHandle) == 64,
                  "CUmemFabricHandle must be 64 B");
    std::vector<CUmemFabricHandle> all_fh(nRanks);
    SLOT_BENCH_MPI_CHECK(MPI_Allgather(
        &local_fh,       sizeof(CUmemFabricHandle), MPI_BYTE,
        all_fh.data(),   sizeof(CUmemFabricHandle), MPI_BYTE,
        MPI_COMM_WORLD));

    // Phase 3: import each peer handle, map each to a fresh VA here.
    buf.h_peer_bufs_host =
        static_cast<void**>(std::malloc(sizeof(void*) * nRanks));
    std::memset(buf.h_peer_bufs_host, 0, sizeof(void*) * nRanks);

    auto& allocs = slot_fabric_allocs_storage();
    auto& vas    = slot_fabric_vas_storage();
    allocs.assign(nRanks, CUmemGenericAllocationHandle{0});
    vas   .assign(nRanks, CUdeviceptr{0});

    for (int r = 0; r < nRanks; r++) {
        if (r == myRank) {
            allocs[r] = local_alloc;
            vas   [r] = local_va;
            buf.h_peer_bufs_host[r] = reinterpret_cast<void*>(local_va);
        } else {
            CUmemGenericAllocationHandle h = 0;
            SLOT_BENCH_CU_CHECK(cuMemImportFromShareableHandle(
                &h, &all_fh[r], CU_MEM_HANDLE_TYPE_FABRIC));
            CUdeviceptr va = 0;
            SLOT_BENCH_CU_CHECK(cuMemAddressReserve(&va, aligned, gran, 0, 0));
            SLOT_BENCH_CU_CHECK(cuMemMap(va, aligned, 0, h, 0));
            SLOT_BENCH_CU_CHECK(cuMemSetAccess(va, aligned, &access, 1));
            allocs[r] = h;
            vas   [r] = va;
            buf.h_peer_bufs_host[r] = reinterpret_cast<void*>(va);
        }
    }
    buf.local_buf = buf.h_peer_bufs_host[myRank];

    // Copy peer-buf pointer array to device so kernels can index peer_bufs[d].
    SLOT_BENCH_CUDA_CHECK(cudaMalloc(&buf.d_peer_bufs_dev,
                                     sizeof(void*) * nRanks));
    SLOT_BENCH_CUDA_CHECK(cudaMemcpy(buf.d_peer_bufs_dev, buf.h_peer_bufs_host,
                                     sizeof(void*) * nRanks,
                                     cudaMemcpyHostToDevice));

    buf.initialized = true;
    (void)stream;
    MPI_Barrier(MPI_COMM_WORLD);
}

void destroy_slot_fabric_buffers(SlotAllocFabricBuffers& buf) {
    if (!buf.initialized) return;

    auto& allocs   = slot_fabric_allocs_storage();
    auto& vas      = slot_fabric_vas_storage();
    const size_t aligned = buf.per_rank_bytes;
    const int    nR      = buf.nRanks;

    for (int r = 0; r < nR; r++) {
        if (r < static_cast<int>(vas.size())) {
            CUdeviceptr va = vas[r];
            if (va) {
                cuMemUnmap(va, aligned);
                cuMemAddressFree(va, aligned);
            }
        }
        if (r < static_cast<int>(allocs.size())) {
            CUmemGenericAllocationHandle h = allocs[r];
            if (h) cuMemRelease(h);
        }
    }
    allocs.clear();
    vas.clear();

    if (buf.d_peer_bufs_dev)  cudaFree(buf.d_peer_bufs_dev);
    if (buf.h_peer_bufs_host) std::free(buf.h_peer_bufs_host);
    std::memset(&buf, 0, sizeof(buf));
}

}}  // namespace nccl_ep::slot_bench
