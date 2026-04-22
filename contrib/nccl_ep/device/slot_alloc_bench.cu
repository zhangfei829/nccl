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
#include <cuda_runtime.h>
#include <cub/device/device_scan.cuh>

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

SlotAllocTiming run_atomic_based(const SlotAllocParams& p) {
    announce_stub_once("atomic", p.myRank);
    fill_output_minus_one(p);
    return {0.f, 0.f, 0.f, 0};
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

}}  // namespace nccl_ep::slot_bench
