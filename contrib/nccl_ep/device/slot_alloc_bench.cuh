/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */
// Phase 2 Stage 1: slot allocation micro-benchmark.
//
// Compares three algorithms for computing, from the per-rank topk_idx,
// where each (token, k) lands in the dest rank's recv buffer:
//   - scan   : reuse Phase 0 metadata_preprocessing scan kernel
//   - atomic : remote atomicAdd on a fabric-memory counter matrix
//   - cumsum : hand-written fabric-memory alltoall + local cumsum
//
// Output semantics: per-src local index (not absolute slot; base offset
// compute is a separate Phase 2 problem). Layout: [tokens, top_k] int32,
// with -1 for masked (topk_idx < 0) entries.
//
// This header and the matching .cu live ONLY inside ep_bench's build graph;
// they are not part of libnccl_ep.so.

#pragma once

#include <cstdint>
#include <cstddef>
#include <cuda_runtime.h>
#include <nccl.h>

namespace nccl_ep { namespace slot_bench {

enum SlotAllocAlgorithm {
    SLOT_SCAN   = 0,
    SLOT_ATOMIC = 1,
    SLOT_CUMSUM = 2,
};

// Symmetric fabric-memory buffers, allocated once per ep_bench run and
// reused across all benchmark iterations. Populated by the host-side
// initializer in ep_bench.cu (commit 3 lands the actual allocator).
struct SlotAllocFabricBuffers {
    int32_t*  count_matrix;      // [nRanks * nRanks] int32, row = this rank's send_count to each dest
    int32_t*  offset_matrix;     // [nRanks * nRanks] int32, row = this rank's (as dest) recv_offset for each src
    uint32_t* barrier_flags;     // [nRanks] uint32, one flag per rank for fabric-memory barrier
    size_t    aligned_size_bytes;
    bool      initialized;
};

struct SlotAllocParams {
    // Device pointers (owned by caller)
    const int64_t* d_topk_idx;         // [tokens, top_k] int64, per-rank
    int32_t*       d_my_slot_at_dest;  // [tokens, top_k] int32 output (-1 for masked)

    // Shape
    int tokens;
    int top_k;
    int num_experts;
    int num_local_experts;
    int nRanks;
    int myRank;

    // Shared fabric-memory buffers (only atomic / cumsum use them)
    SlotAllocFabricBuffers fabric;

    // NCCL + stream
    ncclComm_t   comm;
    cudaStream_t stream;
};

struct SlotAllocTiming {
    float  kernel_us;         // device-side kernel time (from CUPTI or cudaEvent)
    float  collective_us;     // allgather/alltoall time (0 if algo has none)
    float  total_us;          // wall clock, input-ready -> output-ready
    size_t bytes_collective;  // total bytes moved in collective phase
};

// Algorithm entry points. Each is independently timed.
// Later commits fill in the real implementation; commit 1 returns zero
// timings and writes -1 to every output slot so that the bench framework
// can be exercised end-to-end.
SlotAllocTiming run_scan_based   (const SlotAllocParams& p);
SlotAllocTiming run_atomic_based (const SlotAllocParams& p);
SlotAllocTiming run_cumsum_based (const SlotAllocParams& p);

// Thin dispatcher used by ep_bench.cu.
SlotAllocTiming run(SlotAllocAlgorithm algo, const SlotAllocParams& p);

// CSV column names appended to ep_parse.py output; order matches the struct.
inline const char* timing_csv_header() {
    return "slot_alloc_algo,slot_kernel_us,slot_collective_us,slot_total_us,slot_bytes_collective";
}

inline const char* algo_name(SlotAllocAlgorithm a) {
    switch (a) {
        case SLOT_SCAN:   return "scan";
        case SLOT_ATOMIC: return "atomic";
        case SLOT_CUMSUM: return "cumsum";
    }
    return "?";
}

}}  // namespace nccl_ep::slot_bench
