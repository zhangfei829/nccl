/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */
// Phase 2 Stage 1: slot allocation micro-benchmark - stub skeleton.
//
// commit 1: skeleton only. Every entry point writes -1 to every output slot
// (so the downstream bench loop sees a valid, sentinel-filled buffer) and
// returns zero timings with a one-line "not implemented" notice from rank 0.
// Real implementations land in:
//   commit 2: run_scan_based
//   commit 3: run_atomic_based (+ fabric buffer allocator)
//   commit 4: run_cumsum_based

#include "slot_alloc_bench.cuh"

#include <cstdio>
#include <cuda_runtime.h>

namespace nccl_ep { namespace slot_bench {

namespace {

// Fill output with -1 so callers reliably see "unimplemented" output rather
// than uninitialized memory. Keeps bench framework debuggable even with stubs.
void fill_output_minus_one(const SlotAllocParams& p) {
    cudaMemsetAsync(p.d_my_slot_at_dest, 0xFF,
                    static_cast<size_t>(p.tokens) * p.top_k * sizeof(int32_t),
                    p.stream);
    cudaStreamSynchronize(p.stream);
}

void announce_stub_once(const char* name, int myRank) {
    static int announced[3] = {0, 0, 0};
    int idx = (name[0] == 's') ? 0 : (name[0] == 'a' ? 1 : 2);
    if (myRank == 0 && !announced[idx]) {
        announced[idx] = 1;
        fprintf(stderr, "[slot_bench] %s: stub, not implemented yet\n", name);
        fflush(stderr);
    }
}

}  // anonymous namespace

SlotAllocTiming run_scan_based(const SlotAllocParams& p) {
    announce_stub_once("scan",   p.myRank);
    fill_output_minus_one(p);
    return {0.f, 0.f, 0.f, 0};
}

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
    // Unreachable; keep a visible error rather than silently returning zero.
    if (p.myRank == 0) {
        fprintf(stderr, "[slot_bench] run: unknown algorithm %d\n", (int)algo);
    }
    fill_output_minus_one(p);
    return {0.f, 0.f, 0.f, 0};
}

}}  // namespace nccl_ep::slot_bench
