// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Implementation of FULLMESH (Phase 2) dispatch kernel + recv compaction.
// See fullmesh.cuh for the layout contract and the division of labour with
// the host-side ncclEpDispatch wrapper.

#include "fullmesh.cuh"
#include "device_primitives.cuh"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>

namespace nccl_ep {
namespace fullmesh {

namespace {

// Phase 4 commit C1: TWO dispatch kernels (coop vs tma), env var selects.
// ============================================================================
//
// Both kernels do exactly the same work: per (token, k_in_topk) pair, atomicAdd
// a slot at the destination's counter row, write 16B meta to the destination's
// recv_buf entry, then write hidden_bytes payload to entry + meta_bytes. The
// only difference is HOW the payload gets to the peer:
//   coop : 32-lane cooperative uint4 store (one warp per k, all 32 lanes stride
//          hidden_u4 uint4 elements). SM ALU bound. Original Phase 2 path.
//   tma  : single cp.async.bulk store per k, from a per-block smem staging
//          buffer. Frees SM ALU during transfer (TMA is async hardware unit).
//          Cooperative load gmem->smem first, then per-k TMA store smem->peer.
//
// Both kernels share the same name substring "dispatch_kernel" so CUPTI
// substring matching in ep_bench.cu (ktimer.get_total_per_iter("dispatch_
// kernel", iters)) still works -- only one kernel ever launches per call so
// no double counting.
//
// Profile counters (prof_dispatch, optional, 4 x uint64 device buffer):
//   Block 0 lane 0 only (other blocks no-op to avoid contention). Each per-
//   token iter, lane 0 calls clock64() at 4 timestamps and atomicAdd's deltas:
//     [0] : load_cycles        -- cooperative gmem->smem load (tma path only,
//                                  coop path leaves this at 0 since coop has
//                                  no smem staging step)
//     [1] : prelude_cycles     -- per-k atomicAdd peer counter + 16B meta store
//     [2] : payload_cycles     -- per-k payload store path (coop: cooperative
//                                  uint4 store loop; tma: tma_store_1d issue)
//     [3] : tail_cycles        -- post-token sync (tma: tma_store_wait<0>;
//                                  coop: 0 since cooperative store finishes
//                                  before next iter naturally)
//   Host divides by iters*num_tokens/grid_x to get per-token-per-stage ns.

// Helper: clock64 only if prof_dispatch is non-null, block 0 lane 0.
__device__ __forceinline__ unsigned long long clock_if_prof(
    unsigned long long* prof_dispatch, int lane)
{
    if (prof_dispatch == nullptr) return 0ULL;
    if (blockIdx.x != 0) return 0ULL;
    if (lane != 0) return 0ULL;
    return clock64();
}

__device__ __forceinline__ void prof_add(
    unsigned long long* prof_dispatch, int slot, unsigned long long delta)
{
    if (prof_dispatch == nullptr) return;
    if (blockIdx.x != 0) return;
    atomicAdd(prof_dispatch + slot, delta);
}

__global__ void fullmesh_dispatch_kernel_coop(
    const uint4*     __restrict__ x,                 // [num_tokens, hidden_u4]
    const int64_t*   __restrict__ topk_idx,          // [num_tokens, top_k]
    const float*     __restrict__ topk_weights,      // [num_tokens, top_k] or nullptr
    void* const*     __restrict__ peer_recv_vas,    // [nRanks]
    void* const*     __restrict__ peer_counter_vas, // [nRanks]
    int num_tokens,
    int top_k,
    int num_local_experts,
    int myRank,
    int nRanks,
    int max_tokens_per_rank,
    int hidden_u4,
    int bytes_per_entry,
    int meta_bytes,
    unsigned long long* prof_dispatch)
{
    int warp_id = static_cast<int>(threadIdx.x >> 5);
    int lane    = static_cast<int>(threadIdx.x & 31);
    if (warp_id >= top_k) return;
    int k = warp_id;

    for (int t = blockIdx.x; t < num_tokens; t += gridDim.x) {
        unsigned long long ck0 = clock_if_prof(prof_dispatch, lane);

        int64_t eid = topk_idx[static_cast<size_t>(t) * top_k + k];
        if (eid < 0) continue;

        int dest = static_cast<int>(eid / num_local_experts);
        if (dest < 0 || dest >= nRanks) continue;

        int32_t* dest_counter = reinterpret_cast<int32_t*>(peer_counter_vas[dest]);
        int slot = 0;
        if (lane == 0) {
            slot = atomicAdd(&dest_counter[myRank], 1);
        }
        slot = __shfl_sync(0xFFFFFFFFu, slot, 0);
        if (slot >= max_tokens_per_rank) continue;

        uint8_t* dest_recv = reinterpret_cast<uint8_t*>(peer_recv_vas[dest]);
        size_t   entry_idx = static_cast<size_t>(myRank) * max_tokens_per_rank + slot;
        uint8_t* entry     = dest_recv + entry_idx * static_cast<size_t>(bytes_per_entry);

        if (lane == 0) {
            float w = (topk_weights != nullptr)
                    ? topk_weights[static_cast<size_t>(t) * top_k + k]
                    : ((top_k > 0) ? (1.0f / static_cast<float>(top_k)) : 0.f);
            uint4 meta_vec;
            meta_vec.x = static_cast<uint32_t>(myRank);
            meta_vec.y = static_cast<uint32_t>(t);
            meta_vec.z = static_cast<uint32_t>(k);
            meta_vec.w = static_cast<uint32_t>(__float_as_int(w));
            *reinterpret_cast<uint4*>(entry) = meta_vec;
        }

        unsigned long long ck1 = clock_if_prof(prof_dispatch, lane);

        uint4*       dst_payload = reinterpret_cast<uint4*>(entry + meta_bytes);
        const uint4* src_payload = x + static_cast<size_t>(t) * hidden_u4;
        for (int i = lane; i < hidden_u4; i += 32) {
            dst_payload[i] = src_payload[i];
        }

        unsigned long long ck2 = clock_if_prof(prof_dispatch, lane);

        // For coop, [0] load_cycles = 0 (no smem load), [3] tail_cycles = 0.
        prof_add(prof_dispatch, 1, ck1 - ck0);   // prelude (atomic + meta)
        prof_add(prof_dispatch, 2, ck2 - ck1);   // cooperative payload store
    }
}

// Phase 4 commit D1: multi-stage ring buffer with mbarrier producer/consumer.
// ----------------------------------------------------------------------------
// C3 baseline (kFm=2, warp-specialised + multi-token) measured 5.59 us/tok at
// NUM_SMS=16 EP=8 t=8192 = 499 GB/s (HT 599 GB/s, -17%). Stages run serially
// within a block:
//   load    cooperative gmem->smem    2.14 us/tok
//   payload lane-0 atomicAdd + meta + tma_store    2.59 us/tok
//   tail    tma_store_wait<0>    0.86 us/tok
//
// D1 splits the block into:
//   1 producer warp (warp 0):  TMA load gmem -> smem ring buffer
//   top_k consumer warps:      per-k atomicAdd + meta + TMA store
// Two-stage ring: while consumers process token[t] from smem[t&1], producer
// already TMA-loads token[t+1] into smem[(t+1)&1]. Consumer wall and producer
// wall run in parallel, total per-token wall = max(load, payload+tail).
//
// Synchronization via shared-memory mbarriers (HT-style):
//   produced_mbar[2] : signaled by producer when stage payload is in smem
//                      (TMA's complete_tx variant means hardware signals on
//                       cp.async.bulk completion -- no SM cycle spent waiting).
//   consumed_mbar[2] : signaled by consumers when stage smem is no longer
//                      needed (each of top_k consumer warps' lane 0 arrives,
//                      arrive_count = top_k).
//
// Block layout: warp 0 = producer, warps 1..top_k = consumers.
// Block size = 32 * (1 + top_k) threads (= 288 for top_k=8).
// Smem layout:
//   [0,             hidden_bytes)         stage 0 payload
//   [hidden_bytes,  2 * hidden_bytes)     stage 1 payload
//   [2 * hidden_bytes, +16)               produced_mbar[2]
//   [+16, +32)                            consumed_mbar[2]
// Total ~14 KB for hidden_bytes=7168.
//
// Expected at NUM_SMS=16 EP=8 t=8192:
//   per-token wall = max(load, payload+tail) = max(2.14, 3.45) = 3.45 us/tok
//   dispatch_kernel_us 1881 -> ~1170 us
//   dispatch BW 499 -> ~803 GB/s, beating HT 599 by ~34%
__global__ void fullmesh_dispatch_kernel_tma(
    const uint4*     __restrict__ x,                 // [num_tokens, hidden_u4]
    const int64_t*   __restrict__ topk_idx,          // [num_tokens, top_k]
    const float*     __restrict__ topk_weights,      // [num_tokens, top_k] or nullptr
    void* const*     __restrict__ peer_recv_vas,    // [nRanks]
    void* const*     __restrict__ peer_counter_vas, // [nRanks]
    int num_tokens,
    int top_k,
    int num_local_experts,
    int myRank,
    int nRanks,
    int max_tokens_per_rank,
    int hidden_u4,
    int hidden_bytes,
    int bytes_per_entry,
    int meta_bytes,
    unsigned long long* prof_dispatch)
{
    constexpr int kStages = 2;

    extern __shared__ uint8_t smem_raw[];
    uint4* smem_payload[kStages];
    smem_payload[0] = reinterpret_cast<uint4*>(smem_raw);
    smem_payload[1] = reinterpret_cast<uint4*>(smem_raw + hidden_bytes);
    uint64_t* mbar_base = reinterpret_cast<uint64_t*>(smem_raw + kStages * hidden_bytes);
    uint64_t* produced_mbar = mbar_base;            // [kStages]
    uint64_t* consumed_mbar = mbar_base + kStages;  // [kStages]

    const int tid_in_block = threadIdx.x;
    const int warp_id      = tid_in_block >> 5;
    const int lane         = tid_in_block & 31;
    const bool is_producer = (warp_id == 0);
    const int  consumer_k  = warp_id - 1;
    const bool is_consumer = (warp_id >= 1 && warp_id <= top_k);

    // --- (a) Init mbarriers once per block (producer warp lane 0). ---
    if (is_producer && lane == 0) {
        for (int s = 0; s < kStages; s++) {
            mbarrier_init(&produced_mbar[s], 1);          // 1 producer arrival per stage
            mbarrier_init(&consumed_mbar[s], top_k);      // top_k consumer arrivals per stage
        }
        fence_barrier_init();
    }
    __syncthreads();

    // Per-thread phase counters; bit s tracks the parity for stage s.
    uint32_t producer_phase = 0;
    uint32_t consumer_phase = 0;

    // Per-block-0-only profile timestamps. Producer wall == warp 0 lane 0;
    // consumer wall == warp 1 lane 0.
    unsigned long long p_start = 0, p_end = 0;
    unsigned long long c0 = 0, c1 = 0, c2 = 0;

    // --- (b) Pipeline: 1 token / iter, producer/consumer overlap across stages ---
    for (int t = blockIdx.x; t < num_tokens; t += gridDim.x) {
        const int stage = t & (kStages - 1);

        // ===== Producer side (warp 0 lane 0 only) =====
        if (is_producer && lane == 0) {
            p_start = clock_if_prof(prof_dispatch, 0);

            // Wait for previous use of this stage to drain (skip first kStages tokens).
            if (t >= kStages) {
                mbarrier_wait<true>(&consumed_mbar[stage], producer_phase, stage);
            }

            // Issue async TMA load gmem -> smem[stage]; mbarrier.expect_tx tells
            // the barrier how many bytes the TMA will deliver, and the hardware
            // arrives on the barrier when cp.async.bulk completes.
            mbarrier_arrive_and_expect_tx(&produced_mbar[stage], hidden_bytes);
            const uint4* src = x + static_cast<size_t>(t) * hidden_u4;
            tma_load_1d(smem_payload[stage], src, &produced_mbar[stage], hidden_bytes);

            p_end = clock_if_prof(prof_dispatch, 0);
        }

        // ===== Consumer side (warps 1..top_k, each lane 0 issues its k) =====
        if (is_consumer && lane == 0) {
            c0 = clock_if_prof(prof_dispatch, /*lane*/(consumer_k == 0) ? 0 : 1);

            // Wait token[t] payload to be in smem[stage].
            mbarrier_wait<true>(&produced_mbar[stage], consumer_phase, stage);

            c1 = clock_if_prof(prof_dispatch, /*lane*/(consumer_k == 0) ? 0 : 1);

            // Per-(t, k) work.
            const int k = consumer_k;
            int64_t eid = topk_idx[static_cast<size_t>(t) * top_k + k];
            if (eid >= 0) {
                int dest = static_cast<int>(eid / num_local_experts);
                if (dest >= 0 && dest < nRanks) {
                    int32_t* dest_counter = reinterpret_cast<int32_t*>(peer_counter_vas[dest]);
                    int slot = atomicAdd(&dest_counter[myRank], 1);
                    if (slot < max_tokens_per_rank) {
                        uint8_t* dest_recv = reinterpret_cast<uint8_t*>(peer_recv_vas[dest]);
                        size_t   entry_idx = static_cast<size_t>(myRank) * max_tokens_per_rank + slot;
                        uint8_t* entry     = dest_recv + entry_idx * static_cast<size_t>(bytes_per_entry);

                        float w = (topk_weights != nullptr)
                                ? topk_weights[static_cast<size_t>(t) * top_k + k]
                                : ((top_k > 0) ? (1.0f / static_cast<float>(top_k)) : 0.f);
                        uint4 meta_vec;
                        meta_vec.x = static_cast<uint32_t>(myRank);
                        meta_vec.y = static_cast<uint32_t>(t);
                        meta_vec.z = static_cast<uint32_t>(k);
                        meta_vec.w = static_cast<uint32_t>(__float_as_int(w));
                        *reinterpret_cast<uint4*>(entry) = meta_vec;

                        tma_store_1d(smem_payload[stage], entry + meta_bytes, hidden_bytes);
                        tma_store_wait<0>();   // wait this warp's commit_group
                    }
                }
            }

            c2 = clock_if_prof(prof_dispatch, /*lane*/(consumer_k == 0) ? 0 : 1);

            // Signal stage smem reusable for producer.
            mbarrier_arrive(&consumed_mbar[stage]);
        }

        // ===== Per-iter prof commit (block 0, canonical reporters). =====
        // Done outside the producer/consumer divergent paths so atomicAdd
        // happens uniformly per iter (no reordering hazards).
        if (blockIdx.x == 0 && tid_in_block == 0 && prof_dispatch != nullptr) {
            // [0] producer wait + arrive_expect + tma_load_1d issue (~ small)
            prof_add(prof_dispatch, 0, p_end - p_start);
        }
        if (blockIdx.x == 0 && tid_in_block == 32 && prof_dispatch != nullptr) {
            // [1] consumer wait produced_mbar (= 0 if pipeline is hiding load)
            // [2] consumer body (atomic + meta + tma_store + tma_wait)
            // [3] always 0 (slot reserved for symmetry with C3 print format)
            prof_add(prof_dispatch, 1, c1 - c0);
            prof_add(prof_dispatch, 2, c2 - c1);
            prof_add(prof_dispatch, 3, 0ULL);
        }
    }

    // --- (c) Drain: producer is done, but consumers may still be processing
    // the last kStages tokens. The kernel's natural exit waits for all warps,
    // so consumers will finish their tma_store_wait<0> + mbarrier_arrive
    // before kernel returns. No additional drain barrier needed. ---
}

}  // anonymous namespace

void launch_dispatch_kernel(
    const void*    x_void,
    const int64_t* topk_idx,
    const float*   topk_weights,
    void* const*   peer_recv_vas_dev,
    void* const*   peer_counter_vas_dev,
    int  num_tokens,
    int  top_k,
    int  num_local_experts,
    int  myRank,
    int  nRanks,
    int  max_tokens_per_rank,
    int  hidden_bytes,
    int  bytes_per_entry,
    int  meta_bytes,
    int  num_sms,
    DispatchPath path,
    unsigned long long* prof_dispatch,
    cudaStream_t stream)
{
    if (num_tokens <= 0) return;

    if ((hidden_bytes & 15) != 0) {
        fprintf(stderr,
                "[FULLMESH] launch_dispatch_kernel: hidden_bytes=%d is not 16B "
                "aligned; required for uint4 store loop. Aborting launch.\n",
                hidden_bytes);
        return;
    }
    if ((meta_bytes & 15) != 0) {
        fprintf(stderr,
                "[FULLMESH] launch_dispatch_kernel: meta_bytes=%d is not 16B "
                "aligned; uint4 stores at entry+meta_bytes would trap. "
                "Aborting launch.\n",
                meta_bytes);
        return;
    }
    if (top_k > 32) {
        fprintf(stderr,
                "[FULLMESH] launch_dispatch_kernel: top_k=%d > 32 unsupported "
                "(coop path uses one warp per k; tma path uses lane-0 serial).\n",
                top_k);
        return;
    }
    int hidden_u4 = hidden_bytes >> 4;

    // Grid is min(num_tokens, num_sms) to cap SM use; kernel runs a grid-
    // stride loop over tokens. num_sms == 0 means "no cap" -> match num_tokens.
    int grid_x = (num_sms <= 0 || num_sms >= num_tokens) ? num_tokens : num_sms;
    if (grid_x <= 0) return;

    if (path == DispatchPath::kCoop) {
        // Coop: 32 * top_k threads, one warp per k, no smem.
        int threads_per_block = 32 * (top_k > 0 ? top_k : 1);
        dim3 grid(grid_x);
        dim3 block(threads_per_block);
        fullmesh_dispatch_kernel_coop<<<grid, block, 0, stream>>>(
            reinterpret_cast<const uint4*>(x_void),
            topk_idx, topk_weights,
            peer_recv_vas_dev, peer_counter_vas_dev,
            num_tokens, top_k, num_local_experts,
            myRank, nRanks, max_tokens_per_rank,
            hidden_u4, bytes_per_entry, meta_bytes,
            prof_dispatch);
    } else {
        // D1 TMA multi-stage: 1 producer warp + top_k consumer warps.
        // Block = 32 * (1 + top_k) threads, 1 token per iter via grid stride,
        // load and payload overlap across two smem stages with mbarrier sync.
        constexpr int kStages = 2;
        int threads_per_block = 32 * (1 + (top_k > 0 ? top_k : 1));
        if (threads_per_block > 1024) {
            fprintf(stderr,
                    "[FULLMESH] launch_dispatch_kernel: threads_per_block=%d exceeds "
                    "1024-thread/block cap (top_k=%d). Aborting.\n",
                    threads_per_block, top_k);
            return;
        }
        // smem = kStages payload buffers + 2 * kStages 8B mbarriers.
        int smem_bytes = kStages * hidden_bytes + 2 * kStages * 8;
        int grid_x = (num_sms <= 0 || num_sms >= num_tokens) ? num_tokens : num_sms;
        if (grid_x <= 0) return;
        dim3 grid(grid_x);
        dim3 block(threads_per_block);
        fullmesh_dispatch_kernel_tma<<<grid, block, smem_bytes, stream>>>(
            reinterpret_cast<const uint4*>(x_void),
            topk_idx, topk_weights,
            peer_recv_vas_dev, peer_counter_vas_dev,
            num_tokens, top_k, num_local_experts,
            myRank, nRanks, max_tokens_per_rank,
            hidden_u4, hidden_bytes, bytes_per_entry, meta_bytes,
            prof_dispatch);
    }
}

void launch_compact_to_output(
    const void*  recv_local_va,
    void*        output,
    int          nRanks,
    int          max_tokens_per_rank,
    int          hidden_bytes,
    int          bytes_per_entry,
    int          meta_bytes,
    cudaStream_t stream)
{
    size_t num_recv_tokens_cap =
        static_cast<size_t>(nRanks) * static_cast<size_t>(max_tokens_per_rank);
    if (num_recv_tokens_cap == 0) return;

    const uint8_t* payload_start =
        reinterpret_cast<const uint8_t*>(recv_local_va) + meta_bytes;

    cudaMemcpy2DAsync(
        output,
        static_cast<size_t>(hidden_bytes),       // dst pitch (dense)
        payload_start,
        static_cast<size_t>(bytes_per_entry),    // src pitch (meta + hidden)
        static_cast<size_t>(hidden_bytes),       // width
        num_recv_tokens_cap,                     // height
        cudaMemcpyDeviceToDevice,
        stream);
}

// ============================================================================
// Combine (Phase 2 commit 4)
// ============================================================================

namespace {

// Dest-initiated combine push. One block per (src, slot); warp cooperatively
// streams the hidden payload of the FFN output to the src rank's combine_recv
// _buf at [src_token_id, k_in_topk]. See fullmesh.cuh for full contract.
__global__ void fullmesh_combine_kernel_push(
    const uint4*    __restrict__ ffn_output,            // [nRanks*max_tokens, hidden_u4]
    const uint8_t*  __restrict__ recv_local_va,         // this rank's dispatch recv_buf
    const int32_t*  __restrict__ counter_local_va,      // this rank's counter_row[nRanks]
    void* const*    __restrict__ peer_combine_vas_dev,  // [nRanks]
    int nRanks,
    int myRank,
    int max_tokens_per_rank,
    int max_topk_for_combine,
    int hidden_u4,
    int bytes_per_entry,
    int meta_bytes)
{
    // Grid-stride loop over flattened (src, slot) space. Caller may launch
    // grid < nRanks * max_tokens_per_rank (SM cap) so each block handles
    // multiple (src, slot) entries.
    int total = nRanks * max_tokens_per_rank;
    int lane  = static_cast<int>(threadIdx.x & 31);

    for (int idx = blockIdx.x; idx < total; idx += gridDim.x) {
        int src  = idx / max_tokens_per_rank;
        int slot = idx - src * max_tokens_per_rank;

        // Only (src, slot) pairs that a peer actually filled during dispatch
        // are valid. counter_local_va[src] is the total count src pushed into us.
        int filled = counter_local_va[src];
        if (slot >= filled) continue;

        // Read meta off this rank's own dispatch recv buf.
        const uint8_t* entry = recv_local_va +
                               (static_cast<size_t>(src) * max_tokens_per_rank + slot)
                               * static_cast<size_t>(bytes_per_entry);
        uint4 meta_vec = *reinterpret_cast<const uint4*>(entry);
        int src_rank     = static_cast<int>(meta_vec.x);
        int src_token_id = static_cast<int>(meta_vec.y);
        int k_in_topk    = static_cast<int>(meta_vec.z);

        // Defensive: malformed meta (src_rank mismatch) would route the push
        // to the wrong rank. In practice dispatch writes src_rank == src, but
        // if a future change breaks that invariant the combine would silently
        // corrupt unrelated tokens.
        if (src_rank != src) continue;
        if (k_in_topk < 0 || k_in_topk >= max_topk_for_combine) continue;
        if (src_token_id < 0 || src_token_id >= max_tokens_per_rank) continue;

        // Peer target: src_rank's combine_recv_buf at [src_token_id, k_in_topk].
        uint8_t* peer_combine = reinterpret_cast<uint8_t*>(peer_combine_vas_dev[src_rank]);
        size_t   combine_slot = static_cast<size_t>(src_token_id) * max_topk_for_combine
                              + static_cast<size_t>(k_in_topk);
        uint4*   dst_payload  = reinterpret_cast<uint4*>(
                                  peer_combine + combine_slot * static_cast<size_t>(hidden_u4) * 16);

        // Source: this rank's FFN output at the dense row (src * max + slot).
        size_t i = static_cast<size_t>(src) * max_tokens_per_rank + slot;
        const uint4* src_payload = ffn_output + i * hidden_u4;

        for (int j = lane; j < hidden_u4; j += 32) {
            dst_payload[j] = src_payload[j];
        }
    }
}

// Src-side weighted sum across the k dimension.
// Grid <= num_tokens (SM cap), kernel runs grid-stride loop over tokens.
__global__ void fullmesh_combine_kernel_reduce(
    const uint16_t* __restrict__ combine_local,    // [num_tokens*max_topk*hidden] bf16
    const float*    __restrict__ topk_weights,     // [num_tokens, num_topk]
    uint16_t*       __restrict__ combined_output,  // [num_tokens, hidden] bf16
    int num_tokens,
    int num_topk,
    int max_topk_for_combine,
    int hidden)
{
    int tid       = threadIdx.x;
    int nthreads  = blockDim.x;

    const float uniform_w = (num_topk > 0) ? (1.0f / static_cast<float>(num_topk)) : 0.f;

    for (int t = blockIdx.x; t < num_tokens; t += gridDim.x) {
        // Each thread covers a strided slice of the hidden dim. For each
        // assigned h, loop over k, accumulate weights[k] * combine[t,k,h] in
        // fp32, then write bf16 result to combined_output[t,h]. If
        // topk_weights is null (caller did not provide TOPK_WEIGHTS in combine
        // inputs), fall back to uniform 1/num_topk weighting -- this matches
        // HT's forward-combine semantics and keeps FULLMESH working with
        // ep_bench's existing HT tensor setup (num_combine_inputs=1,
        // topk_weights absent) without a per-algorithm branch in the benchmark.
        for (int h = tid; h < hidden; h += nthreads) {
            float acc = 0.f;
            for (int k = 0; k < num_topk; ++k) {
                size_t slot_idx = ((static_cast<size_t>(t) * max_topk_for_combine) + k)
                                * hidden + h;
                // bf16 -> float via bit shift into upper 16 of fp32. Standard
                // bf16 load pattern used elsewhere in the project.
                uint16_t bf   = combine_local[slot_idx];
                uint32_t bits = static_cast<uint32_t>(bf) << 16;
                float    v    = __int_as_float(static_cast<int>(bits));
                float    w    = (topk_weights != nullptr)
                              ? topk_weights[static_cast<size_t>(t) * num_topk + k]
                              : uniform_w;
                acc += w * v;
            }
            // bf16 round-to-nearest-even via "add 0x7fff + lsb" trick.
            uint32_t acc_bits = static_cast<uint32_t>(__float_as_int(acc));
            uint32_t lsb      = (acc_bits >> 16) & 1u;
            uint32_t bias     = 0x7fffu + lsb;
            uint16_t out      = static_cast<uint16_t>((acc_bits + bias) >> 16);
            combined_output[static_cast<size_t>(t) * hidden + h] = out;
        }
    }
}

}  // anonymous namespace

void launch_combine_push_kernel(
    const void*  ffn_output_void,
    const void*  recv_local_va,
    const int32_t* counter_local_va,
    void* const* peer_combine_vas_dev,
    int nRanks,
    int myRank,
    int max_tokens_per_rank,
    int max_topk_for_combine,
    int hidden_bytes,
    int bytes_per_entry,
    int meta_bytes,
    int num_sms,
    cudaStream_t stream)
{
    if ((hidden_bytes & 15) != 0) {
        fprintf(stderr,
                "[FULLMESH] launch_combine_push_kernel: hidden_bytes=%d not "
                "16B aligned. Aborting.\n", hidden_bytes);
        return;
    }
    int hidden_u4 = hidden_bytes >> 4;

    // 1D grid over flattened (src, slot) space, capped at num_sms. Kernel
    // does grid-stride loop. Default (num_sms == 0) uses full
    // nRanks * max_tokens_per_rank grid (legacy behaviour).
    int total = nRanks * max_tokens_per_rank;
    int grid_x = (num_sms <= 0 || num_sms >= total) ? total : num_sms;
    if (grid_x <= 0) return;

    dim3 grid(grid_x);
    dim3 block(32);
    fullmesh_combine_kernel_push<<<grid, block, 0, stream>>>(
        reinterpret_cast<const uint4*>(ffn_output_void),
        reinterpret_cast<const uint8_t*>(recv_local_va),
        counter_local_va,
        peer_combine_vas_dev,
        nRanks, myRank, max_tokens_per_rank, max_topk_for_combine,
        hidden_u4, bytes_per_entry, meta_bytes);
}

void launch_combine_reduce_kernel(
    const void*  combine_local_va_void,
    const float* topk_weights,
    void*        combined_output_void,
    int num_tokens,
    int num_topk,
    int max_topk_for_combine,
    int hidden_bytes,
    int num_sms,
    cudaStream_t stream)
{
    if (num_tokens <= 0) return;
    // hidden is number of bf16 elements, not bytes. combined_output is bf16
    // and combine_local is bf16 (dest pushed bf16 in combine_push).
    int hidden = hidden_bytes / 2;

    // 256 threads covers 7168-hidden in 28-element stripes. Power of two keeps
    // the common-case tail of the loop simple.
    int grid_x = (num_sms <= 0 || num_sms >= num_tokens) ? num_tokens : num_sms;
    dim3 grid(grid_x);
    dim3 block(256);
    fullmesh_combine_kernel_reduce<<<grid, block, 0, stream>>>(
        reinterpret_cast<const uint16_t*>(combine_local_va_void),
        topk_weights,
        reinterpret_cast<uint16_t*>(combined_output_void),
        num_tokens, num_topk, max_topk_for_combine, hidden);
}

// ============================================================================
// Combine FUSED (Phase 3 commit B-fused): single-kernel push+reduce
// ----------------------------------------------------------------------------
// One block per (src, slot) at the dest. Reads the FFN output for that slot
// and the meta (which carries weight_fp32 written by dispatch), then
// atomic_add's weight * ffn into the src rank's combine_buf at column 0.
// After all peers finish, src rank reads column 0 directly as the combined
// weighted sum -- no separate reduce kernel.

namespace {

__global__ void fullmesh_combine_kernel_fused(
    const __nv_bfloat162* __restrict__ ffn_output_pair, // [nRanks*max_tokens, hidden_pair]
    const uint8_t*        __restrict__ recv_local_va,   // dispatch recv buf (for meta)
    const int32_t*        __restrict__ counter_local_va,// counter_row[nRanks]
    void* const*          __restrict__ peer_combine_vas_dev,  // [nRanks]
    int nRanks,
    int myRank,
    int max_tokens_per_rank,
    int max_topk_for_combine,
    int hidden_pair,                                    // hidden_bf16 / 2
    int bytes_per_entry,
    int meta_bytes)
{
    // Grid-stride loop over flattened (src, slot) space.
    int total    = nRanks * max_tokens_per_rank;
    int tid      = threadIdx.x;
    int nthreads = blockDim.x;

    for (int idx = blockIdx.x; idx < total; idx += gridDim.x) {
        int src  = idx / max_tokens_per_rank;
        int slot = idx - src * max_tokens_per_rank;

        // Skip slots a peer never filled this iter (dispatch counter is the
        // ground truth). Saves one iteration's worth of load+atomic for empty
        // slots.
        int filled = counter_local_va[src];
        if (slot >= filled) continue;

        // Read 16B meta in a single transaction. dispatch_kernel's lane-0 stored:
        //   x: src_rank, y: src_token_id, z: k_in_topk, w: weight_fp32 bits
        const uint8_t* entry = recv_local_va +
            (static_cast<size_t>(src) * max_tokens_per_rank + slot) *
            static_cast<size_t>(bytes_per_entry);
        uint4 meta_vec = *reinterpret_cast<const uint4*>(entry);
        int   src_rank     = static_cast<int>(meta_vec.x);
        int   src_token_id = static_cast<int>(meta_vec.y);
        // meta_vec.z = k_in_topk: not used in fused path; we sum directly into
        // col 0 instead of a per-k slot.
        float weight       = __int_as_float(static_cast<int>(meta_vec.w));

        if (src_rank != src) continue;
        if (src_token_id < 0 || src_token_id >= max_tokens_per_rank) continue;

        // Source: dest-local FFN output for this (src, slot) entry. The
        // compaction pass after dispatch projected the recv_buf into a dense
        // [nRanks * max_tokens, hidden] tensor, so the row index is exactly
        // src * max + slot.
        size_t i = static_cast<size_t>(src) * max_tokens_per_rank + slot;
        const __nv_bfloat162* src_payload = ffn_output_pair + i * hidden_pair;

        // Target: peer src_rank's combine_buf at [src_token_id, 0, 0..hidden).
        // Layout = [num_tokens][max_topk_for_combine][hidden_bf16]. A row in
        // bf162-pair units is max_topk_for_combine * hidden_pair pairs. We
        // accumulate into column 0 so all (src, slot) blocks targeting the
        // same src_token_id end up atomic-add'd into the same row[0:hidden]
        // region.
        __nv_bfloat162* dst_base =
            reinterpret_cast<__nv_bfloat162*>(peer_combine_vas_dev[src_rank]);
        size_t row_pair_stride =
            static_cast<size_t>(max_topk_for_combine) * static_cast<size_t>(hidden_pair);
        __nv_bfloat162* dst = dst_base + static_cast<size_t>(src_token_id) * row_pair_stride;
        // col 0 starts at offset 0 inside the row; no further offset.

        // Pre-bake the weight into a bf162 pair so the inner loop is just
        // mul + atomicAdd, no float->bf16 conversions per element.
        __nv_bfloat162 wpair = __floats2bfloat162_rn(weight, weight);

        // Strided loop: 256 threads over hidden_pair=2240 (for hidden=7168)
        // does ~9 iterations/thread. Grid covers all (src, slot, src_token_id),
        // so multiple iterations may race on the same dst pair from different
        // (src, slot) tuples mapped to the same src_token_id; atomic_add on
        // bf162 in fabric memory serialises those writes correctly on sm_90+
        // (GB300 sm_103).
        for (int p = tid; p < hidden_pair; p += nthreads) {
            __nv_bfloat162 v        = src_payload[p];
            __nv_bfloat162 weighted = __hmul2(v, wpair);
            atomicAdd(dst + p, weighted);
        }
    }
}

}  // anonymous namespace

void launch_combine_fused_kernel(
    const void*  ffn_output_void,
    const void*  recv_local_va,
    const int32_t* counter_local_va,
    void* const* peer_combine_vas_dev,
    int nRanks,
    int myRank,
    int max_tokens_per_rank,
    int max_topk_for_combine,
    int hidden_bytes,
    int bytes_per_entry,
    int meta_bytes,
    int num_sms,
    cudaStream_t stream)
{
    if (nRanks <= 0 || max_tokens_per_rank <= 0) return;
    // bf162 atomic_add requires hidden_bytes % 4 == 0. hidden_bytes % 16 == 0
    // already implied by dispatch's uint4 invariant, so this is belt-and-
    // braces.
    if ((hidden_bytes & 3) != 0) {
        fprintf(stderr,
                "[FULLMESH] launch_combine_fused_kernel: hidden_bytes=%d not "
                "4B aligned (bf162 atomic_add requires 4B). Aborting.\n",
                hidden_bytes);
        return;
    }
    int hidden_bf16 = hidden_bytes >> 1;
    int hidden_pair = hidden_bf16 >> 1;

    int total = nRanks * max_tokens_per_rank;
    int grid_x = (num_sms <= 0 || num_sms >= total) ? total : num_sms;
    if (grid_x <= 0) return;

    dim3 grid(grid_x);
    dim3 block(256);
    fullmesh_combine_kernel_fused<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat162*>(ffn_output_void),
        reinterpret_cast<const uint8_t*>(recv_local_va),
        counter_local_va,
        peer_combine_vas_dev,
        nRanks, myRank, max_tokens_per_rank, max_topk_for_combine,
        hidden_pair, bytes_per_entry, meta_bytes);
}

}  // namespace fullmesh
}  // namespace nccl_ep
