// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Implementation of FULLMESH (Phase 2) dispatch kernel + recv compaction.
// See fullmesh.cuh for the layout contract and the division of labour with
// the host-side ncclEpDispatch wrapper.

#include "fullmesh.cuh"

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

namespace nccl_ep {
namespace fullmesh {

namespace {

// Per-(token, k) fused slot allocation + payload push.
//
// Grid:  (num_tokens,)
// Block: (32 * top_k,)   one warp per k in the top-k slice.
//
// Lane-level layout inside a warp for a given (t, k):
//   lane 0   : atomicAdd on peer dest_counter[myRank] to pull a slot id
//              then broadcasts slot to the other 31 lanes via __shfl_sync.
//   lane 0   : stores the 8-byte (src_rank, src_token_id) meta at entry[0..7].
//   all 32   : cooperatively stream the hidden_u4 uint4 payload into
//              entry[meta_bytes..meta_bytes+hidden_bytes).
//
// Warps beyond top_k (if the block is padded up to a multiple of 32 in any
// future tuning) early-exit. Tokens whose topk_idx==-1 (masked / dropped)
// also early-exit without consuming a slot, which matches HT's is-routed
// semantics in hybrid_ep.cuh.
__global__ void dispatch_kernel(
    const uint4*     __restrict__ x,                 // [num_tokens, hidden_u4]
    const int64_t*   __restrict__ topk_idx,          // [num_tokens, top_k]
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
    int meta_bytes)
{
    int t = blockIdx.x;
    if (t >= num_tokens) return;

    int warp_id = static_cast<int>(threadIdx.x >> 5);
    int lane    = static_cast<int>(threadIdx.x & 31);
    if (warp_id >= top_k) return;

    int k = warp_id;
    int64_t eid = topk_idx[static_cast<size_t>(t) * top_k + k];
    if (eid < 0) return;

    int dest = static_cast<int>(eid / num_local_experts);
    // Defensive bound: a malformed topk_idx that indexes past num_experts
    // would otherwise corrupt peer memory at a neighbour rank.
    if (dest < 0 || dest >= nRanks) return;

    int32_t* dest_counter = reinterpret_cast<int32_t*>(peer_counter_vas[dest]);

    int slot = 0;
    if (lane == 0) {
        slot = atomicAdd(&dest_counter[myRank], 1);
    }
    slot = __shfl_sync(0xFFFFFFFFu, slot, 0);

    // Overflow guard: if more than max_tokens_per_rank entries from this src
    // arrive at this dest, the slot index would run past the per-src block
    // and corrupt the neighbouring src's slots. Static sizing makes this
    // impossible when num_tokens <= max_tokens_per_rank (ep_bench config
    // enforces this), but keep the guard so a future caller error is visible
    // as missing data instead of silent corruption.
    if (slot >= max_tokens_per_rank) return;

    uint8_t* dest_recv = reinterpret_cast<uint8_t*>(peer_recv_vas[dest]);
    size_t   entry_idx = static_cast<size_t>(myRank) * max_tokens_per_rank + slot;
    uint8_t* entry     = dest_recv + entry_idx * static_cast<size_t>(bytes_per_entry);

    if (lane == 0) {
        uint32_t lo = static_cast<uint32_t>(myRank);
        uint32_t hi = static_cast<uint32_t>(t);
        uint64_t meta = static_cast<uint64_t>(lo) |
                        (static_cast<uint64_t>(hi) << 32);
        *reinterpret_cast<uint64_t*>(entry) = meta;
    }

    uint4*       dst_payload = reinterpret_cast<uint4*>(entry + meta_bytes);
    const uint4* src_payload = x + static_cast<size_t>(t) * hidden_u4;
    for (int i = lane; i < hidden_u4; i += 32) {
        dst_payload[i] = src_payload[i];
    }
}

}  // anonymous namespace

void launch_dispatch_kernel(
    const void*    x_void,
    const int64_t* topk_idx,
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
    // Same 16B alignment requirement applies to meta_bytes: dst_payload is
    // entry + meta_bytes, so if meta_bytes % 16 != 0 the uint4 stores trap.
    // init_fullmesh_intranode_fabric pads meta to 16 specifically for this;
    // this assert catches any future caller that forgets.
    if ((meta_bytes & 15) != 0) {
        fprintf(stderr,
                "[FULLMESH] launch_dispatch_kernel: meta_bytes=%d is not 16B "
                "aligned; uint4 stores at entry+meta_bytes would trap. "
                "Aborting launch.\n",
                meta_bytes);
        return;
    }
    int hidden_u4 = hidden_bytes >> 4;

    // top_k bounded by 32 warps/block keeps us under the 1024-thread block
    // limit and keeps __shfl_sync's 32-lane semantics intact. Larger top_k
    // would require a different intra-block layout (multiple passes, or
    // per-block loop over k), so fail loudly instead of clamping silently.
    if (top_k > 32) {
        fprintf(stderr,
                "[FULLMESH] launch_dispatch_kernel: top_k=%d > 32 unsupported "
                "in commit-3 layout; would overflow block thread budget.\n",
                top_k);
        return;
    }

    int warps_per_block = (top_k > 0) ? top_k : 1;
    int threads_per_block = 32 * warps_per_block;

    dim3 grid(num_tokens);
    dim3 block(threads_per_block);
    dispatch_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const uint4*>(x_void),
        topk_idx,
        peer_recv_vas_dev,
        peer_counter_vas_dev,
        num_tokens, top_k, num_local_experts,
        myRank, nRanks, max_tokens_per_rank,
        hidden_u4, bytes_per_entry, meta_bytes);
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

}  // namespace fullmesh
}  // namespace nccl_ep
