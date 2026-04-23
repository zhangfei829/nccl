// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// FULLMESH (Phase 2) device entry points.
//
// Layout contract (mirrors ncclEpGroup::fullmesh_buffers in nccl_ep.cc):
//   recv_buf[dest][src][slot]  with entry size = meta_bytes + hidden_bytes
//     bytes 0..3                 : int32 src_rank   (= author of the push)
//     bytes 4..7                 : int32 src_token_id  (= t inside src rank)
//     bytes 8..11                : int32 k_in_topk  (commit 4 combine uses
//                                                    this to route the FFN
//                                                    output back to the
//                                                    src's combine_recv_buf
//                                                    at [src_token_id][k])
//     bytes 12..meta_bytes-1     : reserved padding (currently 0)
//     bytes meta_bytes..end      : hidden payload (dtype opaque)
//   counter_row[dest][src]       : int32 atomic counter, src atomicAdd(1) to
//                                  carve its next slot inside its per-src
//                                  block at dest. Also the combine kernel's
//                                  "how many slots from src do I have to
//                                  push back?" read-only source of truth.
//   combine_recv_buf[src_token_id][k_in_topk]
//                                : hidden-only payload (no meta), one slot
//                                  per (my own token, topk contribution).
//                                  dest rank writes a weighted or raw FFN
//                                  output here; src rank reduces across k
//                                  into combined_output.
//
// ALIGNMENT INVARIANT (enforced in nccl_ep.cc:init_fullmesh_intranode_fabric):
//   meta_bytes % 16 == 0  AND  hidden_bytes % 16 == 0
//   => bytes_per_entry    % 16 == 0
//   => entry+meta_bytes   is 16B-aligned for every (src, slot)
//   Required because the dispatch kernel streams the payload with 32-lane
//   cooperative uint4 stores. Breaking either invariant traps the kernel
//   with "misaligned address" on all slots where entry_idx is odd.
//
// The dispatch kernel is a single fused launch that, per (token, k) pair:
//   (1) computes dest rank from topk_idx / num_local_experts,
//   (2) atomicAdd(1) on peer_counter_vas[dest][myRank] to pull a slot index,
//   (3) writes (src_rank, src_token_id) meta + hidden payload into
//       peer_recv_vas[dest][(myRank*max_tokens + slot) * bytes_per_entry].
//
// launch_compact_to_output is a thin cudaMemcpy2DAsync wrapper that strips
// the 8-byte meta and projects the per-entry hidden payload into a dense
// [nRanks * max_tokens, hidden_bytes] user output tensor, reusing the existing
// HT output tensor shape so the CLI and BW accounting stay unchanged.
//
// Host-side synchronization between dispatch iterations (counter reset, peer
// readiness, kernel completion) is the caller's responsibility; this header
// intentionally exposes only kernel launches so ncclEpDispatch can weave in
// the Q1=C ncclBarrier calls at the right places.

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace nccl_ep {
namespace fullmesh {

// Launch the fused atomicAdd-slot + cooperative payload-push kernel.
//
// Preconditions enforced at the caller (nccl_ep.cc) level:
//   - hidden_bytes % 16 == 0 (required for uint4 store loop)
//   - top_k <= 32 (warps-per-block cap)
//   - max_tokens_per_rank big enough that no peer exhausts its slot block;
//     callers statically size it so total tokens expected at dest <= nRanks *
//     max_tokens_per_rank, matching the user-visible recv capacity.
//   - peer_recv_vas_dev and peer_counter_vas_dev are device-side void*[nRanks]
//     arrays populated by init_fullmesh_intranode_fabric with mapped peer VAs.
//   - This rank's counter_row was cudaMemset(0) on the same stream earlier
//     this iteration and a cross-rank barrier has confirmed all peers did
//     the same before any peer can atomicAdd this rank's row.
void launch_dispatch_kernel(
    const void*       x,                      // [num_tokens, hidden_bytes] device src
    const int64_t*    topk_idx,               // [num_tokens, top_k] int64
    void* const*      peer_recv_vas_dev,      // device [nRanks] void*
    void* const*      peer_counter_vas_dev,   // device [nRanks] void*
    int               num_tokens,
    int               top_k,
    int               num_local_experts,
    int               myRank,
    int               nRanks,
    int               max_tokens_per_rank,
    int               hidden_bytes,
    int               bytes_per_entry,
    int               meta_bytes,
    cudaStream_t      stream);

// Project this rank's recv_buf payload column into a dense user output tensor
// shaped [nRanks * max_tokens_per_rank, hidden_bytes], skipping the meta prefix
// at each entry. Rows beyond sum(counter_row) are undefined; callers currently
// treat the tensor as non-compacted which matches MoE semantics where
// downstream kernels read only valid slots.
//
// This is literally a cudaMemcpy2DAsync wrapper kept on the device side to
// mirror HT's convention of keeping tensor shape translation near the kernels.
void launch_compact_to_output(
    const void*       recv_local_va,          // src: this rank's recv_buf
    void*             output,                 // dst: [nRanks*max_tokens, hidden]
    int               nRanks,
    int               max_tokens_per_rank,
    int               hidden_bytes,
    int               bytes_per_entry,
    int               meta_bytes,
    cudaStream_t      stream);

// ============================================================================
// Combine kernels (Phase 2 commit 4)
// ----------------------------------------------------------------------------
// Combine is the reverse direction of dispatch: expert FFN outputs at dest
// ranks need to be aggregated back to the src rank and weighted-summed by
// topk_weights. FULLMESH's Q2=B choice means dest rank reads its own dispatch
// recv_buf meta to learn the (src, src_token_id, k) triple, then pushes the
// FFN output slot to the src rank's combine_recv_buf. After a cross-rank
// barrier the src rank's combine_reduce_kernel weighted-sums across the k
// dimension into combined_output.

// Kernel: dest-initiated combine push.
//
// Grid:  (nRanks, max_tokens_per_rank)   -- (src_in_dest_layout, slot)
// Block: (32,)                           -- one warp per slot cooperatively
//                                           streams the hidden payload
//
// Per block:
//   src   = blockIdx.x
//   slot  = blockIdx.y
//   counter_local_va[src] is read to decide whether (src, slot) holds a real
//   entry. If slot >= counter, the warp exits; this is the main way masked-
//   token slots (topk_idx == -1) and unused slots are skipped without adding
//   a separate slot map.
//
//   Meta at recv_local_va + (src * max_tokens + slot) * bytes_per_entry holds
//     (src_rank, src_token_id, k_in_topk). dest FFN output for the same i =
//     src * max_tokens + slot is at ffn_output + i * hidden_bytes.
//
// The kernel then writes ffn_output[i] to the src rank's combine_recv_buf at
// peer_combine_vas_dev[src_rank][(src_token_id * max_topk + k_in_topk) *
// hidden_bytes]. Pushed at bf16 precision; the src-side reduce kernel does
// the fp32 accumulation.
void launch_combine_push_kernel(
    const void*       ffn_output,             // [nRanks*max_tokens, hidden] dense bf16
    const void*       recv_local_va,          // this rank's dispatch recv_buf (for meta read)
    const int32_t*    counter_local_va,       // this rank's counter_row[nRanks]
    void* const*      peer_combine_vas_dev,   // device [nRanks] void*, src->combine VAs
    int               nRanks,
    int               myRank,
    int               max_tokens_per_rank,
    int               max_topk_for_combine,
    int               hidden_bytes,
    int               bytes_per_entry,
    int               meta_bytes,
    cudaStream_t      stream);

// Kernel: src-side weighted sum across k.
//
// Grid:  (num_tokens,)
// Block: (threads_per_block,)
//
// Per block:
//   t = blockIdx.x  (this rank's local token id, 0..num_tokens-1)
//   For each k in 0..num_topk-1:
//     load combine_local_va[t * max_topk + k] (hidden bf16)
//     accumulate weighted by topk_weights[t * num_topk + k] in fp32
//   Store acc -> combined_output[t] in bf16.
//
// Assumes the user cudaMemsetAsync'd combine_local_va to zero before combine
// push so slots for k beyond the actual routed count are 0 and contribute
// nothing to the sum. That memset is the caller's job.
void launch_combine_reduce_kernel(
    const void*       combine_local_va,       // src-local [num_tokens][max_topk][hidden]
    const float*      topk_weights,           // [num_tokens, num_topk]
    void*             combined_output,        // [num_tokens, hidden] bf16 dst
    int               num_tokens,
    int               num_topk,
    int               max_topk_for_combine,
    int               hidden_bytes,
    cudaStream_t      stream);

}  // namespace fullmesh
}  // namespace nccl_ep
