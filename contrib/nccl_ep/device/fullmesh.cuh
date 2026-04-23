// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// FULLMESH (Phase 2) device entry points.
//
// Layout contract (mirrors ncclEpGroup::fullmesh_buffers in nccl_ep.cc):
//   recv_buf[dest][src][slot]  with entry size = meta_bytes + hidden_bytes
//     bytes 0..7                 : (int32 src_rank, int32 src_token_id)
//     bytes 8..meta_bytes-1      : reserved padding (not read by combine)
//     bytes meta_bytes..end      : hidden payload (dtype opaque)
//   counter_row[dest][src]       : int32 atomic counter, src atomicAdd(1) to
//                                  carve its next slot inside its per-src
//                                  block at dest.
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
// shaped [nRanks * max_tokens_per_rank, hidden_bytes], skipping the 8-byte
// meta prefix at each entry. Rows beyond sum(counter_row) are undefined;
// callers currently treat the tensor as non-compacted which matches MoE
// semantics where downstream kernels read only valid slots.
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

}  // namespace fullmesh
}  // namespace nccl_ep
