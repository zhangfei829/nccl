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
//     bytes 12..15               : float32 weight   (commit 5 fused combine
//                                                    reads this to scale the
//                                                    FFN output before the
//                                                    cross-rank atomic_add.
//                                                    Written by dispatch
//                                                    from topk_weights or
//                                                    1/num_topk fallback.)
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

// Launch the fused atomicAdd-slot + payload-push kernel.
//
// dispatch_path (commit C1):
//   "coop" : per-(token, k) cooperative 32-lane uint4 store. Legacy path
//            kept for A/B vs the TMA path so we can quantify TMA's actual
//            contribution. Block size = 32 * top_k threads (one warp per k).
//   "tma"  : per-(token, k) cp.async.bulk store from a per-block smem
//            staging buffer. Block size = 32 threads (one warp; lane 0
//            issues TMA serially across k). Aims to free SM ALU during
//            data transfer so per-SM throughput rises.
//   Selected by host via env var NCCL_EP_FULLMESH_DISPATCH_PATH={coop, tma},
//   default "tma".
//
// prof_dispatch (commit C1 instrumentation):
//   Optional device pointer to 4 x uint64 (cycle counters). When non-null,
//   block 0 lane 0 atomicAdd's the per-token clock64 deltas into:
//     [0] cooperative load us cycles
//     [1] tma_store_fence + per-k atomicAdd + meta store cycles
//     [2] sum of TMA store-issue cycles (coop: cooperative store cycles)
//     [3] tma_store_wait<0>() cycles (coop: 0)
//   Other blocks no-op; cuts contention while still giving a representative
//   timeline. host divides by (iters * num_tokens / num_sms) to get per-token
//   ns and prints. Pass nullptr to disable.
//
// SM budget (num_sms): caps grid.x at min(num_tokens, num_sms). When grid <
//   num_tokens the kernel runs a grid-stride loop so each block handles
//   multiple tokens. This keeps HT vs FM benchmarks apples-to-apples:
//   HT hardcodes grid=HYBRIDEP_MAX_NUM_SMS_PER_RANK=16, so FM should match
//   16 by default. Pass 0 to mean "no cap" but expect unfair benchmark
//   numbers vs HT.
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
//
// topk_weights (commit 5): if non-null, read at lane 0 of each (token, k)
//   warp and serialized into the meta.w fp32 slot at the dest. Lets the
//   fused combine kernel weight the FFN output by w[k] before the cross-
//   rank atomic_add without having to expose topk_weights as fabric memory
//   on every combine call. nullptr falls back to a uniform 1/top_k written
//   into meta.w; this matches HT's forward-combine semantics when callers
//   don't pass weights.
enum class DispatchPath { kCoop = 0, kTma = 1 };

void launch_dispatch_kernel(
    const void*       x,                      // [num_tokens, hidden_bytes] device src
    const int64_t*    topk_idx,               // [num_tokens, top_k] int64
    const float*      topk_weights,           // [num_tokens, top_k] fp32 or nullptr
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
    int               num_sms,                // SM cap; 0 = no cap (= num_tokens)
    DispatchPath      path,                   // kCoop or kTma
    unsigned long long* prof_dispatch,        // device [4] cycle counters or nullptr
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
    int               num_sms,                // SM cap (1D flatten of 2D grid)
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
//     (fallback: uniform 1/num_topk if topk_weights == nullptr)
//   Store acc -> combined_output[t] in bf16.
//
// Caller contract on combine_local_va zeroing (Phase 3 Commit A):
//   Slots (t, k) for k in [0, num_topk) MUST be zero before push_kernel runs,
//   because some (t, k) pairs have topk_idx == -1 (masked) and are never
//   written by any peer's push -- reduce_kernel will still read them when
//   iterating k = 0..num_topk-1, so stale residue from a past iteration
//   would corrupt the weighted sum.
//   Slots (t, k) for k in [num_topk, max_topk_for_combine) are never
//   written (push checks k_in_topk < max_topk_for_combine via the meta
//   field, but push_kernel ONLY runs for k < num_topk in practice because
//   dispatch_kernel's warp_id bound is top_k == num_topk) and never read
//   (reduce_kernel stops at k == num_topk), so they can hold arbitrary
//   garbage and the caller should NOT waste bandwidth zeroing them.
//   ncclEpCombine in nccl_ep.cc uses cudaMemset2DAsync to zero exactly
//   the first num_topk columns of every row.
void launch_combine_reduce_kernel(
    const void*       combine_local_va,       // src-local [num_tokens][max_topk][hidden]
    const float*      topk_weights,           // [num_tokens, num_topk] or nullptr => uniform
    void*             combined_output,        // [num_tokens, hidden] bf16 dst
    int               num_tokens,
    int               num_topk,
    int               max_topk_for_combine,
    int               hidden_bytes,
    int               num_sms,                // SM cap; 0 = no cap (= num_tokens)
    cudaStream_t      stream);

// ============================================================================
// Combine FUSED (Phase 3 commit B-fused): single-kernel dest -> src atomic_add
// ----------------------------------------------------------------------------
// Replaces push_kernel + reduce_kernel with one kernel that reads the dest's
// FFN output + meta, computes weight * ffn at the dest, and atomic_add's the
// weighted contribution directly into the src rank's combine_buf at column 0
// (i.e. [src_token_id, 0, hidden] under the same [num_tokens][max_topk][hidden]
// layout, treating column 0 as the running accumulator). The src rank then
// cudaMemcpy2DAsync's column 0 out to the user's combined_output.
//
// Why fused beats push+reduce on NV72:
//   - HBM passes drop from 2 to 1: data lands directly in the accumulator,
//     no separate staging-then-reduce read.
//   - 1 kernel instead of 2: removes the kernel-launch + stream-serialization
//     overhead between push and reduce.
//   - memset shrinks from num_topk columns to 1 column (column 0 only),
//     since fused doesn't need the per-k slots anymore. ~num_topk x smaller.
//
// Why we still need the column 0 zero-init:
//   atomicAdd is +=, not =. Reduce starts at whatever residue last iter left
//   in the accumulator. Without memset(col0), the result is corrupted.
//
// Preconditions:
//   - hidden_bytes % 16 == 0 AND hidden_bytes % 4 == 0
//     (uint4 loads for the source side, __nv_bfloat162 atomic_add for the
//     dest side; the latter wants 4-byte aligned pairs.)
//   - dispatch wrote weight into meta.w via the topk_weights param of
//     launch_dispatch_kernel; otherwise meta.w is 0 and the fused kernel
//     produces zero output.
//   - bf16 atomic_add on fabric memory requires sm_90+. GB300 is sm_103, OK.
//   - peer_combine_vas_dev[src_rank] points at the src rank's combine_buf
//     fabric region (same as push_kernel's target).
//   - The caller has cudaMemset2DAsync'd column 0 of combine_local_va to 0
//     and crossed a barrier so all peers see fresh zeros before atomic_add.
void launch_combine_fused_kernel(
    const void*       ffn_output,             // [nRanks*max_tokens, hidden] dense bf16
    const void*       recv_local_va,          // this rank's dispatch recv_buf (meta + payload)
    const int32_t*    counter_local_va,       // this rank's counter_row[nRanks]
    void* const*      peer_combine_vas_dev,   // device [nRanks] void*, src->combine VAs
    int               nRanks,
    int               myRank,
    int               max_tokens_per_rank,
    int               max_topk_for_combine,   // row stride in src.combine_buf
    int               hidden_bytes,
    int               bytes_per_entry,
    int               meta_bytes,
    int               num_sms,                // SM cap (1D flatten of 2D grid)
    cudaStream_t      stream);

}  // namespace fullmesh
}  // namespace nccl_ep
