// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Implementation of FULLMESH (Phase 2) dispatch kernel + recv compaction.
// See fullmesh.cuh for the layout contract and the division of labour with
// the host-side ncclEpDispatch wrapper.

#include "fullmesh.cuh"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
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
// Name pattern: "fullmesh_<op>_kernel[_<phase>]" so CUPTI name-substring
// filters in ep_bench.cu (ktimer.get_avg_us("dispatch_kernel") and "combine
// _kernel") still match, and grep in sweep logs can pick FULLMESH kernels
// out by the fullmesh_ prefix.
__global__ void fullmesh_dispatch_kernel(
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

    // Lane-0 writes the 16-byte meta header as a single uint4 store (aligned):
    //   uint32[0]: src_rank    (this rank, push author)
    //   uint32[1]: src_token_id (= t in this block's grid index)
    //   uint32[2]: k_in_topk   (= warp_id; combine push_kernel routes the FFN
    //                           output back to the src's combine_recv_buf
    //                           at [src_token_id, k_in_topk])
    //   uint32[3]: weight_fp32 (commit 5 fused combine reads this. Either
    //                           topk_weights[t,k] when caller provided one,
    //                           or 1/top_k uniform fallback. Bit-cast through
    //                           __float_as_int so the meta uint4 store stays
    //                           one 16B transaction.)
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
    fullmesh_dispatch_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const uint4*>(x_void),
        topk_idx,
        topk_weights,
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
    int src  = blockIdx.x;
    int slot = blockIdx.y;
    if (src >= nRanks || slot >= max_tokens_per_rank) return;

    // Only (src, slot) pairs that a peer actually filled during dispatch are
    // valid. counter_local_va[src] is the total count src pushed into us.
    int filled = counter_local_va[src];
    if (slot >= filled) return;

    int lane = static_cast<int>(threadIdx.x & 31);

    // Read meta off this rank's own dispatch recv buf.
    const uint8_t* entry = recv_local_va +
                           (static_cast<size_t>(src) * max_tokens_per_rank + slot)
                           * static_cast<size_t>(bytes_per_entry);
    uint4 meta_vec = *reinterpret_cast<const uint4*>(entry);
    int src_rank     = static_cast<int>(meta_vec.x);
    int src_token_id = static_cast<int>(meta_vec.y);
    int k_in_topk    = static_cast<int>(meta_vec.z);

    // Defensive: malformed meta (src_rank mismatch) would route the push to
    // the wrong rank. In practice dispatch writes src_rank == src always, but
    // if a future change breaks that invariant the combine would silently
    // corrupt unrelated tokens.
    if (src_rank != src) return;
    if (k_in_topk < 0 || k_in_topk >= max_topk_for_combine) return;
    if (src_token_id < 0 || src_token_id >= max_tokens_per_rank) return;

    // Peer target: src_rank's combine_recv_buf at [src_token_id][k_in_topk].
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

// Src-side weighted sum across the k dimension.
// Grid (num_tokens,), block cooperative streams hidden_u4 entries.
__global__ void fullmesh_combine_kernel_reduce(
    const uint16_t* __restrict__ combine_local,    // [num_tokens*max_topk*hidden] bf16
    const float*    __restrict__ topk_weights,     // [num_tokens, num_topk]
    uint16_t*       __restrict__ combined_output,  // [num_tokens, hidden] bf16
    int num_tokens,
    int num_topk,
    int max_topk_for_combine,
    int hidden)
{
    int t = blockIdx.x;
    if (t >= num_tokens) return;

    int tid       = threadIdx.x;
    int nthreads  = blockDim.x;

    // Each thread covers a strided slice of the hidden dim. For each assigned
    // h, loop over k, accumulate weights[k] * combine[t,k,h] in fp32, then
    // write bf16 result to combined_output[t,h]. If topk_weights is null
    // (caller did not provide TOPK_WEIGHTS in combine inputs), fall back to
    // uniform 1/num_topk weighting -- this matches HT's forward-combine
    // semantics and keeps FULLMESH working with ep_bench's existing HT
    // tensor setup (num_combine_inputs=1, topk_weights absent) without a
    // per-algorithm branch in the benchmark.
    const float uniform_w = (num_topk > 0) ? (1.0f / static_cast<float>(num_topk)) : 0.f;
    for (int h = tid; h < hidden; h += nthreads) {
        float acc = 0.f;
        for (int k = 0; k < num_topk; ++k) {
            size_t slot_idx = ((static_cast<size_t>(t) * max_topk_for_combine) + k)
                            * hidden + h;
            // bf16 -> float via bit shift into upper 16 of fp32. This is the
            // standard bf16 load pattern used elsewhere in the project.
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
    cudaStream_t stream)
{
    if ((hidden_bytes & 15) != 0) {
        fprintf(stderr,
                "[FULLMESH] launch_combine_push_kernel: hidden_bytes=%d not "
                "16B aligned. Aborting.\n", hidden_bytes);
        return;
    }
    int hidden_u4 = hidden_bytes >> 4;

    dim3 grid(nRanks, max_tokens_per_rank);
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
    cudaStream_t stream)
{
    if (num_tokens <= 0) return;
    // hidden is number of bf16 elements, not bytes. combined_output is bf16
    // and combine_local is bf16 (dest pushed bf16 in combine_push).
    int hidden = hidden_bytes / 2;

    // 256 threads covers 7168-hidden in 28-element stripes. Power of two keeps
    // the common-case tail of the loop simple.
    dim3 grid(num_tokens);
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
    int src  = blockIdx.x;
    int slot = blockIdx.y;
    if (src >= nRanks || slot >= max_tokens_per_rank) return;

    // Skip slots a peer never filled this iter (dispatch counter is the
    // ground truth). Saves one block's worth of load+atomic for empty slots.
    int filled = counter_local_va[src];
    if (slot >= filled) return;

    int tid      = threadIdx.x;
    int nthreads = blockDim.x;

    // Read 16B meta in a single transaction. dispatch_kernel's lane-0 stored:
    //   x: src_rank, y: src_token_id, z: k_in_topk, w: weight_fp32 bits
    const uint8_t* entry = recv_local_va +
        (static_cast<size_t>(src) * max_tokens_per_rank + slot) *
        static_cast<size_t>(bytes_per_entry);
    uint4 meta_vec = *reinterpret_cast<const uint4*>(entry);
    int   src_rank     = static_cast<int>(meta_vec.x);
    int   src_token_id = static_cast<int>(meta_vec.y);
    // meta_vec.z = k_in_topk: not used in fused path; we sum directly into
    // col 0 instead of a per-k slot, so the dest rank doesn't care which k
    // this slot was for.
    float weight       = __int_as_float(static_cast<int>(meta_vec.w));

    if (src_rank != src) return;
    if (src_token_id < 0 || src_token_id >= max_tokens_per_rank) return;

    // Source: dest-local FFN output for this (src, slot) entry. The compaction
    // pass after dispatch projected the recv_buf into a dense
    // [nRanks * max_tokens, hidden] tensor, so the row index is exactly
    // src * max + slot.
    size_t i = static_cast<size_t>(src) * max_tokens_per_rank + slot;
    const __nv_bfloat162* src_payload = ffn_output_pair + i * hidden_pair;

    // Target: peer src_rank's combine_buf at [src_token_id, 0, 0..hidden).
    // Layout = [num_tokens][max_topk_for_combine][hidden_bf16]. A row in
    // bf162-pair units is max_topk_for_combine * hidden_pair pairs. We
    // accumulate into column 0 so all (src, slot) blocks targeting the same
    // src_token_id end up atomic-add'd into the same row[0:hidden] region.
    __nv_bfloat162* dst_base =
        reinterpret_cast<__nv_bfloat162*>(peer_combine_vas_dev[src_rank]);
    size_t row_pair_stride =
        static_cast<size_t>(max_topk_for_combine) * static_cast<size_t>(hidden_pair);
    __nv_bfloat162* dst = dst_base + static_cast<size_t>(src_token_id) * row_pair_stride;
    // col 0 starts at offset 0 inside the row; no further offset.

    // Pre-bake the weight into a bf162 pair so the inner loop is just
    // mul + atomicAdd, no float->bf16 conversions per element.
    __nv_bfloat162 wpair = __floats2bfloat162_rn(weight, weight);

    // Strided loop: 256 threads over hidden_pair=2240 (for hidden=7168) does
    // ~9 iterations/thread. Grid covers all (src, slot, src_token_id), so
    // multiple blocks may race on the same dst pair from different (src, slot)
    // tuples mapped to the same src_token_id; atomic_add on bf162 in fabric
    // memory serialises those writes correctly on sm_90+ (GB300 sm_103).
    for (int p = tid; p < hidden_pair; p += nthreads) {
        __nv_bfloat162 v        = src_payload[p];
        __nv_bfloat162 weighted = __hmul2(v, wpair);
        atomicAdd(dst + p, weighted);
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

    dim3 grid(nRanks, max_tokens_per_rank);
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
