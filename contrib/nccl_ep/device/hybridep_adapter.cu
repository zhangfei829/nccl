/*
 * Portions of this file are adapted from DeepEP (https://github.com/deepseek-ai/DeepEP).
 * Copyright (c) 2025 DeepSeek. Licensed under the MIT License.
 * SPDX-License-Identifier: MIT
 */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

#include "nccl_device.h"
#include "hybridep_adapter.cuh"
#include "hybridep_configs.cuh"
#include "hybrid_ep.cuh"
#include "device_primitives.cuh"
#include "include/common.hpp"
#include <cstdint>
#include <cstdio>

namespace nccl_ep {
namespace hybridep {

namespace {

__global__ void dispatch_output_tma_copy_bf16_kernel(
    const uint16_t* __restrict__ src,
    uint16_t* __restrict__ dst,
    int num_tokens,
    int hidden,
    int copy_tokens_per_iter)
{
    extern __shared__ uint8_t smem_raw[];
    uint16_t* smem = reinterpret_cast<uint16_t*>(smem_raw);
    const int bytes_per_token = hidden * static_cast<int>(sizeof(uint16_t));
    const int bytes_per_iter = copy_tokens_per_iter * bytes_per_token;
    uint64_t* mbar = reinterpret_cast<uint64_t*>(smem_raw + bytes_per_iter);

    if (threadIdx.x == 0) {
        mbarrier_init(mbar, 1);
        fence_barrier_init();
    }
    __syncthreads();

    uint32_t phase = 0;
    const int num_chunks = (num_tokens + copy_tokens_per_iter - 1) / copy_tokens_per_iter;
    for (int chunk = blockIdx.x; chunk < num_chunks; chunk += gridDim.x) {
        const int token_start = chunk * copy_tokens_per_iter;
        const int remaining_tokens = num_tokens - token_start;
        const int cur_tokens = remaining_tokens < copy_tokens_per_iter ? remaining_tokens : copy_tokens_per_iter;
        const int copy_bytes = cur_tokens * bytes_per_token;
        const size_t elem_offset = static_cast<size_t>(token_start) * hidden;

        if (threadIdx.x == 0) {
            cuda::ptx::cp_async_bulk(
                cuda::ptx::space_shared,
                cuda::ptx::space_global,
                reinterpret_cast<void*>(smem),
                reinterpret_cast<const void*>(src + elem_offset),
                static_cast<uint32_t>(copy_bytes),
                mbar);
            mbarrier_arrive_and_expect_tx(mbar, copy_bytes);
            mbarrier_wait(mbar, phase);

            tma_store_1d(
                reinterpret_cast<const void*>(smem),
                reinterpret_cast<void*>(dst + elem_offset),
                copy_bytes);
            tma_store_wait<0>();
        }
        __syncthreads();
    }
}

} // namespace

void launch_dispatch_output_tma_copy_bf16(
    const void* src,
    void* dst,
    int num_tokens,
    int hidden,
    int num_blocks,
    cudaStream_t stream)
{
    if (src == nullptr || dst == nullptr || num_tokens <= 0 || hidden <= 0) return;
    const int bytes_per_token = hidden * static_cast<int>(sizeof(uint16_t));
    if ((bytes_per_token & 15) != 0) {
        fprintf(stderr,
                "[HT-TMA-COPY] bytes_per_token=%d is not 16B aligned; skip TMA copy\n",
                bytes_per_token);
        return;
    }
    // Keep dynamic shared memory conservative (<48KiB) to avoid launch-attr
    // dependencies. 2 tokens @ hidden=7168 bf16 = 28KiB; plus 8B mbarrier.
    const int copy_tokens_per_iter = 2;
    const int smem_bytes = copy_tokens_per_iter * bytes_per_token + static_cast<int>(sizeof(uint64_t));
    int blocks = num_blocks > 0 ? num_blocks : 16;
    if (blocks > num_tokens) blocks = num_tokens;
    if (blocks <= 0) return;

    dispatch_output_tma_copy_bf16_kernel<<<blocks, 32, smem_bytes, stream>>>(
        reinterpret_cast<const uint16_t*>(src),
        reinterpret_cast<uint16_t*>(dst),
        num_tokens,
        hidden,
        copy_tokens_per_iter);
}

// ============================================================================
// Kernel: Convert sparse topk_idx to dense routing map
// ============================================================================
__global__ void convert_topk_to_routing_map_kernel(
    const int64_t* __restrict__ topk_idx,    // [num_tokens, num_topk]
    uint8_t* __restrict__ routing_bitmap,     // [num_tokens, num_experts_packed]
    int num_tokens,
    int num_topk,
    int num_experts_packed                    // = ceil(num_experts / 8)
) {
    int token = blockIdx.x * blockDim.x + threadIdx.x;
    if (token >= num_tokens) return;

    // Buffer is pre-zeroed by per-iteration memset; just OR in set bits.
    // Each thread exclusively owns its row -- no atomics needed.
    uint8_t* row = routing_bitmap + token * num_experts_packed;
    for (int k = 0; k < num_topk; k++) {
        int expert = static_cast<int>(topk_idx[token * num_topk + k]);
        if (expert >= 0) {
            row[expert / 8] |= (1u << (expert % 8));
        }
    }
}

// ============================================================================
// Convert topk to bitmap routing map
// ============================================================================
void convert_topk_to_routing_map(
    const int64_t* topk_idx,
    uint8_t* routing_bitmap,
    int num_tokens,
    int num_topk,
    int num_experts_packed,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (num_tokens + block_size - 1) / block_size;

    convert_topk_to_routing_map_kernel<<<grid_size, block_size, 0, stream>>>(
        topk_idx, routing_bitmap, num_tokens, num_topk, num_experts_packed);
}

__global__ void build_dispatch_copy_expected_counts_kernel(
    const bool* __restrict__ local_expert_routing_map,
    uint32_t* __restrict__ dispatch_copy_expected,
    int max_recv_tokens,
    int experts_per_rank,
    int chunk_tokens)
{
    int token = blockIdx.x * blockDim.x + threadIdx.x;
    if (token >= max_recv_tokens) return;
    bool active = false;
    const bool* row = local_expert_routing_map + static_cast<size_t>(token) * experts_per_rank;
    for (int e = 0; e < experts_per_rank; e++) {
        active |= row[e];
    }
    if (active) {
        atomicAdd(dispatch_copy_expected + token / chunk_tokens, 1u);
    }
}

void build_dispatch_copy_expected_counts(
    const bool* local_expert_routing_map,
    uint32_t* dispatch_copy_expected,
    int max_recv_tokens,
    int experts_per_rank,
    int chunk_tokens,
    cudaStream_t stream)
{
    if (local_expert_routing_map == nullptr || dispatch_copy_expected == nullptr ||
        max_recv_tokens <= 0 || experts_per_rank <= 0 || chunk_tokens <= 0) {
        return;
    }
    int block = 256;
    int grid = (max_recv_tokens + block - 1) / block;
    build_dispatch_copy_expected_counts_kernel<<<grid, block, 0, stream>>>(
        local_expert_routing_map, dispatch_copy_expected,
        max_recv_tokens, experts_per_rank, chunk_tokens);
}

// ============================================================================
// Kernel: Convert sparse topk_weights to dense prob
// ============================================================================
__global__ void sparse_to_dense_prob_kernel(
    const int64_t* __restrict__ topk_idx,      // [num_tokens, topk]
    const float* __restrict__ topk_weights,    // [num_tokens, topk]
    float* __restrict__ dense_prob,            // [num_tokens, num_experts]
    int num_tokens,
    int num_topk,
    int num_experts
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int token = tid / num_topk;
    int k = tid % num_topk;

    if (token >= num_tokens) return;

    int64_t expert = topk_idx[token * num_topk + k];
    float weight = topk_weights[token * num_topk + k];

    // Scatter weight to the correct expert position
    if (expert >= 0 && expert < num_experts) {
        dense_prob[token * num_experts + expert] = weight;
    }
}

// ============================================================================
// Convert sparse to dense prob
// ============================================================================
void sparse_to_dense_prob(
    const int64_t* topk_idx,
    const float* topk_weights,
    float* dense_prob,
    int num_tokens,
    int num_topk,
    int num_experts,
    cudaStream_t stream
) {
    int total_elements = num_tokens * num_topk;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;

    sparse_to_dense_prob_kernel<<<grid_size, block_size, 0, stream>>>(
        topk_idx, topk_weights, dense_prob, num_tokens, num_topk, num_experts);
}

// ============================================================================
// Kernel: Convert sparse topk_weights to dense prob for combine input
// ============================================================================
// Used for combine backward pass. Uses local_expert_routing_map to determine
// which experts each token is routed to, matching the order from dispatch output.
// Each thread handles one token.
__global__ void sparse_to_dense_prob_combine_kernel(
    const float* __restrict__ topk_weights,           // [num_tokens, topk]
    const bool* __restrict__ local_expert_routing_map, // [num_tokens, experts_per_rank]
    float* __restrict__ dense_prob,                   // [num_tokens, experts_per_node]
    int num_tokens,
    int num_topk,
    int experts_per_rank,
    int experts_per_node,
    int local_rank
) {
    int token = blockIdx.x * blockDim.x + threadIdx.x;
    if (token >= num_tokens) return;

    // Scan local experts in order (matches dense_to_sparse_prob output order)
    int k_in = 0;
    for (int e = 0; e < experts_per_rank && k_in < num_topk; e++) {
        if (local_expert_routing_map[token * experts_per_rank + e]) {
            // This expert is active for this token - take next weight from sparse input
            float weight = topk_weights[token * num_topk + k_in];

            // Place at correct position in dense matrix
            // Local expert e on local_rank maps to: local_rank * experts_per_rank + e
            int dense_idx = token * experts_per_node + local_rank * experts_per_rank + e;
            dense_prob[dense_idx] = weight;

            k_in++;
        }
    }
}

// ============================================================================
// Convert sparse to dense prob for combine input
// ============================================================================
void sparse_to_dense_prob_combine(
    const float* topk_weights,
    const bool* local_expert_routing_map,
    float* dense_prob,
    int num_tokens,
    int num_topk,
    int experts_per_rank,
    int experts_per_node,
    int local_rank,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (num_tokens + block_size - 1) / block_size;

    sparse_to_dense_prob_combine_kernel<<<grid_size, block_size, 0, stream>>>(
        topk_weights, local_expert_routing_map, dense_prob, num_tokens, num_topk,
        experts_per_rank, experts_per_node, local_rank);
}

// ============================================================================
// Kernel: Convert dense prob output to sparse format
// ============================================================================
// Each thread handles one token, scans for non-zero experts
__global__ void dense_to_sparse_prob_kernel(
    const float* __restrict__ dense_prob,              // [num_recv_tokens, experts_per_node]
    const bool* __restrict__ local_expert_routing_map, // [num_recv_tokens, experts_per_rank]
    float* __restrict__ recv_topk_weights,             // [num_recv_tokens, topk]
    int64_t* __restrict__ recv_topk_idx,               // [num_recv_tokens, topk]
    int num_recv_tokens,
    int topk,
    int experts_per_rank,
    int experts_per_node,
    int local_rank
) {
    int token = blockIdx.x * blockDim.x + threadIdx.x;
    if (token >= num_recv_tokens) return;

    int k_out = 0;

    // Scan local experts (the ones this rank is responsible for)
    for (int e = 0; e < experts_per_rank && k_out < topk; e++) {
        // Check if this token is routed to expert e
        if (local_expert_routing_map[token * experts_per_rank + e]) {
            // Use local expert id for NCCL API compatibility (expects 0-based local indices)
            int64_t local_expert = static_cast<int64_t>(e);

            // Get weight from dense output (indexed by local expert within node)
            // dense_prob layout: [token, experts_per_node] where experts_per_node = experts_per_rank * ranks_per_node
            // Local rank's experts are at offset: local_rank * experts_per_rank
            int dense_idx = token * experts_per_node + local_rank * experts_per_rank + e;
            float weight = dense_prob[dense_idx];

            // Write both outputs
            recv_topk_idx[token * topk + k_out] = local_expert;
            recv_topk_weights[token * topk + k_out] = weight;
            k_out++;
        }
    }

    // Zero-fill remaining topk slots if fewer than topk experts found
    for (; k_out < topk; k_out++) {
        recv_topk_idx[token * topk + k_out] = -1;  // Invalid expert marker
        recv_topk_weights[token * topk + k_out] = 0.0f;
    }
}

// ============================================================================
// Kernel: Convert dense prob output to sparse format for combine output
// ============================================================================
// Used for combine backward pass. Converts kernel's dense output to sparse format
// with GLOBAL expert indices (matching original dispatch input format).
// Each thread handles one token.
__global__ void dense_to_sparse_prob_combine_kernel(
    const float* __restrict__ dense_prob,         // [num_tokens, num_experts]
    const uint8_t* __restrict__ routing_bitmap,   // [num_tokens, ceil(num_experts / 8)]
    float* __restrict__ combined_topk_weights,    // [num_tokens, topk]
    int64_t* __restrict__ combined_topk_idx,      // [num_tokens, topk] (optional, can be nullptr)
    int num_tokens,
    int topk,
    int num_experts
) {
    int token = blockIdx.x * blockDim.x + threadIdx.x;
    if (token >= num_tokens) return;

    int packed_cols = (num_experts + 7) / 8;
    int k_out = 0;

    // Scan all experts in order (matches original dispatch input order)
    for (int e = 0; e < num_experts && k_out < topk; e++) {
        if ((routing_bitmap[token * packed_cols + e / 8] >> (e % 8)) & 1) {
            // This expert is active for this token
            float weight = dense_prob[token * num_experts + e];

            combined_topk_weights[token * topk + k_out] = weight;
            if (combined_topk_idx != nullptr) {
                combined_topk_idx[token * topk + k_out] = static_cast<int64_t>(e);  // GLOBAL expert ID
            }
            k_out++;
        }
    }

    // Zero-fill remaining topk slots
    for (; k_out < topk; k_out++) {
        combined_topk_weights[token * topk + k_out] = 0.0f;
        if (combined_topk_idx != nullptr) {
            combined_topk_idx[token * topk + k_out] = -1;
        }
    }
}

// ============================================================================
// Convert dense prob output to sparse format for combine output
// ============================================================================
void dense_to_sparse_prob_combine(
    const float* dense_prob,
    const uint8_t* routing_bitmap,
    float* combined_topk_weights,
    int64_t* combined_topk_idx,
    int num_tokens,
    int topk,
    int num_experts,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (num_tokens + block_size - 1) / block_size;

    dense_to_sparse_prob_combine_kernel<<<grid_size, block_size, 0, stream>>>(
        dense_prob, routing_bitmap, combined_topk_weights, combined_topk_idx,
        num_tokens, topk, num_experts);
}


// ============================================================================
// Dense to sparse prob
// ============================================================================
void dense_to_sparse_prob(
    const float* dense_prob,
    const bool* local_expert_routing_map,
    float* recv_topk_weights,
    int64_t* recv_topk_idx,
    int num_recv_tokens,
    int topk,
    int experts_per_rank,
    int experts_per_node,
    int local_rank,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (num_recv_tokens + block_size - 1) / block_size;

    dense_to_sparse_prob_kernel<<<grid_size, block_size, 0, stream>>>(
        dense_prob, local_expert_routing_map, recv_topk_weights, recv_topk_idx,
        num_recv_tokens, topk, experts_per_rank, experts_per_node, local_rank);
}

// ============================================================================
// Call metadata preprocessing
// ============================================================================
void call_metadata_preprocessing(
    const uint8_t* global_routing_map,
    int32_t* sparse_to_dense_map,
    bool* rdma_to_attn_map,
    bool* attn_to_rdma_map,
    uint32_t* token_rank_mask,
    int32_t* num_tokens_for_experts,
    bool* local_expert_routing_map,
    int32_t* per_expert_token_counts,
    void* scan_tmp,
    int node_rank,
    int local_rank,
    int num_tokens_per_rank,
    int hidden_dim,
    int num_nodes,
    int num_ranks_per_node,
    int experts_per_rank,
    cudaStream_t stream
) {
    if (per_expert_token_counts != nullptr) {
        // Fused scan path accumulates counts with atomicAdd.
        CUDA_CHECK(cudaMemsetAsync(
            per_expert_token_counts, 0, experts_per_rank * sizeof(int32_t), stream));
    }

    // [NV72-ADAPT] Phase 0: confirm which scan kernel specialization is reached, and
    // whether the current (nodes, ranks_per_node) combo fits the <=32 warp-scan limit.
    // Printed at most once per process from (node_rank=0, local_rank=0).
    if (node_rank == 0 && local_rank == 0) {
        static int s_nv72_adapt_scan_printed = 0;
        if (s_nv72_adapt_scan_printed == 0) {
            s_nv72_adapt_scan_printed = 1;
            fprintf(stderr,
                    "[NV72-ADAPT] scan: LSA_TEAM_SIZE=%d NUM_LSA_TEAMS=%d "
                    "(warp_limit_ok=%s will_assert=%s)\n",
                    num_ranks_per_node, num_nodes,
                    num_ranks_per_node <= 32 ? "yes" : "NO",
                    num_ranks_per_node <= 32 ? "no" : "YES");
            fflush(stderr);
        }
    }

    // MNNVL configurations (> 32 GPUs per LSA domain) are not yet supported: the scan
    // kernel uses warp-reduction (LSA_TEAM_SIZE <= 32) and HYBRIDEP_SWITCH_LSA_TEAM_SIZE
    // is only instantiated up to 32. Extend both when adding MNNVL support.
    EP_HOST_ASSERT(num_ranks_per_node <= 32 && "metadata_preprocessing: LSA team size > 32 not yet supported (MNNVL)");

    HYBRIDEP_SWITCH_NUM_LSA_TEAMS(num_nodes, {
        HYBRIDEP_SWITCH_LSA_TEAM_SIZE(num_ranks_per_node, {
            using HybridEPType = ::hybrid_ep::hybrid_ep<MAX_SUPPORTED_TOKENS_PER_RANK, NUM_LSA_TEAMS, LSA_TEAM_SIZE>;
            HybridEPType::template metadata_preprocessing<
                HYBRIDEP_NUM_THREADS_PER_BLOCK_PREPROCESSING, HYBRIDEP_NUM_BLOCKS_PREPROCESSING>(
                global_routing_map,
                reinterpret_cast<::hybrid_ep::tmp_state_t*>(scan_tmp),
                sparse_to_dense_map,
                rdma_to_attn_map,
                attn_to_rdma_map,
                token_rank_mask,
                num_tokens_for_experts,
                local_expert_routing_map,
                per_expert_token_counts,
                node_rank,
                local_rank,
                num_tokens_per_rank,
                num_ranks_per_node,
                experts_per_rank,
                stream
            );
        });
    });
}

size_t get_preprocessing_scan_tmp_size(int num_ranks_per_node) {
    return HYBRIDEP_NUM_BLOCKS_PREPROCESSING * num_ranks_per_node * sizeof(::hybrid_ep::tmp_state_t);
}

// ============================================================================
// Dispatch wrapper implementation
// ============================================================================

// Helper to populate dispatch_kernel_param_t from DispatchParams
template<typename TOKEN_DATA_TYPE, int LSA_TEAM_SIZE>
::hybrid_ep::dispatch_kernel_param_t<TOKEN_DATA_TYPE, LSA_TEAM_SIZE>
build_dispatch_param(const DispatchParams& params) {
    ::hybrid_ep::dispatch_kernel_param_t<TOKEN_DATA_TYPE, LSA_TEAM_SIZE> kp{};
    // Model configuration
    kp.hidden_dim = params.hidden_dim;
    kp.experts_per_rank = params.experts_per_rank;
    kp.num_of_ranks_per_node = params.num_ranks_per_node;
    // User input buffers
    kp.attn_input_token = reinterpret_cast<const TOKEN_DATA_TYPE*>(params.attn_input_token);
    kp.attn_input_prob = params.attn_input_prob;
    kp.attn_input_token_scaling_factor = params.attn_input_scaling_factor;

    // Copy IPC buffer pointers from HOST arrays into embedded param struct arrays.
    // This allows fast __grid_constant__ access in the kernel (vs slow global memory indirection).
    for (int i = 0; i < params.num_ranks_per_node; i++) {
        kp.expert_output_token[i] =
            reinterpret_cast<TOKEN_DATA_TYPE*>(params.expert_output_token_ptrs[i]);
        kp.expert_output_prob[i] = params.expert_output_prob_ptrs ?
            params.expert_output_prob_ptrs[i] : nullptr;
        kp.expert_output_scaling_factor[i] = params.expert_output_scaling_factor_ptrs ?
            params.expert_output_scaling_factor_ptrs[i] : nullptr;
        kp.dispatch_copy_ready[i] = params.dispatch_copy_ready_ptrs ?
            params.dispatch_copy_ready_ptrs[i] : nullptr;
    }
    kp.dispatch_copy_expected = params.dispatch_copy_expected;
    kp.user_output_token = reinterpret_cast<TOKEN_DATA_TYPE*>(params.user_output_token);
    kp.user_output_num_tokens = params.user_output_num_tokens;
    kp.dispatch_copy_chunk_tokens = params.dispatch_copy_chunk_tokens;

    // Metadata and sync flags
    kp.rdma_to_attn_map = params.rdma_to_attn_map;
    kp.attn_to_rdma_map = params.attn_to_rdma_map;
    kp.sparse_to_dense_map = params.sparse_to_dense_map;
    kp.expected_rdma_flag_value = params.expected_rdma_flag_value;
    kp.expected_intra_node_flag_value = params.expected_intra_node_flag_value;
    kp.rdma_inter_node_group_flags = params.rdma_inter_node_group_flags;
    kp.intra_node_write_completion_flags = params.intra_node_write_completion_flags;
    kp.dispatch_grid_barrier_counter = params.dispatch_grid_barrier_counter;
    kp.num_tokens_for_experts = params.num_tokens_for_experts;

    // Runtime config
    kp.local_rank = params.local_rank;
    kp.node_rank = params.node_rank;
    kp.num_of_tokens_per_rank = params.num_tokens_per_rank;

    // Pass device communicators and windows
    kp.dcomms = params.dcomms;
    kp.nccl_window = params.nccl_window;
    kp.num_gin_comms = params.num_gin_comms;
    kp.num_ctx_per_comm = params.num_ctx_per_comm;
    kp.gin_base_ptr = params.gin_base_ptr;
    kp.signals_base = params.signals_base;
    // Use offsets relative to gin_base_ptr
    kp.mr_info = {
               .attn_input_token_offset = params.mr_info.attn_input_token_offset,
               .attn_input_prob_offset = params.mr_info.attn_input_prob_offset,
               .attn_input_scaling_factor_offset = params.mr_info.attn_input_scaling_factor_offset,
               // Batched staging parameters (packed layout)
               .rdma_send_staging_offset = params.mr_info.rdma_send_staging_offset,
               .rdma_inter_node_group_packed_offset = params.mr_info.rdma_inter_node_group_packed_offset,
               .bytes_per_entry = params.mr_info.bytes_per_entry,
               .max_tokens_per_dest = params.mr_info.max_tokens_per_dest,
               // Streaming signal parameters
               .signals_tail_base = params.mr_info.signals_tail_base,
               .num_max_rdma_chunked_send_tokens = params.mr_info.num_max_rdma_chunked_send_tokens
            };

    return kp;
}

// Template dispatch launcher for forward/backward and sync modes
template<bool FORWARD_DISPATCH>
void dispatch_impl(
    const DispatchParams& params,
    int max_tokens_per_rank,
    int num_nodes,
    bool use_fp8,
    int num_blocks,
    int chunk_tokens,
    cudaStream_t stream
) {
    HYBRIDEP_SWITCH_DATATYPE(use_fp8, {
        HYBRIDEP_SWITCH_NUM_LSA_TEAMS(num_nodes, {
            HYBRIDEP_SWITCH_LSA_TEAM_SIZE(params.num_ranks_per_node, {
                // TMA requires prob buffer (experts_per_node * sizeof(float)) to be 16B aligned
                // Check alignment at runtime now that experts_per_rank is dynamic
                const int experts_per_node = params.experts_per_rank * params.num_ranks_per_node;
                assert((experts_per_node * sizeof(float)) % 16 == 0 &&
                       "experts_per_node must be multiple of 4 for TMA alignment");

                using HybridEPType = ::hybrid_ep::hybrid_ep<
                    MAX_SUPPORTED_TOKENS_PER_RANK,
                    NUM_LSA_TEAMS,
                    LSA_TEAM_SIZE>;

                auto kp = build_dispatch_param<TOKEN_DATA_TYPE, LSA_TEAM_SIZE>(params);

                // [LSA_TEAM_SIZE>16 SMEM relief] At LSA_TEAM_SIZE=32 (EP32
                // single-node MNNVL inside NVL72), s2d_map + prob inflate
                // dispatch SMEM past sm_103 ~228KiB cap with the default
                // 12-stage / 4-in-flight ring at chunk=128. Drop to 8/3
                // (same depth as dispatch TMA-overlap path, well validated).
                // EP4/EP8 single-node MNNVL keep the default 12/4 because
                // their experts_per_node is small enough.
                constexpr int dispatch_num_stages_lsa = (LSA_TEAM_SIZE > 16)
                    ? HYBRIDEP_DISPATCH_NUM_OF_STAGES_LSA32
                    : HYBRIDEP_DISPATCH_NUM_OF_STAGES;
                constexpr int dispatch_num_inflight_lsa = (LSA_TEAM_SIZE > 16)
                    ? HYBRIDEP_DISPATCH_NUM_OF_IN_FLIGHT_S2G_LSA32
                    : HYBRIDEP_DISPATCH_NUM_OF_IN_FLIGHT_S2G;
                // Same LSA_TEAM_SIZE>16 SMEM relief but for the ENABLE_TMA_COPY=true
                // (dispatch_overlap) path. The OVERLAP variant adds a ~56 KiB TMA copy
                // ring; EP32 single-node MNNVL doubles s2d_map vs EP16, pushing total
                // past sm_103 cap at the default OVERLAP stages=8. Drop to 6 / 2.
                constexpr int dispatch_num_stages_overlap_lsa = (LSA_TEAM_SIZE > 16)
                    ? HYBRIDEP_DISPATCH_NUM_OF_STAGES_OVERLAP_LSA32
                    : HYBRIDEP_DISPATCH_NUM_OF_STAGES_OVERLAP;
                constexpr int dispatch_num_inflight_overlap_lsa = (LSA_TEAM_SIZE > 16)
                    ? HYBRIDEP_DISPATCH_NUM_OF_IN_FLIGHT_S2G_OVERLAP_LSA32
                    : HYBRIDEP_DISPATCH_NUM_OF_IN_FLIGHT_S2G_OVERLAP;

                if constexpr (NUM_LSA_TEAMS == 1) {
                    // NV72 / MNNVL single-LSA-domain path. Production build
                    // only instantiates the 3 cells (16,64) / (32,128) /
                    // (64,128) that select_ht_nv72_tuning ever picks at
                    // runtime, cutting dispatch_kernel instantiations from 9
                    // to 3 (and from 36 to 12 once you fold dtype x forward).
                    //
                    // Calibration build (rebuild with
                    // -DNCCL_EP_HT_NV72_FULL_MATRIX) keeps the full 9-cell
                    // matrix so env overrides like NCCL_EP_HT_NV72_NUM_SMS
                    // can pick any (NUM_SMS, CHUNK) pair for sweeps.
                    //
                    // ENABLE_TMA_COPY=true is opt-in via macro and limited to
                    // BF16 + (32,128) / (64,128) only.  The kAllowTmaCopy
                    // gate is identical in both build flavours so the
                    // production cells get the right instances either way.
                    //
                    // Note: preprocessor directives inside macro arguments
                    // are undefined behavior (ISO C99 6.10.3p11), so the
                    // overlap and full-matrix #ifdef wrap the macro call
                    // rather than sit inside the body.
#ifdef NCCL_EP_HT_NV72_FULL_MATRIX
#  ifdef NCCL_EP_ENABLE_HT_TMA_COPY_OVERLAP
                    HYBRIDEP_SWITCH_NUM_BLOCKS(num_blocks, {
                        HYBRIDEP_SWITCH_CHUNK_TOKENS(chunk_tokens, {
                            constexpr bool kAllowTmaCopy =
                                std::is_same<TOKEN_DATA_TYPE, uint16_t>::value &&
                                ((NUM_BLOCKS_TUNED == 32 && CHUNK_TOKENS_TUNED == 128) ||
                                 (NUM_BLOCKS_TUNED == 64 && CHUNK_TOKENS_TUNED == 128));
                            if constexpr (kAllowTmaCopy) {
                                if (params.user_output_token != nullptr) {
                                    HybridEPType::template dispatch<
                                        TOKEN_DATA_TYPE,
                                        dispatch_num_stages_overlap_lsa,
                                        dispatch_num_inflight_overlap_lsa,
                                        CHUNK_TOKENS_TUNED,
                                        NUM_BLOCKS_TUNED,
                                        FORWARD_DISPATCH,
                                        true>(kp, stream);
                                } else {
                                    HybridEPType::template dispatch<
                                        TOKEN_DATA_TYPE,
                                        dispatch_num_stages_lsa,
                                        dispatch_num_inflight_lsa,
                                        CHUNK_TOKENS_TUNED,
                                        NUM_BLOCKS_TUNED,
                                        FORWARD_DISPATCH,
                                        false>(kp, stream);
                                }
                            } else {
                                HybridEPType::template dispatch<
                                    TOKEN_DATA_TYPE,
                                    dispatch_num_stages_lsa,
                                    dispatch_num_inflight_lsa,
                                    CHUNK_TOKENS_TUNED,
                                    NUM_BLOCKS_TUNED,
                                    FORWARD_DISPATCH,
                                    false>(kp, stream);
                            }
                        });
                    });
#  else
                    HYBRIDEP_SWITCH_NUM_BLOCKS(num_blocks, {
                        HYBRIDEP_SWITCH_CHUNK_TOKENS(chunk_tokens, {
                            // [perf-test bigwork] chunk=256 + LSA32 (EP32 NV72) needs
                            // extra SMEM relief: s2d_map=chunk*ranks_per_node*16=128KiB
                            // alone, default stages=8 token buf 112KiB pushes total
                            // past 228KiB cap. Reduce stages 8->4 to fit
                            // (token 56 + s2d 128 + prob 24 = ~208 KiB).
                            constexpr int dispatch_num_stages_dyn =
                                (LSA_TEAM_SIZE > 16 && CHUNK_TOKENS_TUNED >= 256)
                                    ? 4
                                    : dispatch_num_stages_lsa;
                            constexpr int dispatch_num_inflight_dyn =
                                (LSA_TEAM_SIZE > 16 && CHUNK_TOKENS_TUNED >= 256)
                                    ? 2
                                    : dispatch_num_inflight_lsa;
                            HybridEPType::template dispatch<
                                TOKEN_DATA_TYPE,
                                dispatch_num_stages_dyn,
                                dispatch_num_inflight_dyn,
                                CHUNK_TOKENS_TUNED,
                                NUM_BLOCKS_TUNED,
                                FORWARD_DISPATCH,
                                false>(kp, stream);
                        });
                    });
#  endif
#else
#  ifdef NCCL_EP_ENABLE_HT_TMA_COPY_OVERLAP
                    HYBRIDEP_SWITCH_NV72_CELL(num_blocks, chunk_tokens, {
                        constexpr bool kAllowTmaCopy =
                            std::is_same<TOKEN_DATA_TYPE, uint16_t>::value &&
                            ((NUM_BLOCKS_TUNED == 32 && CHUNK_TOKENS_TUNED == 128) ||
                             (NUM_BLOCKS_TUNED == 64 && CHUNK_TOKENS_TUNED == 128));
                        if constexpr (kAllowTmaCopy) {
                            if (params.user_output_token != nullptr) {
                                // ENABLE_TMA_COPY=true instance uses the
                                // reduced NUM_OF_STAGES_OVERLAP (8 vs 12)
                                // and NUM_OF_IN_FLIGHT_S2G_OVERLAP (3 vs 4)
                                // to free smem room for the 4-warp TMA
                                // copy ring buffer (4 slots * 14 KiB =
                                // 56 KiB).  Without these reductions
                                // cudaFuncSetAttribute returns 'invalid
                                // argument' on sm_103 because total smem
                                // exceeds the per-block max ~228 KiB.
                                HybridEPType::template dispatch<
                                    TOKEN_DATA_TYPE,
                                    dispatch_num_stages_overlap_lsa,
                                    dispatch_num_inflight_overlap_lsa,
                                    CHUNK_TOKENS_TUNED,
                                    NUM_BLOCKS_TUNED,
                                    FORWARD_DISPATCH,
                                    true>(kp, stream);
                            } else {
                                HybridEPType::template dispatch<
                                    TOKEN_DATA_TYPE,
                                    dispatch_num_stages_lsa,
                                    dispatch_num_inflight_lsa,
                                    CHUNK_TOKENS_TUNED,
                                    NUM_BLOCKS_TUNED,
                                    FORWARD_DISPATCH,
                                    false>(kp, stream);
                            }
                        } else {
                            HybridEPType::template dispatch<
                                TOKEN_DATA_TYPE,
                                dispatch_num_stages_lsa,
                                dispatch_num_inflight_lsa,
                                CHUNK_TOKENS_TUNED,
                                NUM_BLOCKS_TUNED,
                                FORWARD_DISPATCH,
                                false>(kp, stream);
                        }
                    });
#  else
                    HYBRIDEP_SWITCH_NV72_CELL(num_blocks, chunk_tokens, {
                        HybridEPType::template dispatch<
                            TOKEN_DATA_TYPE,
                            dispatch_num_stages_lsa,
                            dispatch_num_inflight_lsa,
                            CHUNK_TOKENS_TUNED,
                            NUM_BLOCKS_TUNED,
                            FORWARD_DISPATCH,
                            false>(kp, stream);
                    });
#  endif
#endif
                } else {
                    HybridEPType::template dispatch<
                        TOKEN_DATA_TYPE,
                        dispatch_num_stages_lsa,
                        dispatch_num_inflight_lsa,
                        HT_OF_NUM_TOKENS_PER_CHUNK,
                        HYBRIDEP_DISPATCH_NUM_OF_BLOCKS,
                        FORWARD_DISPATCH,
                        false>(kp, stream);
                }
            });
        });
    });
}

void call_dispatch(
    const DispatchParams& params,
    int max_tokens_per_rank,
    int num_nodes,
    bool use_fp8,
    bool forward_dispatch,
    int num_blocks,
    int chunk_tokens,
    cudaStream_t stream
) {
    // Dispatch based on forward/backward and sync mode
    if (forward_dispatch) {
        dispatch_impl<true>(
            params, max_tokens_per_rank,
            num_nodes, use_fp8, num_blocks, chunk_tokens, stream);

    } else {
        dispatch_impl<false>(
            params, max_tokens_per_rank,
            num_nodes, use_fp8, num_blocks, chunk_tokens, stream);

    }
}

// ============================================================================
// Combine wrapper implementation
// ============================================================================

// Helper to populate combine_kernel_param_t from CombineParams
template<int LSA_TEAM_SIZE>
::hybrid_ep::combine_kernel_param_t<LSA_TEAM_SIZE>
build_combine_param(const CombineParams& params) {
    ::hybrid_ep::combine_kernel_param_t<LSA_TEAM_SIZE> kp{};

    // Copy IPC buffer pointers from HOST arrays into embedded param struct arrays.
    // This allows fast __grid_constant__ access in the kernel (vs slow global memory indirection).
    for (int i = 0; i < params.num_ranks_per_node; i++) {
        kp.expert_input_token[i] = params.expert_input_token_ptrs[i];
        kp.expert_input_prob[i] = params.expert_input_prob_ptrs ?
            params.expert_input_prob_ptrs[i] : nullptr;
    }

    // Model configuration
    kp.hidden_dim = params.hidden_dim;
    kp.experts_per_rank = params.experts_per_rank;
    kp.num_of_ranks_per_node = params.num_ranks_per_node;
    // User output buffers
    kp.attn_output_token = reinterpret_cast<uint16_t*>(params.attn_output_token);
    kp.attn_output_prob = params.attn_output_prob;

    // RDMA buffers (multi-node only)
    kp.rdma_intra_node_red_token = params.rdma_intra_node_red_token;
    kp.rdma_intra_node_red_prob = params.rdma_intra_node_red_prob;
    kp.rdma_inter_node_group_token = params.combine_rdma_inter_node_group_token;
    kp.rdma_inter_node_group_prob = params.combine_rdma_inter_node_group_prob;

    // Metadata
    kp.sparse_to_dense_map = params.sparse_to_dense_map;
    kp.rdma_to_attn_map = params.rdma_to_attn_map;
    kp.attn_to_rdma_map = params.attn_to_rdma_map;

    // Sync flags
    kp.expected_rdma_flag_value = params.combine_expected_rdma_flag_value;
    kp.expected_intra_node_flag_value = params.combine_expected_intra_node_flag_value;
    kp.rdma_inter_node_group_flags = params.combine_rdma_inter_node_group_flags;
    kp.intra_node_write_completion_flags = params.combine_intra_node_write_completion_flags;

    // Runtime config
    kp.local_rank = params.local_rank;
    kp.node_rank = params.node_rank;
    kp.num_of_tokens_per_rank = params.num_tokens_per_rank;

    // Pass device communicators and windows
    kp.dcomms = params.dcomms;
    kp.nccl_window = params.nccl_window;
    kp.num_gin_comms = params.num_gin_comms;
    kp.num_ctx_per_comm = params.num_ctx_per_comm;
    kp.gin_base_ptr = params.gin_base_ptr;
    kp.signals_base = params.signals_base;
    kp.combine_signal_offset = params.combine_signal_offset;
    // Use offsets relative to gin_base_ptr
    kp.mr_info = {
               .rdma_intra_node_red_token_offset = params.mr_info.rdma_intra_node_red_token_offset,
               .combine_rdma_inter_node_group_token_offset = params.mr_info.combine_rdma_inter_node_group_token_offset,
               .rdma_intra_node_red_prob_offset = params.mr_info.rdma_intra_node_red_prob_offset,
               .combine_rdma_inter_node_group_prob_offset = params.mr_info.combine_rdma_inter_node_group_prob_offset
    };

    // [Combine input-copy overlap, 2026-05-09] Pass user_input_token + per-peer
    // ready ptrs + chunk size + cumulative expected counter.  Only used by
    // the kernel template instance compiled with ENABLE_INPUT_COPY=true; for
    // the baseline path these stay nullptr/0 and the gate in adapter.cu
    // selects ENABLE_INPUT_COPY=false.
    kp.user_input_token = params.user_input_token;
    kp.combine_input_chunk_tokens = params.combine_input_chunk_tokens;
    kp.combine_input_expected = params.combine_input_expected;
    if (params.combine_input_ready_ptrs != nullptr) {
        for (int r = 0; r < LSA_TEAM_SIZE; r++) {
            kp.combine_input_ready[r] = params.combine_input_ready_ptrs[r];
        }
    } else {
        for (int r = 0; r < LSA_TEAM_SIZE; r++) {
            kp.combine_input_ready[r] = nullptr;
        }
    }
    kp.num_recv_tokens = params.num_recv_tokens;

    return kp;
}

// Template combine launcher for forward/backward
template<bool BACKWARD_COMBINE>
void combine_impl(
    const CombineParams& params,
    int max_tokens_per_rank,
    int num_nodes,
    int num_blocks,
    int chunk_tokens,
    cudaStream_t stream
) {
    // HT combine doesn't support FP8, only BF16
    using TOKEN_DATA_TYPE = uint16_t;

        HYBRIDEP_SWITCH_NUM_LSA_TEAMS(num_nodes, {
            HYBRIDEP_SWITCH_LSA_TEAM_SIZE(params.num_ranks_per_node, {
                // TMA requires prob buffer (experts_per_node * sizeof(float)) to be 16B aligned
                const int experts_per_node = params.experts_per_rank * params.num_ranks_per_node;
                assert((experts_per_node * sizeof(float)) % 16 == 0 &&
                       "experts_per_node must be multiple of 4 for TMA alignment");

                using HybridEPType = ::hybrid_ep::hybrid_ep<
                    MAX_SUPPORTED_TOKENS_PER_RANK,
                    NUM_LSA_TEAMS,
                    LSA_TEAM_SIZE>;

                auto kp = build_combine_param<LSA_TEAM_SIZE>(params);

                // Select config based on NUM_LSA_TEAMS (single-node: 12 stages/2 pipelines, multi-node: 5 stages/1 pipeline)
                // [LSA_TEAM_SIZE>16 SMEM relief, 2026-05-18] EP32 single-node
                // MNNVL (LSA_TEAM_SIZE=32) inflates prob/inter buffers, push
                // combine SMEM past sm_103 cap. Drop SINGLENODE G2S depth from
                // 12 to 6 in that case. EP4/EP8/EP16 single-node MNNVL keep
                // the deeper 12-stage default since their experts_per_node is
                // small enough.
                constexpr int num_stages_g2s = (NUM_LSA_TEAMS == 1)
                    ? ((LSA_TEAM_SIZE > 16)
                        ? HYBRIDEP_COMBINE_SINGLENODE_NUM_OF_STAGES_G2S_LSA32
                        : HYBRIDEP_COMBINE_SINGLENODE_NUM_OF_STAGES_G2S)
                    : HYBRIDEP_COMBINE_MULTINODE_NUM_OF_STAGES_G2S;
                constexpr int num_stages_s2g = (NUM_LSA_TEAMS == 1)
                    ? HYBRIDEP_COMBINE_SINGLENODE_NUM_OF_STAGES_S2G
                    : HYBRIDEP_COMBINE_MULTINODE_NUM_OF_STAGES_S2G;

                if constexpr (NUM_LSA_TEAMS == 1) {
                    // Production: only the 3 NV72 cells select_ht_nv72_tuning
                    // ever picks.  Calibration build (full 9-cell matrix)
                    // is gated by NCCL_EP_HT_NV72_FULL_MATRIX, identical
                    // intent as dispatch_impl above.
#ifdef NCCL_EP_HT_NV72_FULL_MATRIX
                    HYBRIDEP_SWITCH_NUM_BLOCKS(num_blocks, {
                        HYBRIDEP_SWITCH_CHUNK_TOKENS(chunk_tokens, {
                            HybridEPType::template combine<
                                num_stages_g2s,
                                num_stages_s2g,
                                CHUNK_TOKENS_TUNED,
                                HYBRIDEP_COMBINE_NUM_OF_TOKENS_PER_GROUP,
                                NUM_BLOCKS_TUNED,
                                HYBRIDEP_COMBINE_NUM_OF_ADDITIONAL_IN_FLIGHT_S2G,
                                BACKWARD_COMBINE>(kp, stream);
                        });
                    });
#else
#  ifdef NCCL_EP_ENABLE_HT_COMBINE_INPUT_COPY
                    HYBRIDEP_SWITCH_NV72_CELL(num_blocks, chunk_tokens, {
                        // [Combine input-copy overlap, 2026-05-09] Allow
                        // ENABLE_INPUT_COPY only on the production NV72
                        // cells (32,128) and (64,128) where chunked
                        // pipelining beats baseline cudaMemcpyAsync.
                        constexpr bool kAllowInputCopy =
                            (NUM_BLOCKS_TUNED == 32 && CHUNK_TOKENS_TUNED == 128) ||
                            (NUM_BLOCKS_TUNED == 64 && CHUNK_TOKENS_TUNED == 128);
                        if constexpr (kAllowInputCopy) {
                            if (kp.user_input_token != nullptr) {
                                // ENABLE_INPUT_COPY=true uses reduced
                                // NUM_OF_STAGES_G2S (8 vs 12) to fit the
                                // INPUT_COPY ring within sm_103's 228 KiB
                                // dynamic-SMEM cap (see configs comment).
                                HybridEPType::template combine<
                                    HYBRIDEP_COMBINE_INPUT_COPY_NUM_OF_STAGES_G2S,
                                    num_stages_s2g,
                                    CHUNK_TOKENS_TUNED,
                                    HYBRIDEP_COMBINE_NUM_OF_TOKENS_PER_GROUP,
                                    NUM_BLOCKS_TUNED,
                                    HYBRIDEP_COMBINE_NUM_OF_ADDITIONAL_IN_FLIGHT_S2G,
                                    BACKWARD_COMBINE,
                                    /*ENABLE_INPUT_COPY=*/true>(kp, stream);
                            } else {
                                HybridEPType::template combine<
                                    num_stages_g2s,
                                    num_stages_s2g,
                                    CHUNK_TOKENS_TUNED,
                                    HYBRIDEP_COMBINE_NUM_OF_TOKENS_PER_GROUP,
                                    NUM_BLOCKS_TUNED,
                                    HYBRIDEP_COMBINE_NUM_OF_ADDITIONAL_IN_FLIGHT_S2G,
                                    BACKWARD_COMBINE,
                                    /*ENABLE_INPUT_COPY=*/false>(kp, stream);
                            }
                        } else {
                            HybridEPType::template combine<
                                num_stages_g2s,
                                num_stages_s2g,
                                CHUNK_TOKENS_TUNED,
                                HYBRIDEP_COMBINE_NUM_OF_TOKENS_PER_GROUP,
                                NUM_BLOCKS_TUNED,
                                HYBRIDEP_COMBINE_NUM_OF_ADDITIONAL_IN_FLIGHT_S2G,
                                BACKWARD_COMBINE,
                                /*ENABLE_INPUT_COPY=*/false>(kp, stream);
                        }
                    });
#  else
                    HYBRIDEP_SWITCH_NV72_CELL(num_blocks, chunk_tokens, {
                        HybridEPType::template combine<
                            num_stages_g2s,
                            num_stages_s2g,
                            CHUNK_TOKENS_TUNED,
                            HYBRIDEP_COMBINE_NUM_OF_TOKENS_PER_GROUP,
                            NUM_BLOCKS_TUNED,
                            HYBRIDEP_COMBINE_NUM_OF_ADDITIONAL_IN_FLIGHT_S2G,
                            BACKWARD_COMBINE>(kp, stream);
                    });
#  endif
#endif
                } else {
                    HybridEPType::template combine<
                        num_stages_g2s,
                        num_stages_s2g,
                        HT_OF_NUM_TOKENS_PER_CHUNK,
                        HYBRIDEP_COMBINE_NUM_OF_TOKENS_PER_GROUP,
                        HYBRIDEP_COMBINE_NUM_OF_BLOCKS,
                        HYBRIDEP_COMBINE_NUM_OF_ADDITIONAL_IN_FLIGHT_S2G,
                        BACKWARD_COMBINE>(kp, stream);
                }
            });
        });
}

void call_combine(
    const CombineParams& params,
    int max_tokens_per_rank,
    int num_nodes,
    bool backward_combine,
    int num_blocks,
    int chunk_tokens,
    cudaStream_t stream
) {
    if (backward_combine) {
        combine_impl<true>(
            params, max_tokens_per_rank,
            num_nodes, num_blocks, chunk_tokens, stream);
    } else {
        combine_impl<false>(
            params, max_tokens_per_rank,
            num_nodes, num_blocks, chunk_tokens, stream);
    }
}

} // namespace hybridep
} // namespace nccl_ep
