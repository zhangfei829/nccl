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

#pragma once

// ============================================================================
// HT-specific configuration constants
// ============================================================================
#define HYBRIDEP_MAX_NUM_SMS_PER_RANK 16
// ============================================================================
// Dispatch configuration constants
// ============================================================================
#define HYBRIDEP_DISPATCH_NUM_OF_STAGES 12
#define HYBRIDEP_DISPATCH_NUM_OF_IN_FLIGHT_S2G 4
#define HYBRIDEP_DISPATCH_NUM_OF_BLOCKS HYBRIDEP_MAX_NUM_SMS_PER_RANK
#define HYBRIDEP_DISPATCH_NUM_OF_PIPELINES_PER_BLOCK 2
#define HYBRIDEP_DISPATCH_N2N_WARPS 2
// Maximum consecutive tokens batched into a single RDMA put in dispatch N2N.
// Larger batches reduce NIC doorbell overhead but may delay first-byte latency.
#define HYBRIDEP_DISPATCH_RDMA_BATCH_SIZE 4

// Experimental dispatch output copy-overlap path.  Peer ranks still write
// this rank's internal HT output buffer; the COPY warp group inside
// dispatch_kernel copies ready output chunks from internal buffer to the
// user recv_x buffer.
//
// Multi-warp design (per the 2026-05-07 PTX-driven redesign):
//   The bottleneck of the depth=2 single-warp prototype was issue-side
//   sequential overhead: lane 0 sequentially executes wait_mbar -> fence
//   -> issue_s2g -> commit_group -> wait_group_read -> issue_g2s_prefetch
//   (~6 PTX ops, ~9 us per token round).  PTX ISA 9.7.9.25.6.1 says the
//   bulk async-group is per-thread, so 4 lanes (one per warp) can issue
//   in parallel and each tracks its own in-flight group.  Multi-warp =
//   4x issue rate, saturating the TMA hardware that was previously idle
//   on single-warp.
//
//   Each warp owns 1 ring slot (single-stage per warp); 4 warps total
//   = 4 in-flight transfers system-wide.  Token assignment: warp w
//   handles tokens w, w+4, w+8, ...  No cross-warp mbarrier is needed
//   because each warp uses its own slot, mbarrier and commit-group.
#define HYBRIDEP_DISPATCH_TMA_COPY_WARPS 4
#define HYBRIDEP_DISPATCH_TMA_COPY_TOKENS_PER_STAGE 1
#define HYBRIDEP_DISPATCH_TMA_COPY_CHUNK_TOKENS 32

// Per-warp ring depth.  1 means single-stage within each warp (g2s and
// s2g serialise inside the warp), but kNumWarps warps in parallel give
// kNumWarps total in-flight transfers, hiding the per-warp
// wait_group_read<0> cost.
#define HYBRIDEP_DISPATCH_TMA_COPY_NUM_STAGES_PER_WARP 1

// Total number of ring slots in the TMA copy SMEM region.  Each warp's
// lane 0 owns slots [warp_id * NUM_STAGES_PER_WARP ... (warp_id+1) *
// NUM_STAGES_PER_WARP).  The dispatch_smem_layout allocator and
// mbarrier-init loop both use this as the total slot count.
//
// CRITICAL: this MUST equal NUM_WARPS * NUM_STAGES_PER_WARP.  An earlier
// commit accidentally set this to 1 (per-warp depth) while the device
// function still indexed slots by warp_id 0..NUM_WARPS-1, causing OOB
// SMEM access on warps 1..3.  This computed form keeps the two in sync.
#define HYBRIDEP_DISPATCH_TMA_COPY_NUM_STAGES \
    (HYBRIDEP_DISPATCH_TMA_COPY_WARPS * HYBRIDEP_DISPATCH_TMA_COPY_NUM_STAGES_PER_WARP)

// Reduced main G2S/S2G pipeline depth used only by ENABLE_TMA_COPY=true
// dispatch_kernel instances.  Frees smem for the 4-slot TMA copy ring
// buffer (4 * hidden * sizeof(token) = 56 KiB at hidden=7168 BF16).
//
// SMEM budget for ENABLE_TMA_COPY=true (64,128) BF16:
//   main token  : 8 stages * 14 KiB = 112 KiB
//   main prob   : 8 KiB
//   s2d         : 32 KiB
//   TMA ring    : 4 slots * 14 KiB = 56 KiB
//   total       : ~208 KiB                  fits sm_103 ~228 KiB
//
// 8 / NUM_PIPELINES=2 = 4 stages_per_pipeline.  The static_assert in
// S2G_warp_group_device_function requires NUM_OF_IN_FLIGHT_S2G <
// stages_per_pipeline (strictly), so we need a reduced in-flight count
// for the overlap path (see HYBRIDEP_DISPATCH_NUM_OF_IN_FLIGHT_S2G_OVERLAP
// below).
#define HYBRIDEP_DISPATCH_NUM_OF_STAGES_OVERLAP 8

// Reduced in-flight S2G depth for ENABLE_TMA_COPY=true dispatch_kernel.
// Default (4) doesn't fit because NUM_OF_STAGES_OVERLAP=8 / NUM_PIPELINES=2
// = 4 stages_per_pipeline, and the assertion is < (strict).
#define HYBRIDEP_DISPATCH_NUM_OF_IN_FLIGHT_S2G_OVERLAP 3

// =============== Path C: small-token LDST copy (2026-05-09) ================
// For the (16,64) cell at small tokens (t<4096), TMA pipeline overhead
// (~150 us per chunk: mbarrier_init + cross-rank wait + multi-stage
// commit/wait_group) dwarfs the actual D2D physical transfer time
// (12-24 MB total / 1.2 TB/s HBM ~0 us).  baseline cudaMemcpyAsync stream
// tail is ~27 us for these tokens; an in-kernel copy can beat that only
// if its overhead is much smaller than 27 us.
//
// Path C uses simple vectorized ld.global.v4 + st.global.v4 (16-byte
// load/store per thread) to do the copy in-kernel.  No mbarrier, no SMEM
// ring buffer, no commit_group/wait_group.  Single warp is enough because
// the per-chunk physical transfer is sub-microsecond and adding warps
// only adds entry-barrier overhead.
//
// Cost breakdown estimate (t=128, chunk_active=8, hidden=7168 BF16):
//   - warp_group entry barrier:               ~3 us
//   - cross-rank dispatch_copy_ready spin:    ~5 us  (same physical floor as TMA)
//   - vec ld+st 8 tokens * 14336 B / 512 B per warp transaction = 224 ops:
//                                             ~0.9 us @ 1 GHz SM clock
//   - exit drain + sync:                      ~3 us
//   total Δ_kernel ≈ 12 us  (vs TMA's ~150 us, vs cudaMemcpyAsync stream tail 27 us)
//
// Net win on t=128 ≈ 27 - 12 = +15 us  ~10% improvement on dispatch_avg.
#define HYBRIDEP_DISPATCH_LDST_COPY_WARPS 1


// ============================================================================
// Combine configuration constants
// ============================================================================
// Single-node configuration: optimized for intra-node only (2 pipelines, deep FIFO)
#define HYBRIDEP_COMBINE_SINGLENODE_NUM_OF_STAGES_G2S 12
#define HYBRIDEP_COMBINE_SINGLENODE_NUM_OF_STAGES_S2G 2

// Multi-node configuration: optimized for inter-node RDMA (1 pipeline, shallow FIFO)
#define HYBRIDEP_COMBINE_MULTINODE_NUM_OF_STAGES_G2S 4
#define HYBRIDEP_COMBINE_MULTINODE_NUM_OF_STAGES_S2G 2

#define HYBRIDEP_COMBINE_NUM_OF_TOKENS_PER_GROUP 4
#define HYBRIDEP_COMBINE_NUM_OF_BLOCKS HYBRIDEP_MAX_NUM_SMS_PER_RANK
#define HYBRIDEP_COMBINE_NUM_OF_ADDITIONAL_IN_FLIGHT_S2G 2

// Streaming overlap: tokens between drain+signal from reduction warp to RDMA warp.
// 0 = disable streaming (fall back to chunk-level mbarrier only).
#define HYBRIDEP_COMBINE_RDMA_STREAMING_BATCH 8

// ============================================================================
// Preprocessing kernel configuration
// ============================================================================
#define HYBRIDEP_NUM_THREADS_PER_BLOCK_PREPROCESSING 512
#define HYBRIDEP_NUM_BLOCKS_PREPROCESSING HYBRIDEP_MAX_NUM_SMS_PER_RANK
