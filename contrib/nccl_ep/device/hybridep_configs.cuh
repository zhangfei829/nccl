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

// Reduced combine NUM_OF_STAGES_G2S used only by ENABLE_INPUT_COPY=true
// instances to free SMEM for the INPUT_COPY ring buffer.
//
// SMEM budget (single-node, BF16, hidden=7168, NUM_OF_STAGES_S2G=2):
//   inter_node_token_G2S  : 8 stages * 14 KiB    = 112 KiB  (vs 168 at 12)
//   inter_node_token_S2G  : 2 stages * 14 KiB    =  28 KiB
//   inter_node_token_tail :                       ~24 KiB
//   INPUT_COPY ring       : 2 stages * 14 KiB    =  28 KiB
//   mbar/flag/misc        :                        <1 KiB
//   total                 :                       ~193 KiB  fits sm_103 max ~228 KiB
//
// (At default NUM_OF_STAGES_G2S=12 the total is ~248 KiB which trips
// cudaFuncSetAttribute(...,MaxDynamicSharedMemorySize) -> invalid argument
// at hybrid_ep.cuh:5173.)
#define HYBRIDEP_COMBINE_INPUT_COPY_NUM_OF_STAGES_G2S 8

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

// ============== Combine input-copy overlap (2026-05-09) ==================
// Mirror of dispatch TMA copy overlap (commit 9cfbfce) but for combine's
// pre-kernel input D2D copy:
//   baseline: host cudaMemcpyAsync(expert_input_token, x->data, ...) ~555us
//             at t=8192 EP=16 (33% of combine total_stream).
//   overlap:  in-kernel COPY warp_group does chunk-by-chunk
//             cp.async.bulk(x->data → SMEM → expert_input_token) and
//             atomicAdd combine_input_ready[chunk_id]; the G2S/RED warps
//             on peer ranks spin-wait combine_input_ready[chunk_id] before
//             reading peer's expert_input_token chunk_id.  This pipelines
//             input copy with main combine reduce (chunked dependency).
//
// Sizing:
//   - 1 warp/block (32 lanes) -- input copy is throughput-bound (bytes/cycle),
//     not latency-bound, so 1 warp per CTA already saturates the DMA path.
//   - 1 SMEM ring slot per warp (single-stage), 2 mbarriers (g2s + s2g).
//   - chunk_tokens = 32 (matches dispatch_copy_chunk_tokens for symmetric
//     fabric memory layout / counter array sizing).
// CORRECTION (2026-05-11): expert_input_token is allocated by cuMemCreate
// (CU_MEM_LOCATION_TYPE_DEVICE) on the LOCAL rank's HBM and only the FABRIC
// handle is shared with peers.  The src->dst path of combine_input_copy
// (user_input_token -> expert_input_token) is entirely LOCAL D2D on the
// current GPU's HBM; no NVSwitch traffic until a peer later reads our
// expert_input_token via its imported fabric handle.  Earlier comments that
// blamed "fabric STORE serialization" or "fabric memory write ordering" for
// V1/V3/V4 issues were wrong; the slowness is local D2D mechanics in-kernel.
//
// V5-A: multi-warp same-warp LOAD+STORE pattern (clones the production
// dispatch_tma_copy_warp_group_device_function in hybrid_ep.cuh:3846 which
// works for HBM dst with HYBRIDEP_DISPATCH_TMA_COPY_WARPS=4).  Each warp
// owns 1 SMEM slot + 1 mbarrier and processes tokens
// [warp_id, warp_id+kNumWarps, warp_id+2*kNumWarps, ...] within each chunk.
//
// V3 (1 PROD + 1 CONS cross-warp serial) measured combine_kernel +1153us
// (= ~2.25us/token, ~30x above what cp.async.bulk per-token cost should be
// on local HBM).  Single-warp lane-0 issue inside the combine kernel where
// other warp_groups (G2S/S2G/N2N) also run gets long per-token mbarrier +
// fence + issue overhead due to SM scheduling sharing.  V4 in-flight
// wait_group_read<3> made it worse (-64% wall BW) because deferred
// mbarrier_arrive broke the PROD/CONS overlap.
//
// V5-A spreads that overhead across 4 warps issuing independently:
// 4 SMEM slots, 4 mbarriers, each warp's lane-0 issues its own LOAD+STORE
// pipeline for token subset [w, w+4, w+8, ...].
//
// V1 (1 warp same-warp) crashed with rc=134 -- root cause unknown (NOT
// fabric write since this is local D2D).  V5-A's per-warp same-warp pattern
// matches V1's structure but with 4 warps; if V5-A also crashes, the V1
// root cause is structural (e.g. SMEM sharing with combine kernel's other
// warp_groups, or mbarrier init ordering) and needs different debugging.
#define HYBRIDEP_COMBINE_INPUT_COPY_WARPS 4
#define HYBRIDEP_COMBINE_INPUT_COPY_CHUNK_TOKENS 32
#define HYBRIDEP_COMBINE_INPUT_COPY_NUM_STAGES 4  // = NUM_WARPS in V5-A (1 slot per warp)

// ============================================================================
// Preprocessing kernel configuration
// ============================================================================
#define HYBRIDEP_NUM_THREADS_PER_BLOCK_PREPROCESSING 512
#define HYBRIDEP_NUM_BLOCKS_PREPROCESSING HYBRIDEP_MAX_NUM_SMS_PER_RANK
