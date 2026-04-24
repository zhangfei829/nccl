/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

#include <unistd.h>
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <new>
#include <optional>
#include <set>
#include <string>
#include <vector>
#include <nccl.h>
#include <nccl_device.h>
#include "include/common.hpp"
#include "include/nccl_ep.h"

// HT (High Throughput) includes
#include "device/hybridep_adapter.cuh"
#include "device/hybridep_configs.cuh"
#include "device/fullmesh.cuh"

// Internal definition of the opaque ncclNDTensor type
struct ncclNDTensor {
    unsigned int version;
    unsigned int ndim;
    unsigned int* sizes;
    unsigned int* strides;
    ncclDataType_t datatype;
    void* data;
    unsigned int tag;
    ncclEpTensorFlags_t flags;
    bool owns_data;
};

// Forward declarations for HT functions
static ncclResult_t init_hybridep_intranode(ncclEpGroup_t ep_group, const ncclEpGroupConfig_t* config, cudaStream_t stream);
static ncclResult_t destroy_hybridep_intranode(ncclEpGroup_t ep_group);
static ncclResult_t init_hybridep_intranode_fabric(ncclEpGroup_t ep_group, const ncclEpGroupConfig_t* config, cudaStream_t stream);
static ncclResult_t destroy_hybridep_intranode_fabric(ncclEpGroup_t ep_group);
static ncclResult_t init_hybridep_internode(ncclEpGroup_t ep_group, const ncclEpGroupConfig_t* config, cudaStream_t stream);
static ncclResult_t destroy_hybridep_internode(ncclEpGroup_t ep_group);
// Forward declarations for FULLMESH (Phase 2) functions
static ncclResult_t init_fullmesh_intranode_fabric(ncclEpGroup_t ep_group, const ncclEpGroupConfig_t* config, cudaStream_t stream);
static ncclResult_t destroy_fullmesh_intranode_fabric(ncclEpGroup_t ep_group);
static void tensor_free(ncclEpGroup_t group, ncclNDTensor_t t);

// Define NCCL_CHECK_RESULT macro for NCCL error checking
#ifndef NCCL_CHECK_RESULT
#define NCCL_CHECK_RESULT(cmd) do {                 \
  ncclResult_t res = cmd;                           \
  if (res != ncclSuccess) {                         \
    fprintf(stderr, "NCCL error %d at %s:%d\n",     \
            res, __FILE__, __LINE__);               \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)
#endif

// CUDA driver API error checking (for VMM / fabric handle calls)
#ifndef CU_CHECK
#define CU_CHECK(cmd) do {                                  \
  CUresult _cu_res = (cmd);                                 \
  if (_cu_res != CUDA_SUCCESS) {                            \
    const char* _cu_err = nullptr;                          \
    cuGetErrorString(_cu_res, &_cu_err);                    \
    fprintf(stderr, "CU error %d (%s) at %s:%d\n",          \
            (int)_cu_res, _cu_err ? _cu_err : "?",          \
            __FILE__, __LINE__);                            \
    return ncclSystemError;                                 \
  }                                                         \
} while(0)
#endif

// CUDACHECK_RET macro for CUDA calls in functions returning ncclResult_t.
// Defined early here so it is visible to init_hybridep_intranode_fabric below
// (the original definition remained later in the file and is kept via #ifndef).
#ifndef CUDACHECK_RET
#define CUDACHECK_RET(cmd) do {                             \
    cudaError_t err = cmd;                                  \
    if (err != cudaSuccess) {                               \
        fprintf(stderr, "CUDA error %s:%d '%s'\n",          \
                __FILE__, __LINE__, cudaGetErrorString(err));\
        return ncclInternalError;                            \
    }                                                        \
} while(0)
#endif

// Helper function to convert ncclDataType_t to cudaDataType_t
static cudaDataType_t ncclDataTypeToCudaDataType(ncclDataType_t nccl_type) {
    switch (nccl_type) {
        case ncclFloat16:    return CUDA_R_16F;
        case ncclFloat32:    return CUDA_R_32F;
        case ncclFloat64:    return CUDA_R_64F;
        case ncclBfloat16:   return CUDA_R_16BF;
        case ncclInt8:       return CUDA_R_8I;
        case ncclInt32:      return CUDA_R_32I;
        case ncclInt64:      return CUDA_R_64I;
        case ncclUint8:      return CUDA_R_8U;
        case ncclUint32:     return CUDA_R_32U;
        case ncclUint64:     return CUDA_R_64U;
        default:
            assert(false && "Unsupported ncclDataType_t for conversion to cudaDataType_t");
            return CUDA_R_16BF; // Default fallback
    }
}

static size_t ncclTypeSize(ncclDataType_t nccl_type) {
    switch (nccl_type) {
        case ncclInt8:
        case ncclUint8:
        case ncclFloat8e4m3:
        case ncclFloat8e5m2:
            return 1;
        case ncclFloat16:
        case ncclBfloat16:
            return 2;
        case ncclInt32:
        case ncclUint32:
        case ncclFloat32:
            return 4;
        case ncclInt64:
        case ncclUint64:
        case ncclFloat64:
            return 8;
        default:
            assert(false && "Unsupported ncclDataType_t for size query");
            return 0;
    }
}

// Allgather on host memory using NCCL (used once for hostname exchange).
// This operates on a single in-place host buffer, unlike batchAllGatherIpcHandles
// which batches multiple IPC handles via a packed device buffer.
// Each rank contributes element_size bytes at offset rank * element_size.
static void ncclAllGatherHost(
    void* host_buffer,
    size_t element_size,
    int rank,
    int nRanks,
    ncclComm_t comm,
    cudaStream_t stream
) {
    const size_t total_size = element_size * nRanks;
    void* d_buffer;
    CUDA_CHECK(cudaMalloc(&d_buffer, total_size));
    CUDA_CHECK(cudaMemcpy(
        static_cast<uint8_t*>(d_buffer) + rank * element_size,
        static_cast<uint8_t*>(host_buffer) + rank * element_size,
        element_size, cudaMemcpyHostToDevice));
    NCCL_CHECK_RESULT(ncclAllGather(
        static_cast<uint8_t*>(d_buffer) + rank * element_size,
        d_buffer, element_size, ncclUint8, comm, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(host_buffer, d_buffer, total_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_buffer));
}

// NCCL barrier using AllReduce.
// If workspace is provided, it is used directly (must be at least sizeof(int) device bytes).
// Otherwise a temporary cudaMalloc/cudaFree pair is used as fallback.
static ncclResult_t ncclBarrier(ncclComm_t comm, cudaStream_t stream, void* workspace = nullptr) {
    int *nccl_barrier_var = nullptr;
    bool owns_memory = false;
    if (workspace) {
        nccl_barrier_var = static_cast<int*>(workspace);
    } else {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&nccl_barrier_var), sizeof(int)));
        owns_memory = true;
    }
    CUDA_CHECK(cudaMemset(nccl_barrier_var, 0, sizeof(int)));
    NCCL_CHECK_RESULT(ncclAllReduce(nccl_barrier_var, nccl_barrier_var, 1, ncclInt, ncclSum, comm, stream));
    CUDA_CHECK(cudaDeviceSynchronize());
    if (owns_memory) {
        CUDA_CHECK(cudaFree(nccl_barrier_var));
    }
    return ncclSuccess;
}

// Opaque struct definitions
struct ncclEpGroup {
    ncclComm_t comm;

    int nRanks;               // Total number of ranks (from ncclCommCount)
    int rank;                 // This rank's ID
    int nNodes;               // Number of nodes

    void* ep_workspace;       // Device workspace for EP operations
    int cuda_device_id;        // CUDA device ID
    int lsa_team_size;        // LSA team size: ncclTeamLsa(comm).nRanks
    int rdma_ranks;           // RDMA ranks (nRanks / lsa_team_size)
    int rdma_rank;            // RDMA rank (rank / lsa_team_size)
    void* rdma_buffer;
    ncclEpGroupConfig_t config;         // Stored configuration

    struct {
        // Split communicator: groups ranks with same local_rank across nodes (nNodes ranks each)
        ncclComm_t split_comm = nullptr;      // Sub-communicator for GIN (nNodes ranks)
        // Device communicator (single comm, multiple contexts)
        ncclDevComm_t* dcomms = nullptr;      // Host array of device communicators
        ncclDevComm_t* d_dcomms = nullptr;    // Device array of device communicators
        int num_comms = 0;                     // Number of communicators (always 1)
        int num_dcomms = 0;                    // Number of device comms
        int qps_per_rank = 0;                  // Total QPs (connections) per rank
        int num_ctx_per_comm = 0;              // Number of contexts per communicator

        // GIN memory base pointer and window
        void* gin_base_ptr = nullptr;         // Base pointer for all GIN memory
        ncclWindow_t nccl_window = {};        // Single registered window handle (pointer-sized)
        unsigned signals_base = 0;            // Base signal ID for dispatch
        unsigned combine_signal_offset = 0;   // Signal offset for combine operations
        int num_total_signals = 0;            // Total number of signals

        // Used by kernels to calculate actual addresses for RDMA puts
        size_t rdma_intra_node_red_token_offset = 0;
        size_t combine_rdma_inter_node_group_token_offset = 0;
        size_t rdma_intra_node_red_prob_offset = 0;
        size_t combine_rdma_inter_node_group_prob_offset = 0;
        size_t token_staging_offset = 0;
        size_t dense_prob_offset = 0;
        size_t scaling_factor_staging_offset = 0;

        // Layout: [NUM_LSA_TEAMS-1][BATCH_SIZE * bytes_per_entry]
        // bytes_per_entry = hidden * sizeof(TOKEN_DATA_TYPE) + prob_size + sf_size
        size_t rdma_send_staging_offset = 0;
        size_t rdma_inter_node_group_packed_offset = 0;  // Packed receive buffer (token+prob+sf)

        unsigned signals_tail_base = 0;         // Base signal ID for tail tracking (sender -> receiver)
        int num_max_rdma_chunked_send_tokens = HYBRIDEP_DISPATCH_RDMA_BATCH_SIZE;

    } gin_config;

    int num_local_experts;    // Number of local experts (num_experts / comm->nRanks)
    int max_recv_tokens;      // Worst-case receive capacity: nRanks * max_tokens_per_rank
    int hidden;               // Hidden size (token_size_bytes / ncclTypeSize(ncclBfloat16))
    unsigned int device_sm_count; // Number of SMs on the device
    unsigned int num_sms_ht; // Number of SMs to use for HT kernels

    // Custom allocator function pointers
    ncclEpAllocFn_t alloc_fn;
    ncclEpFreeFn_t free_fn;

    // Physical node properties (CUDA device assignment, IPC between co-located GPUs)
    int gpus_per_node;    // Physical GPUs per node (nRanks / nNodes)
    int rank_in_node;     // Per-node CUDA device ordinal (= cuda_device_id)
    int node_id;          // Physical node index (rank / gpus_per_node)

    // Kernel params (node-based for HT; lsa_rank_count == gpus_per_node)
    int lsa_rank_count;   // Ranks per node passed to HT kernels as num_ranks_per_node

    // NCCL device API
    size_t num_nccl_comms;
    std::vector<ncclComm_t> nccl_comms;
    ncclDevComm_t* nccl_dev_comms;
    ncclWindow_t* nccl_wins;
    int num_dispatch_signals;

    // HT buffers for intranode communication
    struct {
        // IPC-mapped buffer pointer arrays (fixed-size, indexed by local NVL rank)
        // Host arrays for population and cleanup
        void **dispatch_expert_output_token_buffer_ptrs;
        float **dispatch_expert_output_prob_buffer_ptrs;
        float **dispatch_expert_output_scaling_factor_buffer_ptrs;
        uint16_t **combine_expert_input_token_buffer_ptrs;
        float **combine_expert_input_prob_buffer_ptrs;

        // Local buffers (owned by this rank)
        void *expert_output_token;
        float *expert_output_prob;
        float *expert_output_scaling_factor;
        uint16_t *expert_input_token;
        float *expert_input_prob;

        // Sync flags (rank 0 allocates, others IPC-map)
        uint32_t *intra_node_write_completion_flags;
        uint32_t *combine_intra_node_write_completion_flags;
        // Grid barrier counter for fused device_sync in dispatch tail (per-rank, not IPC-shared)
        uint32_t *dispatch_grid_barrier_counter;
        // Host-side expected flag counters (replaces device-side update_expected_value_kernel)
        uint64_t host_dispatch_expected_rdma = 0;
        uint32_t host_dispatch_expected_intra = 0;
        uint64_t host_combine_expected_rdma = 0;
        uint32_t host_combine_expected_intra = 0;

        // RDMA buffers (multi-node only)
        uint64_t *rdma_inter_node_group_flags;
        uint16_t *rdma_intra_node_red_token;
        float *rdma_intra_node_red_prob;
        uint16_t *combine_rdma_inter_node_group_token;
        float *combine_rdma_inter_node_group_prob;
        uint64_t *combine_rdma_inter_node_group_flags;

        // Pre-registered dispatch buffers (group-level, allocated during Group Create)
        // These are pre-registered with GIN to avoid ~60ms registration overhead during dispatch
        void *token_staging_buffer;          // Pre-registered staging buffer for user tokens
        float *dense_prob_buffer;            // Pre-registered buffer for sparse→dense prob conversion
        float *scaling_factor_staging_buffer; // Pre-registered staging buffer for FP8 scaling factors

        // Merged IPC buffer (single cudaMalloc for all IPC-shared buffers)
        void* ipc_mega_buffer = nullptr;
        size_t ipc_mega_buffer_size = 0;
        size_t ipc_dispatch_token_offset = 0;
        size_t ipc_dispatch_prob_offset = 0;
        size_t ipc_combine_token_offset = 0;
        size_t ipc_combine_prob_offset = 0;

        void** peer_ipc_base_ptrs = nullptr;   // Opened IPC base pointers per peer (for cleanup)
        void* host_ptr_block = nullptr;        // Single cudaHostAlloc for all pointer arrays

        // Fabric-memory-based intranode (used when MNNVL LSA covers all ranks).
        // When enabled, replaces cudaIpc* with CUDA VMM Fabric Handles so that peers
        // in different hostnames but the same NVLink fabric domain can share buffers
        // directly via NVSwitch, without falling through to RDMA.
        bool use_fabric_memory = false;
        CUmemGenericAllocationHandle fabric_local_alloc = 0;           // local mega buffer alloc
        size_t fabric_aligned_size = 0;                                // mega buffer size rounded to granularity
        CUmemGenericAllocationHandle* fabric_peer_allocs = nullptr;    // [nRanks], own alloc at [rank]
        CUdeviceptr* fabric_peer_vas = nullptr;                        // [nRanks], VAs for unmap
        CUmemGenericAllocationHandle fabric_flags_alloc = 0;           // rank 0: own; others: imported
        CUdeviceptr fabric_flags_va = 0;                               // VA for flags (rank 0 own, others imported)
        size_t fabric_flags_aligned_size = 0;

        // Config
        bool initialized;
        bool internode_initialized;
    } ht_buffers;

    // FULLMESH (Phase 2) buffers: per-dest padded recv block + per-dest atomic
    // counter row, both FABRIC-exported so every rank in the MNNVL clique can
    // directly load / store them from its GPU. Indexing convention:
    //
    //   recv_buf_of[dest][src][slot]           // layed out at dest rank,
    //                                          // mapped at every src rank
    //   entry bytes = meta_bytes + hidden_bytes
    //   meta_bytes  = 8  (int32 src_rank + int32 src_token_id)
    //
    //   counter_row_of[dest][src]              // int32 atomic counter,
    //                                          // src rank atomicAdd 1 to
    //                                          // carve its slot in its own
    //                                          // [max_tokens_per_rank] block
    //
    // Sizes (total bytes, pre-granularity-align):
    //   recv:    nRanks * max_tokens_per_rank * (meta_bytes + hidden_bytes)
    //   counter: nRanks * sizeof(int32)
    //
    // Two independent fabric allocations keep the layout readable; the extra
    // handle-exchange cost is 64B per alloc (negligible) and granularity waste
    // on the counter row is one fabric page (~2 MiB), also negligible on HBM3e.
    struct {
        bool initialized = false;

        // Layout metadata (filled in by init_fullmesh_intranode_fabric).
        size_t meta_bytes = 0;
        size_t bytes_per_entry = 0;      // meta_bytes + hidden_bytes
        size_t bytes_per_src_block = 0;  // max_tokens_per_rank * bytes_per_entry
        size_t bytes_per_recv_buf = 0;   // nRanks * bytes_per_src_block
        size_t bytes_per_counter_row = 0;  // nRanks * sizeof(int32)

        // Combine layout: per-src block of [max_tokens][top_k][hidden_bytes].
        // No meta is stored here because src rank already owns (t, k), so the
        // dest push target is just the hidden slab. bytes_per_combine_buf is
        // computed from config at init time once top_k bound is known (we
        // upper-bound it to NCCL_EP_MAX_TOPK so the allocation is fixed per
        // group regardless of which handle is currently in flight).
        size_t max_topk_for_combine = 0;
        size_t bytes_per_combine_token = 0;  // top_k * hidden_bytes
        size_t bytes_per_combine_buf = 0;    // max_tokens * bytes_per_combine_token

        // Granularity-aligned sizes actually passed to cuMemCreate.
        size_t recv_aligned_size = 0;
        size_t counter_aligned_size = 0;
        size_t combine_aligned_size = 0;

        // Local-owned fabric allocations.
        CUmemGenericAllocationHandle recv_local_alloc = 0;
        CUdeviceptr                  recv_local_va   = 0;
        CUmemGenericAllocationHandle counter_local_alloc = 0;
        CUdeviceptr                  counter_local_va   = 0;
        // Combine: this rank's OWN combine_recv_buf (src viewpoint -- peers,
        // acting as dest, push FFN outputs back here). Shape
        //   [max_tokens_per_rank][max_topk_for_combine][hidden_bytes]
        CUmemGenericAllocationHandle combine_local_alloc = 0;
        CUdeviceptr                  combine_local_va   = 0;

        // Peer mappings (indexed by rank; self entry points at local_va).
        // peer_recv_vas[dest]     = peer's dispatch recv buffer (src pushes here)
        // peer_counter_vas[dest]  = peer's counter row           (src atomicAdd here)
        // peer_combine_vas[src]   = peer's combine_recv_buf      (dest pushes here)
        // peer_*_allocs hold the imported handles for destroy()  cleanup.
        CUdeviceptr*                  peer_recv_vas = nullptr;
        CUdeviceptr*                  peer_counter_vas = nullptr;
        CUdeviceptr*                  peer_combine_vas = nullptr;
        CUmemGenericAllocationHandle* peer_recv_allocs = nullptr;
        CUmemGenericAllocationHandle* peer_counter_allocs = nullptr;
        CUmemGenericAllocationHandle* peer_combine_allocs = nullptr;

        // Device-side pointer arrays for one-indirection access from kernels.
        void** peer_recv_vas_dev = nullptr;        // [nRanks]
        void** peer_counter_vas_dev = nullptr;     // [nRanks]
        void** peer_combine_vas_dev = nullptr;     // [nRanks]
    } fullmesh_buffers;

    // Constructor to properly initialize all members
    ncclEpGroup() :
        comm(nullptr),
        nRanks(0),
        rank(0),
        nNodes(0),
        ep_workspace(nullptr),
        cuda_device_id(0),
        rdma_ranks(0),
        rdma_rank(0),
        rdma_buffer(nullptr),
        config{},
        num_local_experts(0),
        max_recv_tokens(0),
        hidden(0),
        device_sm_count(0),
        num_sms_ht(0),
        alloc_fn(nullptr),
        free_fn(nullptr),
        gpus_per_node(0),
        rank_in_node(0),
        node_id(0),
        lsa_rank_count(0),
        num_nccl_comms(0),
        nccl_comms{},
        nccl_dev_comms(nullptr),
        nccl_wins(nullptr),
        num_dispatch_signals(0),
        ht_buffers{},
        fullmesh_buffers{} {}
};

// Batch allgather of multiple IPC handles in a single NCCL collective.
// Reduces N separate cudaMalloc/ncclAllGather/cudaFree cycles to just 1.
// local_handles: array of num_handles local IPC handle pointers (each CUDA_IPC_HANDLE_SIZE bytes)
// all_handles:   array of num_handles output buffers (each CUDA_IPC_HANDLE_SIZE * nRanks bytes)
static void batchAllGatherIpcHandles(
    const void* const* local_handles,
    void** all_handles,
    int num_handles,
    int rank,
    int nRanks,
    ncclComm_t comm,
    cudaStream_t stream
) {
    const size_t per_rank_size = static_cast<size_t>(num_handles) * CUDA_IPC_HANDLE_SIZE;
    const size_t total_size = per_rank_size * nRanks;
    void* d_buffer;
    CUDA_CHECK(cudaMalloc(&d_buffer, total_size));

    // Pack all local handles into a contiguous host buffer, then copy to device
    std::unique_ptr<uint8_t[]> pack_buf(new uint8_t[per_rank_size]);
    for (int i = 0; i < num_handles; i++) {
        memcpy(pack_buf.get() + i * CUDA_IPC_HANDLE_SIZE,
               local_handles[i], CUDA_IPC_HANDLE_SIZE);
    }
    CUDA_CHECK(cudaMemcpy(
        static_cast<uint8_t*>(d_buffer) + rank * per_rank_size,
        pack_buf.get(), per_rank_size, cudaMemcpyHostToDevice));

    // Single NCCL AllGather for all handles
    NCCL_CHECK_RESULT(ncclAllGather(
        static_cast<uint8_t*>(d_buffer) + rank * per_rank_size,
        d_buffer, per_rank_size, ncclUint8, comm, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Copy all results back to host, then unpack per-handle
    std::unique_ptr<uint8_t[]> host_buf(new uint8_t[total_size]);
    CUDA_CHECK(cudaMemcpy(host_buf.get(), d_buffer, total_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_buffer));

    // Deinterleave: for each handle i, extract rank r's copy from the packed layout
    for (int i = 0; i < num_handles; i++) {
        uint8_t* out = static_cast<uint8_t*>(all_handles[i]);
        for (int r = 0; r < nRanks; r++) {
            memcpy(out + r * CUDA_IPC_HANDLE_SIZE,
                   host_buf.get() + r * per_rank_size + i * CUDA_IPC_HANDLE_SIZE,
                   CUDA_IPC_HANDLE_SIZE);
        }
    }
}

// HT Intranode Initialization (adapted for public NCCL APIs)
static ncclResult_t init_hybridep_intranode(ncclEpGroup_t ep_group,
                                    const ncclEpGroupConfig_t* in_config,
                                    cudaStream_t stream)
{
    // If MNNVL LSA covers all ranks, use fabric memory handles instead of cudaIPC.
    if (ep_group->ht_buffers.use_fabric_memory) {
        return init_hybridep_intranode_fabric(ep_group, in_config, stream);
    }

    ncclComm_t comm = ep_group->comm;
    int nRanks = ep_group->nRanks;
    int rank = ep_group->rank;
    // Node semantics: IPC, buffer sizing, and indexing use per-node device ordinals.
    // Note: for MNNVL this only covers same-node peers; cross-node MNNVL HT is not supported.
    int gpus_per_node  = ep_group->gpus_per_node;
    int rank_in_node   = ep_group->rank_in_node;
    int node_id        = ep_group->node_id;
    int hidden = ep_group->hidden;
    int num_local_experts = ep_group->num_local_experts;
    int max_recv_tokens = ep_group->max_recv_tokens;

    ep_group->ht_buffers.initialized = false;

    // Enable P2P access between GPUs on the same physical node.
    // cudaDeviceCanAccessPeer/cudaDeviceEnablePeerAccess operate on per-node device ordinals
    // (rank_in_node, gpus_per_node) — these are node concepts, not LSA concepts.
    // Cross-node MNNVL P2P is handled by the NVLink fabric through IPC handles; it does not
    // need cudaDeviceEnablePeerAccess and cannot be probed with cudaDeviceCanAccessPeer since
    // remote node device ordinals are not valid on the local node.
    // TODO: replace this loop with a NCCL API that queries P2P capability (e.g. ncclCommQueryProperties).
    for (int i = 0; i < gpus_per_node; i++) {
        if (i == rank_in_node) continue;
        int can_p2p = 0;
        CUDA_CHECK(cudaDeviceCanAccessPeer(&can_p2p, rank_in_node, i));
        if (can_p2p) {
            cudaError_t err = cudaDeviceEnablePeerAccess(i, 0);
            if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
                fprintf(stderr, "HT: Failed to enable P2P from GPU %d to GPU %d\n", rank_in_node, i);
            }
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // =========================================================================
    // Phase 1: Allocate all buffers upfront (before batch IPC exchange)
    // =========================================================================

    // Consolidated IPC mega-buffer: single cudaMalloc for all 4 IPC-shared buffers.
    // Expert-prob buffers sized by gpus_per_node (node semantics).
    auto align_ipc = [](size_t s) -> size_t { return (s + 255) & ~size_t(255); };

    size_t expert_output_token_sz = static_cast<size_t>(max_recv_tokens) * hidden * sizeof(uint16_t);
    size_t expert_output_prob_sz = static_cast<size_t>(max_recv_tokens) * num_local_experts * gpus_per_node * sizeof(float);
    size_t expert_input_token_sz = static_cast<size_t>(max_recv_tokens) * hidden * sizeof(uint16_t);
    size_t expert_input_prob_sz = static_cast<size_t>(max_recv_tokens) * num_local_experts * gpus_per_node * sizeof(float);

    size_t dispatch_token_aligned = align_ipc(expert_output_token_sz);
    size_t dispatch_prob_aligned  = align_ipc(expert_output_prob_sz);
    size_t combine_token_aligned  = align_ipc(expert_input_token_sz);
    size_t combine_prob_aligned   = align_ipc(expert_input_prob_sz);

    size_t mega_sz = dispatch_token_aligned + dispatch_prob_aligned
                   + combine_token_aligned + combine_prob_aligned;
    CUDA_CHECK(cudaMalloc(&ep_group->ht_buffers.ipc_mega_buffer, mega_sz));
    ep_group->ht_buffers.ipc_mega_buffer_size = mega_sz;

    uint8_t* mega_base = static_cast<uint8_t*>(ep_group->ht_buffers.ipc_mega_buffer);
    ep_group->ht_buffers.ipc_dispatch_token_offset = 0;
    ep_group->ht_buffers.expert_output_token = mega_base;

    ep_group->ht_buffers.ipc_dispatch_prob_offset = dispatch_token_aligned;
    ep_group->ht_buffers.expert_output_prob = reinterpret_cast<float*>(mega_base + dispatch_token_aligned);

    ep_group->ht_buffers.ipc_combine_token_offset = dispatch_token_aligned + dispatch_prob_aligned;
    ep_group->ht_buffers.expert_input_token = reinterpret_cast<uint16_t*>(
        mega_base + dispatch_token_aligned + dispatch_prob_aligned);

    ep_group->ht_buffers.ipc_combine_prob_offset = dispatch_token_aligned + dispatch_prob_aligned + combine_token_aligned;
    ep_group->ht_buffers.expert_input_prob = reinterpret_cast<float*>(
        mega_base + dispatch_token_aligned + dispatch_prob_aligned + combine_token_aligned);

    // Host pointer arrays indexed by per-node position (one entry per node peer).
    size_t host_block_sz = sizeof(void*) * gpus_per_node
                         + sizeof(float*) * gpus_per_node
                         + sizeof(uint16_t*) * gpus_per_node
                         + sizeof(float*) * gpus_per_node;
    CUDA_CHECK(cudaHostAlloc(&ep_group->ht_buffers.host_ptr_block, host_block_sz, cudaHostAllocMapped));

    uint8_t* hptr = static_cast<uint8_t*>(ep_group->ht_buffers.host_ptr_block);
    ep_group->ht_buffers.dispatch_expert_output_token_buffer_ptrs = reinterpret_cast<void**>(hptr);
    hptr += sizeof(void*) * gpus_per_node;
    ep_group->ht_buffers.dispatch_expert_output_prob_buffer_ptrs = reinterpret_cast<float**>(hptr);
    hptr += sizeof(float*) * gpus_per_node;
    ep_group->ht_buffers.combine_expert_input_token_buffer_ptrs = reinterpret_cast<uint16_t**>(hptr);
    hptr += sizeof(uint16_t*) * gpus_per_node;
    ep_group->ht_buffers.combine_expert_input_prob_buffer_ptrs = reinterpret_cast<float**>(hptr);

    // Merged completion flags: rank_in_node 0 is the primary within the node.
    if (rank_in_node == 0) {
        uint32_t* completion_flags_base;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&completion_flags_base), 2 * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemsetAsync(completion_flags_base, 0, 2 * sizeof(uint32_t), stream));
        ep_group->ht_buffers.intra_node_write_completion_flags = completion_flags_base;
        ep_group->ht_buffers.combine_intra_node_write_completion_flags = completion_flags_base + 1;
    }

    // Dispatch grid barrier counter (local to each rank, NOT IPC-shared)
    {
        uint32_t* grid_barrier_base;
        CUDA_CHECK(ep_group->alloc_fn(reinterpret_cast<void**>(&grid_barrier_base), sizeof(uint32_t)));
        CUDA_CHECK(cudaMemsetAsync(grid_barrier_base, 0, sizeof(uint32_t), stream));
        ep_group->ht_buffers.dispatch_grid_barrier_counter = grid_barrier_base;
    }

    // =========================================================================
    // Phase 2: Get IPC handles and batch-allgather in a single NCCL collective
    // Consolidated: [0] = mega buffer (token+prob+combine), [1] = completion flags
    // =========================================================================
    constexpr int NUM_IPC_HANDLES = 2;
    cudaIpcMemHandle_t local_handles[NUM_IPC_HANDLES];
    memset(local_handles, 0, sizeof(local_handles));

    CUDA_CHECK(cudaIpcGetMemHandle(&local_handles[0], ep_group->ht_buffers.ipc_mega_buffer));
    if (rank_in_node == 0) {
        CUDA_CHECK(cudaIpcGetMemHandle(&local_handles[1], ep_group->ht_buffers.intra_node_write_completion_flags));
    }

    std::unique_ptr<uint8_t[]> all_handles_buf[NUM_IPC_HANDLES];
    for (int i = 0; i < NUM_IPC_HANDLES; i++) {
        all_handles_buf[i].reset(new uint8_t[CUDA_IPC_HANDLE_SIZE * nRanks]);
    }

    const void* local_handle_ptrs[NUM_IPC_HANDLES];
    void* all_handle_ptrs[NUM_IPC_HANDLES];
    for (int i = 0; i < NUM_IPC_HANDLES; i++) {
        local_handle_ptrs[i] = &local_handles[i];
        all_handle_ptrs[i] = all_handles_buf[i].get();
    }

    batchAllGatherIpcHandles(local_handle_ptrs, all_handle_ptrs, NUM_IPC_HANDLES,
                             rank, nRanks, comm, stream);

    // =========================================================================
    // Phase 3: Open IPC handles for same-node peers (node semantics).
    // Indexed by rank_in_node; node_id identifies which physical node this rank belongs to.
    // =========================================================================

    // Track opened IPC base pointers for cleanup (one per node peer)
    CUDA_CHECK(cudaHostAlloc(&ep_group->ht_buffers.peer_ipc_base_ptrs,
        sizeof(void*) * gpus_per_node, cudaHostAllocMapped));
    memset(ep_group->ht_buffers.peer_ipc_base_ptrs, 0, sizeof(void*) * gpus_per_node);

    // Open 1 mega-buffer IPC handle per node peer, derive 4 pointers via offsets
    for (int i = 0; i < gpus_per_node; i++) {
        if (i == rank_in_node) {
            ep_group->ht_buffers.dispatch_expert_output_token_buffer_ptrs[i] =
                ep_group->ht_buffers.expert_output_token;
            ep_group->ht_buffers.dispatch_expert_output_prob_buffer_ptrs[i] =
                ep_group->ht_buffers.expert_output_prob;
            ep_group->ht_buffers.combine_expert_input_token_buffer_ptrs[i] =
                ep_group->ht_buffers.expert_input_token;
            ep_group->ht_buffers.combine_expert_input_prob_buffer_ptrs[i] =
                ep_group->ht_buffers.expert_input_prob;
        } else {
            int peer_global = node_id * gpus_per_node + i;
            cudaIpcMemHandle_t peer_handle;
            memcpy(&peer_handle, all_handles_buf[0].get() + peer_global * CUDA_IPC_HANDLE_SIZE,
                   CUDA_IPC_HANDLE_SIZE);
            void* peer_base;
            CUDA_CHECK(cudaIpcOpenMemHandle(&peer_base, peer_handle, cudaIpcMemLazyEnablePeerAccess));
            ep_group->ht_buffers.peer_ipc_base_ptrs[i] = peer_base;

            uint8_t* pb = static_cast<uint8_t*>(peer_base);
            ep_group->ht_buffers.dispatch_expert_output_token_buffer_ptrs[i] =
                pb + ep_group->ht_buffers.ipc_dispatch_token_offset;
            ep_group->ht_buffers.dispatch_expert_output_prob_buffer_ptrs[i] =
                reinterpret_cast<float*>(pb + ep_group->ht_buffers.ipc_dispatch_prob_offset);
            ep_group->ht_buffers.combine_expert_input_token_buffer_ptrs[i] =
                reinterpret_cast<uint16_t*>(pb + ep_group->ht_buffers.ipc_combine_token_offset);
            ep_group->ht_buffers.combine_expert_input_prob_buffer_ptrs[i] =
                reinterpret_cast<float*>(pb + ep_group->ht_buffers.ipc_combine_prob_offset);
        }
    }

    // Merged completion flags: non-rank_in_node-0 ranks IPC-map from node-local rank 0.
    if (rank_in_node != 0) {
        int node_rank0_global = node_id * gpus_per_node;
        cudaIpcMemHandle_t rank0_handle;
        memcpy(&rank0_handle, all_handles_buf[1].get() + node_rank0_global * CUDA_IPC_HANDLE_SIZE, CUDA_IPC_HANDLE_SIZE);
        void* ptr;
        CUDA_CHECK(cudaIpcOpenMemHandle(&ptr, rank0_handle, cudaIpcMemLazyEnablePeerAccess));
        ep_group->ht_buffers.intra_node_write_completion_flags = static_cast<uint32_t*>(ptr);
        ep_group->ht_buffers.combine_intra_node_write_completion_flags = static_cast<uint32_t*>(ptr) + 1;
    }

    ep_group->ht_buffers.initialized = true;
    CUDA_CHECK(cudaDeviceSynchronize());

    return ncclSuccess;
}

// HT Intranode Cleanup
static ncclResult_t destroy_hybridep_intranode(ncclEpGroup_t ep_group) {
    if (!ep_group->ht_buffers.initialized) return ncclSuccess;
    if (ep_group->ht_buffers.use_fabric_memory) {
        return destroy_hybridep_intranode_fabric(ep_group);
    }

    // Node semantics: IPC cleanup mirrors Phase 3 — indexed by per-node position.
    int rank_in_node  = ep_group->rank_in_node;
    int gpus_per_node = ep_group->gpus_per_node;

    // Close IPC handles: 1 mega-buffer handle per node peer + 1 completion flag handle
    for (int i = 0; i < gpus_per_node; i++) {
        if (i != rank_in_node && ep_group->ht_buffers.peer_ipc_base_ptrs &&
            ep_group->ht_buffers.peer_ipc_base_ptrs[i]) {
            cudaIpcCloseMemHandle(ep_group->ht_buffers.peer_ipc_base_ptrs[i]);
        }
    }

    // Close completion flag IPC handle (non-rank_in_node-0)
    if (rank_in_node != 0) {
        if (ep_group->ht_buffers.intra_node_write_completion_flags) {
            cudaIpcCloseMemHandle(ep_group->ht_buffers.intra_node_write_completion_flags);
        }
    }

    // Free consolidated IPC mega-buffer (replaces 4 individual cudaFree calls)
    if (ep_group->ht_buffers.ipc_mega_buffer) {
        cudaFree(ep_group->ht_buffers.ipc_mega_buffer);
        ep_group->ht_buffers.ipc_mega_buffer = nullptr;
        ep_group->ht_buffers.expert_output_token = nullptr;
        ep_group->ht_buffers.expert_output_prob = nullptr;
        ep_group->ht_buffers.expert_input_token = nullptr;
        ep_group->ht_buffers.expert_input_prob = nullptr;
    }
    if (ep_group->ht_buffers.expert_output_scaling_factor) {
        ep_group->free_fn(ep_group->ht_buffers.expert_output_scaling_factor);
    }
    // Free dispatch grid barrier counter
    if (ep_group->ht_buffers.dispatch_grid_barrier_counter) {
        ep_group->free_fn(ep_group->ht_buffers.dispatch_grid_barrier_counter);
    }

    // Free merged completion flags (rank_in_node 0 only; base pointer covers both dispatch and combine)
    if (rank_in_node == 0) {
        if (ep_group->ht_buffers.intra_node_write_completion_flags) {
            cudaFree(ep_group->ht_buffers.intra_node_write_completion_flags);
        }
    }

    // Free consolidated host pointer block and peer IPC tracking array
    if (ep_group->ht_buffers.dispatch_expert_output_scaling_factor_buffer_ptrs) {
        cudaFreeHost(ep_group->ht_buffers.dispatch_expert_output_scaling_factor_buffer_ptrs);
    }
    if (ep_group->ht_buffers.host_ptr_block) {
        cudaFreeHost(ep_group->ht_buffers.host_ptr_block);
        ep_group->ht_buffers.host_ptr_block = nullptr;
    }
    if (ep_group->ht_buffers.peer_ipc_base_ptrs) {
        cudaFreeHost(ep_group->ht_buffers.peer_ipc_base_ptrs);
        ep_group->ht_buffers.peer_ipc_base_ptrs = nullptr;
    }

    ep_group->ht_buffers.initialized = false;
    return ncclSuccess;
}

// ============================================================================
// HT Intranode Fabric (MNNVL full-coverage variant)
// ----------------------------------------------------------------------------
// When the NCCL LSA team covers all ranks (NVL72 intra-pod), we treat the
// whole EP group as a single "fabric node" and share HT staging buffers via
// CUDA VMM Fabric Memory Handles instead of cudaIPC. Fabric handles work for
// all peers in the NVLink fabric domain (both same-hostname and different-
// hostname, since NVSwitch is a unified fabric), so HT kernels can do direct
// peer load/store for the full EP group without falling back to RDMA.
//
// Key differences vs init_hybridep_intranode (cudaIPC variant):
//   * Peer-pointer arrays are sized by nRanks (== lsa_rank_count) instead of
//     gpus_per_node. In this code path they are equal by construction, but the
//     semantic is "fabric domain ranks".
//   * All allocations use cuMemCreate(CU_MEM_HANDLE_TYPE_FABRIC), then
//     cuMemAddressReserve + cuMemMap + cuMemSetAccess. Peer mappings use
//     cuMemImportFromShareableHandle + cuMemMap + cuMemSetAccess.
//   * Handle exchange reuses ncclAllGatherHost with the 64-byte
//     CUmemFabricHandle (same size as cudaIpcMemHandle_t).
//   * cudaDeviceEnablePeerAccess is NOT called because fabric memory uses a
//     different access path controlled by cuMemSetAccess per VA range.
// ============================================================================
static size_t fabric_round_up(size_t sz, size_t granularity) {
    if (granularity == 0) granularity = 1;
    return (sz + granularity - 1) & ~(granularity - 1);
}

static ncclResult_t init_hybridep_intranode_fabric(ncclEpGroup_t ep_group,
                                    const ncclEpGroupConfig_t* in_config,
                                    cudaStream_t stream)
{
    (void)in_config;
    ncclComm_t comm = ep_group->comm;
    int nRanks = ep_group->nRanks;
    int rank = ep_group->rank;
    // In fabric mode, rank_in_node == rank, gpus_per_node == nRanks (forced in ncclEpCreateGroup).
    int n_fabric_ranks = ep_group->lsa_rank_count; // == nRanks
    int hidden = ep_group->hidden;
    int num_local_experts = ep_group->num_local_experts;
    int max_recv_tokens = ep_group->max_recv_tokens;

    ep_group->ht_buffers.initialized = false;

    // Compute layout (same as IPC path, only storage backing differs).
    auto align_ipc = [](size_t s) -> size_t { return (s + 255) & ~size_t(255); };
    size_t expert_output_token_sz = static_cast<size_t>(max_recv_tokens) * hidden * sizeof(uint16_t);
    size_t expert_output_prob_sz  = static_cast<size_t>(max_recv_tokens) * num_local_experts * n_fabric_ranks * sizeof(float);
    size_t expert_input_token_sz  = static_cast<size_t>(max_recv_tokens) * hidden * sizeof(uint16_t);
    size_t expert_input_prob_sz   = static_cast<size_t>(max_recv_tokens) * num_local_experts * n_fabric_ranks * sizeof(float);

    size_t dispatch_token_aligned = align_ipc(expert_output_token_sz);
    size_t dispatch_prob_aligned  = align_ipc(expert_output_prob_sz);
    size_t combine_token_aligned  = align_ipc(expert_input_token_sz);
    size_t combine_prob_aligned   = align_ipc(expert_input_prob_sz);

    size_t mega_sz = dispatch_token_aligned + dispatch_prob_aligned
                   + combine_token_aligned + combine_prob_aligned;

    // Setup fabric allocation prop and compute granularity.
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = ep_group->cuda_device_id;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;

    size_t granularity = 0;
    CU_CHECK(cuMemGetAllocationGranularity(&granularity, &prop,
                                           CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    size_t aligned_mega = fabric_round_up(mega_sz, granularity);
    ep_group->ht_buffers.fabric_aligned_size = aligned_mega;

    // --- Phase 1: allocate local mega buffer via fabric handle, map into local VA ---
    CUmemGenericAllocationHandle local_alloc = 0;
    CU_CHECK(cuMemCreate(&local_alloc, aligned_mega, &prop, 0));
    ep_group->ht_buffers.fabric_local_alloc = local_alloc;

    CUdeviceptr local_va = 0;
    CU_CHECK(cuMemAddressReserve(&local_va, aligned_mega, granularity, 0, 0));
    CU_CHECK(cuMemMap(local_va, aligned_mega, 0, local_alloc, 0));

    CUmemAccessDesc access = {};
    access.location = prop.location;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CU_CHECK(cuMemSetAccess(local_va, aligned_mega, &access, 1));

    ep_group->ht_buffers.ipc_mega_buffer = reinterpret_cast<void*>(local_va);
    ep_group->ht_buffers.ipc_mega_buffer_size = aligned_mega;

    uint8_t* mega_base = reinterpret_cast<uint8_t*>(local_va);
    ep_group->ht_buffers.ipc_dispatch_token_offset = 0;
    ep_group->ht_buffers.expert_output_token = mega_base;
    ep_group->ht_buffers.ipc_dispatch_prob_offset = dispatch_token_aligned;
    ep_group->ht_buffers.expert_output_prob = reinterpret_cast<float*>(mega_base + dispatch_token_aligned);
    ep_group->ht_buffers.ipc_combine_token_offset = dispatch_token_aligned + dispatch_prob_aligned;
    ep_group->ht_buffers.expert_input_token = reinterpret_cast<uint16_t*>(
        mega_base + dispatch_token_aligned + dispatch_prob_aligned);
    ep_group->ht_buffers.ipc_combine_prob_offset = dispatch_token_aligned + dispatch_prob_aligned + combine_token_aligned;
    ep_group->ht_buffers.expert_input_prob = reinterpret_cast<float*>(
        mega_base + dispatch_token_aligned + dispatch_prob_aligned + combine_token_aligned);

    // Host peer-pointer block (sized by n_fabric_ranks == nRanks).
    size_t host_block_sz = sizeof(void*) * n_fabric_ranks
                         + sizeof(float*) * n_fabric_ranks
                         + sizeof(uint16_t*) * n_fabric_ranks
                         + sizeof(float*) * n_fabric_ranks;
    CUDACHECK_RET(cudaHostAlloc(&ep_group->ht_buffers.host_ptr_block, host_block_sz, cudaHostAllocMapped));
    uint8_t* hptr = static_cast<uint8_t*>(ep_group->ht_buffers.host_ptr_block);
    ep_group->ht_buffers.dispatch_expert_output_token_buffer_ptrs = reinterpret_cast<void**>(hptr);
    hptr += sizeof(void*) * n_fabric_ranks;
    ep_group->ht_buffers.dispatch_expert_output_prob_buffer_ptrs = reinterpret_cast<float**>(hptr);
    hptr += sizeof(float*) * n_fabric_ranks;
    ep_group->ht_buffers.combine_expert_input_token_buffer_ptrs = reinterpret_cast<uint16_t**>(hptr);
    hptr += sizeof(uint16_t*) * n_fabric_ranks;
    ep_group->ht_buffers.combine_expert_input_prob_buffer_ptrs = reinterpret_cast<float**>(hptr);

    // Completion flags (2 uint32_t). Global rank 0 owns; others import fabric handle.
    size_t flags_bytes = 2 * sizeof(uint32_t);
    size_t flags_granularity = granularity;
    size_t flags_aligned = fabric_round_up(flags_bytes, flags_granularity);
    ep_group->ht_buffers.fabric_flags_aligned_size = flags_aligned;
    if (rank == 0) {
        CUmemGenericAllocationHandle flags_alloc = 0;
        CU_CHECK(cuMemCreate(&flags_alloc, flags_aligned, &prop, 0));
        ep_group->ht_buffers.fabric_flags_alloc = flags_alloc;
        CUdeviceptr flags_va = 0;
        CU_CHECK(cuMemAddressReserve(&flags_va, flags_aligned, flags_granularity, 0, 0));
        CU_CHECK(cuMemMap(flags_va, flags_aligned, 0, flags_alloc, 0));
        CU_CHECK(cuMemSetAccess(flags_va, flags_aligned, &access, 1));
        ep_group->ht_buffers.fabric_flags_va = flags_va;
        CUDACHECK_RET(cudaMemsetAsync(reinterpret_cast<void*>(flags_va), 0, flags_bytes, stream));
        ep_group->ht_buffers.intra_node_write_completion_flags = reinterpret_cast<uint32_t*>(flags_va);
        ep_group->ht_buffers.combine_intra_node_write_completion_flags = reinterpret_cast<uint32_t*>(flags_va) + 1;
    }

    // Grid barrier counter (local, not shared).
    {
        uint32_t* grid_barrier_base;
        CUDACHECK_RET(ep_group->alloc_fn(reinterpret_cast<void**>(&grid_barrier_base), sizeof(uint32_t)));
        CUDACHECK_RET(cudaMemsetAsync(grid_barrier_base, 0, sizeof(uint32_t), stream));
        ep_group->ht_buffers.dispatch_grid_barrier_counter = grid_barrier_base;
    }

    // --- Phase 2: export local fabric handle, allgather to all ranks ---
    CUmemFabricHandle local_mega_fh = {};
    CU_CHECK(cuMemExportToShareableHandle(&local_mega_fh, local_alloc,
                                          CU_MEM_HANDLE_TYPE_FABRIC, 0));
    CUmemFabricHandle local_flags_fh = {};
    if (rank == 0) {
        CU_CHECK(cuMemExportToShareableHandle(&local_flags_fh,
                                              ep_group->ht_buffers.fabric_flags_alloc,
                                              CU_MEM_HANDLE_TYPE_FABRIC, 0));
    }

    static_assert(sizeof(CUmemFabricHandle) == 64, "CUmemFabricHandle expected to be 64 bytes");

    // AllGather the two handles together (mega + flags). Use host buffer path.
    const size_t FH_BYTES = sizeof(CUmemFabricHandle);
    std::unique_ptr<uint8_t[]> gathered(new uint8_t[2 * FH_BYTES * nRanks]);
    std::memset(gathered.get(), 0, 2 * FH_BYTES * nRanks);
    // Place this rank's handles at its slot.
    std::memcpy(gathered.get() + rank * 2 * FH_BYTES, &local_mega_fh, FH_BYTES);
    std::memcpy(gathered.get() + rank * 2 * FH_BYTES + FH_BYTES, &local_flags_fh, FH_BYTES);
    ncclAllGatherHost(gathered.get(), 2 * FH_BYTES, rank, nRanks, comm, stream);

    // --- Phase 3: import peer handles, map into local VA per peer ---
    CUDACHECK_RET(cudaHostAlloc(&ep_group->ht_buffers.peer_ipc_base_ptrs,
                                sizeof(void*) * n_fabric_ranks, cudaHostAllocMapped));
    std::memset(ep_group->ht_buffers.peer_ipc_base_ptrs, 0, sizeof(void*) * n_fabric_ranks);

    ep_group->ht_buffers.fabric_peer_allocs =
        static_cast<CUmemGenericAllocationHandle*>(std::calloc(n_fabric_ranks, sizeof(CUmemGenericAllocationHandle)));
    ep_group->ht_buffers.fabric_peer_vas =
        static_cast<CUdeviceptr*>(std::calloc(n_fabric_ranks, sizeof(CUdeviceptr)));
    if (!ep_group->ht_buffers.fabric_peer_allocs || !ep_group->ht_buffers.fabric_peer_vas) {
        fprintf(stderr, "init_hybridep_intranode_fabric: calloc failed for peer arrays\n");
        return ncclSystemError;
    }

    for (int i = 0; i < n_fabric_ranks; i++) {
        if (i == rank) {
            // Own buffer already mapped.
            ep_group->ht_buffers.fabric_peer_vas[i] = local_va;
            ep_group->ht_buffers.fabric_peer_allocs[i] = 0; // sentinel: don't release own handle via peer cleanup
            ep_group->ht_buffers.peer_ipc_base_ptrs[i] = reinterpret_cast<void*>(local_va);
            ep_group->ht_buffers.dispatch_expert_output_token_buffer_ptrs[i] = ep_group->ht_buffers.expert_output_token;
            ep_group->ht_buffers.dispatch_expert_output_prob_buffer_ptrs[i]  = ep_group->ht_buffers.expert_output_prob;
            ep_group->ht_buffers.combine_expert_input_token_buffer_ptrs[i]   = ep_group->ht_buffers.expert_input_token;
            ep_group->ht_buffers.combine_expert_input_prob_buffer_ptrs[i]    = ep_group->ht_buffers.expert_input_prob;
            continue;
        }
        CUmemFabricHandle peer_fh;
        std::memcpy(&peer_fh, gathered.get() + i * 2 * FH_BYTES, FH_BYTES);

        CUmemGenericAllocationHandle peer_alloc = 0;
        CU_CHECK(cuMemImportFromShareableHandle(&peer_alloc, &peer_fh, CU_MEM_HANDLE_TYPE_FABRIC));
        ep_group->ht_buffers.fabric_peer_allocs[i] = peer_alloc;

        CUdeviceptr peer_va = 0;
        CU_CHECK(cuMemAddressReserve(&peer_va, aligned_mega, granularity, 0, 0));
        CU_CHECK(cuMemMap(peer_va, aligned_mega, 0, peer_alloc, 0));
        CU_CHECK(cuMemSetAccess(peer_va, aligned_mega, &access, 1));
        ep_group->ht_buffers.fabric_peer_vas[i] = peer_va;

        uint8_t* pb = reinterpret_cast<uint8_t*>(peer_va);
        ep_group->ht_buffers.peer_ipc_base_ptrs[i] = pb;
        ep_group->ht_buffers.dispatch_expert_output_token_buffer_ptrs[i] =
            pb + ep_group->ht_buffers.ipc_dispatch_token_offset;
        ep_group->ht_buffers.dispatch_expert_output_prob_buffer_ptrs[i] =
            reinterpret_cast<float*>(pb + ep_group->ht_buffers.ipc_dispatch_prob_offset);
        ep_group->ht_buffers.combine_expert_input_token_buffer_ptrs[i] =
            reinterpret_cast<uint16_t*>(pb + ep_group->ht_buffers.ipc_combine_token_offset);
        ep_group->ht_buffers.combine_expert_input_prob_buffer_ptrs[i] =
            reinterpret_cast<float*>(pb + ep_group->ht_buffers.ipc_combine_prob_offset);
    }

    // Non-rank-0 ranks import rank 0's flags fabric handle.
    if (rank != 0) {
        CUmemFabricHandle rank0_flags_fh;
        std::memcpy(&rank0_flags_fh, gathered.get() + 0 * 2 * FH_BYTES + FH_BYTES, FH_BYTES);
        CUmemGenericAllocationHandle flags_alloc_imp = 0;
        CU_CHECK(cuMemImportFromShareableHandle(&flags_alloc_imp, &rank0_flags_fh, CU_MEM_HANDLE_TYPE_FABRIC));
        ep_group->ht_buffers.fabric_flags_alloc = flags_alloc_imp;

        CUdeviceptr flags_va = 0;
        CU_CHECK(cuMemAddressReserve(&flags_va, flags_aligned, flags_granularity, 0, 0));
        CU_CHECK(cuMemMap(flags_va, flags_aligned, 0, flags_alloc_imp, 0));
        CU_CHECK(cuMemSetAccess(flags_va, flags_aligned, &access, 1));
        ep_group->ht_buffers.fabric_flags_va = flags_va;
        ep_group->ht_buffers.intra_node_write_completion_flags =
            reinterpret_cast<uint32_t*>(flags_va);
        ep_group->ht_buffers.combine_intra_node_write_completion_flags =
            reinterpret_cast<uint32_t*>(flags_va) + 1;
    }

    ep_group->ht_buffers.initialized = true;
    CUDACHECK_RET(cudaDeviceSynchronize());
    return ncclSuccess;
}

static ncclResult_t destroy_hybridep_intranode_fabric(ncclEpGroup_t ep_group) {
    int nRanks = ep_group->nRanks;
    int rank = ep_group->rank;
    size_t aligned_mega = ep_group->ht_buffers.fabric_aligned_size;
    size_t flags_aligned = ep_group->ht_buffers.fabric_flags_aligned_size;

    // Unmap and release peers.
    if (ep_group->ht_buffers.fabric_peer_vas) {
        for (int i = 0; i < nRanks; i++) {
            if (i == rank) continue;
            CUdeviceptr va = ep_group->ht_buffers.fabric_peer_vas[i];
            if (va) {
                cuMemUnmap(va, aligned_mega);
                cuMemAddressFree(va, aligned_mega);
            }
            CUmemGenericAllocationHandle h = ep_group->ht_buffers.fabric_peer_allocs[i];
            if (h) cuMemRelease(h);
        }
        std::free(ep_group->ht_buffers.fabric_peer_vas);
        ep_group->ht_buffers.fabric_peer_vas = nullptr;
        std::free(ep_group->ht_buffers.fabric_peer_allocs);
        ep_group->ht_buffers.fabric_peer_allocs = nullptr;
    }

    // Release flags (rank 0 and imported ranks both have a va + alloc to release).
    if (ep_group->ht_buffers.fabric_flags_va) {
        cuMemUnmap(ep_group->ht_buffers.fabric_flags_va, flags_aligned);
        cuMemAddressFree(ep_group->ht_buffers.fabric_flags_va, flags_aligned);
        ep_group->ht_buffers.fabric_flags_va = 0;
    }
    if (ep_group->ht_buffers.fabric_flags_alloc) {
        cuMemRelease(ep_group->ht_buffers.fabric_flags_alloc);
        ep_group->ht_buffers.fabric_flags_alloc = 0;
    }

    // Grid barrier counter (local).
    if (ep_group->ht_buffers.dispatch_grid_barrier_counter) {
        ep_group->free_fn(ep_group->ht_buffers.dispatch_grid_barrier_counter);
        ep_group->ht_buffers.dispatch_grid_barrier_counter = nullptr;
    }

    // Unmap local mega and release own alloc.
    if (ep_group->ht_buffers.ipc_mega_buffer) {
        CUdeviceptr va = reinterpret_cast<CUdeviceptr>(ep_group->ht_buffers.ipc_mega_buffer);
        cuMemUnmap(va, aligned_mega);
        cuMemAddressFree(va, aligned_mega);
        ep_group->ht_buffers.ipc_mega_buffer = nullptr;
    }
    if (ep_group->ht_buffers.fabric_local_alloc) {
        cuMemRelease(ep_group->ht_buffers.fabric_local_alloc);
        ep_group->ht_buffers.fabric_local_alloc = 0;
    }

    if (ep_group->ht_buffers.host_ptr_block) {
        cudaFreeHost(ep_group->ht_buffers.host_ptr_block);
        ep_group->ht_buffers.host_ptr_block = nullptr;
    }
    if (ep_group->ht_buffers.peer_ipc_base_ptrs) {
        cudaFreeHost(ep_group->ht_buffers.peer_ipc_base_ptrs);
        ep_group->ht_buffers.peer_ipc_base_ptrs = nullptr;
    }

    ep_group->ht_buffers.initialized = false;
    return ncclSuccess;
}

// ============================================================================
// FULLMESH (Phase 2): direct peer-store fabric-memory init / destroy
// ----------------------------------------------------------------------------
// Allocates three independent FABRIC-exported buffers per rank. All three are
// imported by every other rank via cuMemImportFromShareableHandle so every
// rank can load / store every other rank's dispatch recv, counter row and
// combine recv with a single peer-mapped VA per buffer per rank.
//
//   (1) dispatch_recv_buf[nRanks][max_tokens_per_rank][meta_bytes + hidden_bytes]
//       Big. Every src rank atomicAdd-carves a slot in its own per-src block
//       at dest, then cooperatively vec-stores (meta, payload) into that slot.
//       Layout keeps src blocks disjoint so atomicAdd contention is
//       per-(src,dest), not per-dest, matching the Phase 2 Stage 1 winner.
//
//   (2) counter_row[nRanks]                                       (int32)
//       Small. One 32-bit counter per (dest, src). dest rank owns the whole
//       row; every src rank atomicAdd(1) remotely to pull a slot. Commit 4
//       combine ALSO reads this row (as a dest reading its own counter) to
//       know how many slots each src filled and therefore how many entries
//       to push back.
//
//   (3) combine_recv_buf[max_tokens_per_rank][max_topk_for_combine][hidden_bytes]
//       Big. src-owned buffer into which peer dest ranks push weighted FFN
//       outputs during commit-4 combine. No meta header (src already knows
//       the (t, k) coordinate; dest knows where to push by reading meta off
//       its own dispatch recv). A src-local reduce kernel then sums across
//       the k dimension weighted by topk_weights to produce combined_output.
//       max_topk_for_combine is a group-level upper bound so the allocation
//       is stable across handles regardless of per-handle top_k.
//
// Handles are exchanged with one ncclAllGatherHost of 3*FH_BYTES per rank
// (recv + counter + combine packed). All three peer maps get local-device
// READWRITE access so the dispatch kernel (src pushes to peer recv/counter)
// and the combine kernel (dest pushes to peer combine) both work without
// any per-call additional access descriptor changes.
// ============================================================================
static ncclResult_t init_fullmesh_intranode_fabric(ncclEpGroup_t ep_group,
                                                    const ncclEpGroupConfig_t* in_config,
                                                    cudaStream_t stream) {
    (void)in_config;
    ncclComm_t comm = ep_group->comm;
    int nRanks = ep_group->nRanks;
    int rank = ep_group->rank;
    int hidden = ep_group->hidden;
    size_t max_tokens = static_cast<size_t>(ep_group->config.max_tokens_per_rank);
    size_t token_bytes = static_cast<size_t>(ep_group->config.token_size_bytes);

    auto& fb = ep_group->fullmesh_buffers;
    fb.initialized = false;

    // Meta header layout is 16B, see device/fullmesh.cuh for the exact uint32
    // slot assignment. meta_bytes % 16 == 0 is load-bearing: the dispatch
    // kernel uses uint4 (16-byte) cooperative stores at entry+meta_bytes,
    // which traps with "misaligned address" if the invariant is violated.
    fb.meta_bytes           = 16;
    fb.bytes_per_entry      = fb.meta_bytes + token_bytes;
    fb.bytes_per_src_block  = max_tokens * fb.bytes_per_entry;
    fb.bytes_per_recv_buf   = static_cast<size_t>(nRanks) * fb.bytes_per_src_block;
    fb.bytes_per_counter_row = static_cast<size_t>(nRanks) * sizeof(int32_t);

    // Group-level upper bound on top_k. Commit 4 combine kernel asserts the
    // per-handle num_topk does not exceed this. 32 matches the dispatch
    // kernel's warps-per-block cap and covers every production MoE top_k
    // we've seen (k=1, 2, 4, 6, 8, 16); revisit only if a caller needs more.
    fb.max_topk_for_combine     = 32;
    fb.bytes_per_combine_token  = fb.max_topk_for_combine * token_bytes;
    fb.bytes_per_combine_buf    = max_tokens * fb.bytes_per_combine_token;

    // Fabric allocation prop (FABRIC handle + device-pinned).
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = ep_group->cuda_device_id;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;

    size_t gran = 0;
    CU_CHECK(cuMemGetAllocationGranularity(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));

    auto round_up = [](size_t sz, size_t g) -> size_t {
        if (g == 0) g = 1;
        return (sz + g - 1) & ~(g - 1);
    };
    fb.recv_aligned_size    = round_up(fb.bytes_per_recv_buf,     gran);
    fb.counter_aligned_size = round_up(fb.bytes_per_counter_row,  gran);
    fb.combine_aligned_size = round_up(fb.bytes_per_combine_buf,  gran);

    // --- Allocate local 3 buffers, map into local VA, grant self access ---
    CU_CHECK(cuMemCreate(&fb.recv_local_alloc,    fb.recv_aligned_size,    &prop, 0));
    CU_CHECK(cuMemCreate(&fb.counter_local_alloc, fb.counter_aligned_size, &prop, 0));
    CU_CHECK(cuMemCreate(&fb.combine_local_alloc, fb.combine_aligned_size, &prop, 0));

    CU_CHECK(cuMemAddressReserve(&fb.recv_local_va,    fb.recv_aligned_size,    gran, 0, 0));
    CU_CHECK(cuMemMap(fb.recv_local_va,    fb.recv_aligned_size,    0, fb.recv_local_alloc,    0));
    CU_CHECK(cuMemAddressReserve(&fb.counter_local_va, fb.counter_aligned_size, gran, 0, 0));
    CU_CHECK(cuMemMap(fb.counter_local_va, fb.counter_aligned_size, 0, fb.counter_local_alloc, 0));
    CU_CHECK(cuMemAddressReserve(&fb.combine_local_va, fb.combine_aligned_size, gran, 0, 0));
    CU_CHECK(cuMemMap(fb.combine_local_va, fb.combine_aligned_size, 0, fb.combine_local_alloc, 0));

    CUmemAccessDesc access = {};
    access.location = prop.location;  // local device
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CU_CHECK(cuMemSetAccess(fb.recv_local_va,    fb.recv_aligned_size,    &access, 1));
    CU_CHECK(cuMemSetAccess(fb.counter_local_va, fb.counter_aligned_size, &access, 1));
    CU_CHECK(cuMemSetAccess(fb.combine_local_va, fb.combine_aligned_size, &access, 1));

    // Zero counter so the very first dispatch sees slot 0 for every src.
    // recv / combine are left undefined on purpose — kernels write before
    // anyone reads.
    CUDACHECK_RET(cudaMemsetAsync(reinterpret_cast<void*>(fb.counter_local_va),
                                  0, fb.bytes_per_counter_row, stream));

    // --- Export 3 local fabric handles and allgather them in one shot ---
    CUmemFabricHandle local_recv_fh = {};
    CUmemFabricHandle local_counter_fh = {};
    CUmemFabricHandle local_combine_fh = {};
    CU_CHECK(cuMemExportToShareableHandle(&local_recv_fh,    fb.recv_local_alloc,
                                          CU_MEM_HANDLE_TYPE_FABRIC, 0));
    CU_CHECK(cuMemExportToShareableHandle(&local_counter_fh, fb.counter_local_alloc,
                                          CU_MEM_HANDLE_TYPE_FABRIC, 0));
    CU_CHECK(cuMemExportToShareableHandle(&local_combine_fh, fb.combine_local_alloc,
                                          CU_MEM_HANDLE_TYPE_FABRIC, 0));

    static_assert(sizeof(CUmemFabricHandle) == 64, "CUmemFabricHandle expected to be 64 bytes");
    const size_t FH_BYTES = sizeof(CUmemFabricHandle);
    const size_t SLOT_BYTES = 3 * FH_BYTES;  // recv | counter | combine

    std::unique_ptr<uint8_t[]> gathered(new uint8_t[SLOT_BYTES * nRanks]);
    std::memset(gathered.get(), 0, SLOT_BYTES * nRanks);
    uint8_t* my_slot = gathered.get() + rank * SLOT_BYTES;
    std::memcpy(my_slot + 0 * FH_BYTES, &local_recv_fh,    FH_BYTES);
    std::memcpy(my_slot + 1 * FH_BYTES, &local_counter_fh, FH_BYTES);
    std::memcpy(my_slot + 2 * FH_BYTES, &local_combine_fh, FH_BYTES);
    ncclAllGatherHost(gathered.get(), SLOT_BYTES, rank, nRanks, comm, stream);

    // --- Import peer handles + map per-peer VA for all 3 buffers ---
    fb.peer_recv_vas        = static_cast<CUdeviceptr*>(std::calloc(nRanks, sizeof(CUdeviceptr)));
    fb.peer_counter_vas     = static_cast<CUdeviceptr*>(std::calloc(nRanks, sizeof(CUdeviceptr)));
    fb.peer_combine_vas     = static_cast<CUdeviceptr*>(std::calloc(nRanks, sizeof(CUdeviceptr)));
    fb.peer_recv_allocs     = static_cast<CUmemGenericAllocationHandle*>(std::calloc(nRanks, sizeof(CUmemGenericAllocationHandle)));
    fb.peer_counter_allocs  = static_cast<CUmemGenericAllocationHandle*>(std::calloc(nRanks, sizeof(CUmemGenericAllocationHandle)));
    fb.peer_combine_allocs  = static_cast<CUmemGenericAllocationHandle*>(std::calloc(nRanks, sizeof(CUmemGenericAllocationHandle)));
    if (!fb.peer_recv_vas || !fb.peer_counter_vas || !fb.peer_combine_vas ||
        !fb.peer_recv_allocs || !fb.peer_counter_allocs || !fb.peer_combine_allocs) {
        fprintf(stderr, "init_fullmesh_intranode_fabric: calloc failed for peer arrays\n");
        return ncclSystemError;
    }

    for (int i = 0; i < nRanks; ++i) {
        if (i == rank) {
            fb.peer_recv_vas[i]       = fb.recv_local_va;
            fb.peer_counter_vas[i]    = fb.counter_local_va;
            fb.peer_combine_vas[i]    = fb.combine_local_va;
            fb.peer_recv_allocs[i]    = 0;  // sentinel: don't release own handle via peer cleanup
            fb.peer_counter_allocs[i] = 0;
            fb.peer_combine_allocs[i] = 0;
            continue;
        }
        const uint8_t* peer_slot = gathered.get() + i * SLOT_BYTES;
        CUmemFabricHandle peer_recv_fh, peer_counter_fh, peer_combine_fh;
        std::memcpy(&peer_recv_fh,    peer_slot + 0 * FH_BYTES, FH_BYTES);
        std::memcpy(&peer_counter_fh, peer_slot + 1 * FH_BYTES, FH_BYTES);
        std::memcpy(&peer_combine_fh, peer_slot + 2 * FH_BYTES, FH_BYTES);

        CUmemGenericAllocationHandle peer_recv_h = 0, peer_counter_h = 0, peer_combine_h = 0;
        CU_CHECK(cuMemImportFromShareableHandle(&peer_recv_h,    &peer_recv_fh,
                                                CU_MEM_HANDLE_TYPE_FABRIC));
        CU_CHECK(cuMemImportFromShareableHandle(&peer_counter_h, &peer_counter_fh,
                                                CU_MEM_HANDLE_TYPE_FABRIC));
        CU_CHECK(cuMemImportFromShareableHandle(&peer_combine_h, &peer_combine_fh,
                                                CU_MEM_HANDLE_TYPE_FABRIC));
        fb.peer_recv_allocs[i]    = peer_recv_h;
        fb.peer_counter_allocs[i] = peer_counter_h;
        fb.peer_combine_allocs[i] = peer_combine_h;

        CUdeviceptr peer_recv_va = 0, peer_counter_va = 0, peer_combine_va = 0;
        CU_CHECK(cuMemAddressReserve(&peer_recv_va,    fb.recv_aligned_size,    gran, 0, 0));
        CU_CHECK(cuMemMap           ( peer_recv_va,    fb.recv_aligned_size,    0, peer_recv_h,    0));
        CU_CHECK(cuMemSetAccess     ( peer_recv_va,    fb.recv_aligned_size,    &access, 1));
        CU_CHECK(cuMemAddressReserve(&peer_counter_va, fb.counter_aligned_size, gran, 0, 0));
        CU_CHECK(cuMemMap           ( peer_counter_va, fb.counter_aligned_size, 0, peer_counter_h, 0));
        CU_CHECK(cuMemSetAccess     ( peer_counter_va, fb.counter_aligned_size, &access, 1));
        CU_CHECK(cuMemAddressReserve(&peer_combine_va, fb.combine_aligned_size, gran, 0, 0));
        CU_CHECK(cuMemMap           ( peer_combine_va, fb.combine_aligned_size, 0, peer_combine_h, 0));
        CU_CHECK(cuMemSetAccess     ( peer_combine_va, fb.combine_aligned_size, &access, 1));

        fb.peer_recv_vas[i]    = peer_recv_va;
        fb.peer_counter_vas[i] = peer_counter_va;
        fb.peer_combine_vas[i] = peer_combine_va;
    }

    // --- Build device-side peer pointer arrays for kernel one-indirection ---
    std::vector<void*> peer_recv_host(nRanks, nullptr);
    std::vector<void*> peer_counter_host(nRanks, nullptr);
    std::vector<void*> peer_combine_host(nRanks, nullptr);
    for (int i = 0; i < nRanks; ++i) {
        peer_recv_host[i]    = reinterpret_cast<void*>(fb.peer_recv_vas[i]);
        peer_counter_host[i] = reinterpret_cast<void*>(fb.peer_counter_vas[i]);
        peer_combine_host[i] = reinterpret_cast<void*>(fb.peer_combine_vas[i]);
    }
    CUDACHECK_RET(ep_group->alloc_fn(reinterpret_cast<void**>(&fb.peer_recv_vas_dev),    nRanks * sizeof(void*)));
    CUDACHECK_RET(ep_group->alloc_fn(reinterpret_cast<void**>(&fb.peer_counter_vas_dev), nRanks * sizeof(void*)));
    CUDACHECK_RET(ep_group->alloc_fn(reinterpret_cast<void**>(&fb.peer_combine_vas_dev), nRanks * sizeof(void*)));
    CUDACHECK_RET(cudaMemcpyAsync(fb.peer_recv_vas_dev,    peer_recv_host.data(),
                                  nRanks * sizeof(void*), cudaMemcpyHostToDevice, stream));
    CUDACHECK_RET(cudaMemcpyAsync(fb.peer_counter_vas_dev, peer_counter_host.data(),
                                  nRanks * sizeof(void*), cudaMemcpyHostToDevice, stream));
    CUDACHECK_RET(cudaMemcpyAsync(fb.peer_combine_vas_dev, peer_combine_host.data(),
                                  nRanks * sizeof(void*), cudaMemcpyHostToDevice, stream));

    // Rank-0 trace for the sweep to grep and confirm all 3 buffers were
    // exchanged + mapped, including the newly-added combine buffer.
    if (rank == 0) {
        fprintf(stderr,
                "[FULLMESH] init_fullmesh_intranode_fabric OK: "
                "nRanks=%d max_tokens_per_rank=%zu hidden=%d token_bytes=%zu "
                "meta_bytes=%zu bytes_per_entry=%zu bytes_per_src_block=%zu "
                "recv_total=%zu (aligned=%zu) counter_total=%zu (aligned=%zu) "
                "combine_total=%zu (aligned=%zu, max_topk=%zu) granularity=%zu\n",
                nRanks, max_tokens, hidden, token_bytes,
                fb.meta_bytes, fb.bytes_per_entry, fb.bytes_per_src_block,
                fb.bytes_per_recv_buf, fb.recv_aligned_size,
                fb.bytes_per_counter_row, fb.counter_aligned_size,
                fb.bytes_per_combine_buf, fb.combine_aligned_size, fb.max_topk_for_combine,
                gran);
        for (int i = 0; i < nRanks && i < 4; ++i) {
            fprintf(stderr,
                    "[FULLMESH] peer[%d] recv_va=%p counter_va=%p combine_va=%p\n",
                    i,
                    reinterpret_cast<void*>(fb.peer_recv_vas[i]),
                    reinterpret_cast<void*>(fb.peer_counter_vas[i]),
                    reinterpret_cast<void*>(fb.peer_combine_vas[i]));
        }
        fflush(stderr);
    }

    fb.initialized = true;
    CUDACHECK_RET(cudaStreamSynchronize(stream));
    return ncclSuccess;
}

static ncclResult_t destroy_fullmesh_intranode_fabric(ncclEpGroup_t ep_group) {
    int nRanks = ep_group->nRanks;
    int rank = ep_group->rank;
    auto& fb = ep_group->fullmesh_buffers;

    // Unmap + release peers (skip self sentinel handle) for all 3 buffers.
    if (fb.peer_recv_vas) {
        for (int i = 0; i < nRanks; ++i) {
            if (i == rank) continue;
            if (fb.peer_recv_vas[i]) {
                cuMemUnmap(fb.peer_recv_vas[i], fb.recv_aligned_size);
                cuMemAddressFree(fb.peer_recv_vas[i], fb.recv_aligned_size);
            }
            if (fb.peer_recv_allocs && fb.peer_recv_allocs[i]) {
                cuMemRelease(fb.peer_recv_allocs[i]);
            }
            if (fb.peer_counter_vas && fb.peer_counter_vas[i]) {
                cuMemUnmap(fb.peer_counter_vas[i], fb.counter_aligned_size);
                cuMemAddressFree(fb.peer_counter_vas[i], fb.counter_aligned_size);
            }
            if (fb.peer_counter_allocs && fb.peer_counter_allocs[i]) {
                cuMemRelease(fb.peer_counter_allocs[i]);
            }
            if (fb.peer_combine_vas && fb.peer_combine_vas[i]) {
                cuMemUnmap(fb.peer_combine_vas[i], fb.combine_aligned_size);
                cuMemAddressFree(fb.peer_combine_vas[i], fb.combine_aligned_size);
            }
            if (fb.peer_combine_allocs && fb.peer_combine_allocs[i]) {
                cuMemRelease(fb.peer_combine_allocs[i]);
            }
        }
        std::free(fb.peer_recv_vas);        fb.peer_recv_vas = nullptr;
        std::free(fb.peer_counter_vas);     fb.peer_counter_vas = nullptr;
        std::free(fb.peer_combine_vas);     fb.peer_combine_vas = nullptr;
        std::free(fb.peer_recv_allocs);     fb.peer_recv_allocs = nullptr;
        std::free(fb.peer_counter_allocs);  fb.peer_counter_allocs = nullptr;
        std::free(fb.peer_combine_allocs);  fb.peer_combine_allocs = nullptr;
    }

    // Release local allocations (recv + counter + combine).
    if (fb.recv_local_va) {
        cuMemUnmap(fb.recv_local_va, fb.recv_aligned_size);
        cuMemAddressFree(fb.recv_local_va, fb.recv_aligned_size);
        fb.recv_local_va = 0;
    }
    if (fb.recv_local_alloc) {
        cuMemRelease(fb.recv_local_alloc);
        fb.recv_local_alloc = 0;
    }
    if (fb.counter_local_va) {
        cuMemUnmap(fb.counter_local_va, fb.counter_aligned_size);
        cuMemAddressFree(fb.counter_local_va, fb.counter_aligned_size);
        fb.counter_local_va = 0;
    }
    if (fb.counter_local_alloc) {
        cuMemRelease(fb.counter_local_alloc);
        fb.counter_local_alloc = 0;
    }
    if (fb.combine_local_va) {
        cuMemUnmap(fb.combine_local_va, fb.combine_aligned_size);
        cuMemAddressFree(fb.combine_local_va, fb.combine_aligned_size);
        fb.combine_local_va = 0;
    }
    if (fb.combine_local_alloc) {
        cuMemRelease(fb.combine_local_alloc);
        fb.combine_local_alloc = 0;
    }

    // Device-side pointer arrays.
    if (fb.peer_recv_vas_dev) {
        ep_group->free_fn(fb.peer_recv_vas_dev);
        fb.peer_recv_vas_dev = nullptr;
    }
    if (fb.peer_counter_vas_dev) {
        ep_group->free_fn(fb.peer_counter_vas_dev);
        fb.peer_counter_vas_dev = nullptr;
    }
    if (fb.peer_combine_vas_dev) {
        ep_group->free_fn(fb.peer_combine_vas_dev);
        fb.peer_combine_vas_dev = nullptr;
    }

    fb.initialized = false;
    return ncclSuccess;
}

// NCCLCHECK macro for public API error checking
// Undef any existing definition from internal NCCL headers to avoid using ncclDebugLog
#ifdef NCCLCHECK
#undef NCCLCHECK
#endif
#define NCCLCHECK(cmd) do {                                 \
    ncclResult_t res = cmd;                                 \
    if (res != ncclSuccess) {                               \
        fprintf(stderr, "NCCL error %s:%d '%s'\n",          \
                __FILE__, __LINE__, ncclGetErrorString(res));\
        return res;                                          \
    }                                                        \
} while(0)

// CUDACHECK_RET macro for CUDA calls in functions returning ncclResult_t
#ifndef CUDACHECK_RET
#define CUDACHECK_RET(cmd) do {                             \
    cudaError_t err = cmd;                                  \
    if (err != cudaSuccess) {                               \
        fprintf(stderr, "CUDA error %s:%d '%s'\n",          \
                __FILE__, __LINE__, cudaGetErrorString(err));\
        return ncclInternalError;                            \
    }                                                        \
} while(0)
#endif

// Constants for GIN configuration
static constexpr int HYBRIDEP_GIN_MAX_CONTEXTS = 32;
static constexpr int HYBRIDEP_GIN_CTXS_PER_COMM = 4;
static constexpr int MAX_BARRIER_SESSIONS = 32;

static ncclResult_t init_hybridep_internode(ncclEpGroup_t ep_group,
    const ncclEpGroupConfig_t* in_config,
    cudaStream_t stream)
{
    // Initialize using public NCCL APIs
    if (!in_config || !ep_group) {
        fprintf(stderr, "init_hybridep_internode: null config or ep_group\n");
        return ncclInvalidArgument;
    }

    int nNodes = ep_group->nNodes;
    // HT internode always uses physical node semantics: RDMA groups = nNodes, IPC = gpus_per_node.
    // MNNVL (multiple nodes with NVLink fabric) is treated like regular multi-node RDMA for HT.
    int gpus_per_node    = ep_group->gpus_per_node;
    int rank_in_node     = ep_group->rank_in_node;
    int node_id          = ep_group->node_id;
    int n_ranks_per_node = gpus_per_node;
    ep_group->ht_buffers.internode_initialized = false;

    if (nNodes <= 1) {
        // Single physical node — no internode RDMA needed
        return ncclSuccess;
    }

    // =========================================================================
    // Phase 1: All local allocations (no collectives)
    // ncclMemAlloc + buffer partitioning moved here from after ncclDevCommCreate
    // to remove them from the collective critical path.
    // =========================================================================

    constexpr size_t GIN_ALIGNMENT = 4096;
    auto align_size = [](size_t sz, size_t alignment) {
        return (sz + alignment - 1) & ~(alignment - 1);
    };

    // These buffers are accessed with stride MAX_SUPPORTED_TOKENS_PER_RANK (compile-time constant
    // used as rdma_remote_node_id * MAX_SUPPORTED_TOKENS_PER_RANK + token_offset in the kernel).
    // They must be sized for that stride regardless of the runtime max_tokens_per_rank.
    size_t rdma_intra_node_red_token_sz = align_size(static_cast<size_t>(MAX_SUPPORTED_TOKENS_PER_RANK * (nNodes - 1)) * ep_group->hidden * sizeof(uint16_t), GIN_ALIGNMENT);
    size_t combine_rdma_inter_node_group_token_sz = rdma_intra_node_red_token_sz;
    size_t rdma_intra_node_red_prob_sz = align_size(static_cast<size_t>(MAX_SUPPORTED_TOKENS_PER_RANK * (nNodes - 1)) * (ep_group->num_local_experts * n_ranks_per_node) * sizeof(float), GIN_ALIGNMENT);
    size_t combine_rdma_inter_node_group_prob_sz = rdma_intra_node_red_prob_sz;
    size_t flags_sz = align_size(static_cast<size_t>(nNodes) * sizeof(uint64_t), GIN_ALIGNMENT);
    size_t token_staging_sz = align_size(static_cast<size_t>(ep_group->config.max_tokens_per_rank) * ep_group->hidden * sizeof(uint16_t), GIN_ALIGNMENT);
    size_t dense_prob_sz = align_size(static_cast<size_t>(ep_group->config.max_tokens_per_rank) * ep_group->config.num_experts * sizeof(float), GIN_ALIGNMENT);
    size_t scaling_factor_staging_sz = align_size(static_cast<size_t>(ep_group->config.max_tokens_per_rank) * sizeof(float), GIN_ALIGNMENT);

    size_t bytes_per_token_entry = ep_group->hidden * sizeof(uint16_t);
    size_t bytes_per_prob_entry = (ep_group->num_local_experts * n_ranks_per_node) * sizeof(float);
    size_t bytes_per_sf_entry = (ep_group->hidden / 128) * sizeof(float);
    size_t bytes_per_entry = bytes_per_token_entry + bytes_per_prob_entry + bytes_per_sf_entry;
    size_t rdma_send_staging_sz = align_size(static_cast<size_t>(nNodes - 1) * ep_group->config.max_tokens_per_rank * bytes_per_entry, GIN_ALIGNMENT);
    size_t rdma_recv_packed_sz = align_size(static_cast<size_t>(nNodes - 1) * ep_group->config.max_tokens_per_rank * bytes_per_entry, GIN_ALIGNMENT);

    size_t total_gin_buffer_size = 0;
    total_gin_buffer_size += rdma_intra_node_red_token_sz;
    total_gin_buffer_size += combine_rdma_inter_node_group_token_sz;
    total_gin_buffer_size += rdma_intra_node_red_prob_sz;
    total_gin_buffer_size += combine_rdma_inter_node_group_prob_sz;
    total_gin_buffer_size += flags_sz * 2;
    total_gin_buffer_size += token_staging_sz;
    total_gin_buffer_size += dense_prob_sz;
    total_gin_buffer_size += scaling_factor_staging_sz;
    total_gin_buffer_size += rdma_send_staging_sz;
    total_gin_buffer_size += rdma_recv_packed_sz;

    NCCLCHECK(ncclMemAlloc(&ep_group->gin_config.gin_base_ptr, total_gin_buffer_size));

    // Partition the buffer into individual regions
    uint8_t* ptr = reinterpret_cast<uint8_t*>(ep_group->gin_config.gin_base_ptr);
    size_t offset = 0;

    ep_group->ht_buffers.rdma_intra_node_red_token = reinterpret_cast<uint16_t*>(ptr + offset);
    offset += rdma_intra_node_red_token_sz;

    ep_group->ht_buffers.combine_rdma_inter_node_group_token = reinterpret_cast<uint16_t*>(ptr + offset);
    offset += combine_rdma_inter_node_group_token_sz;

    ep_group->ht_buffers.rdma_intra_node_red_prob = reinterpret_cast<float*>(ptr + offset);
    offset += rdma_intra_node_red_prob_sz;

    ep_group->ht_buffers.combine_rdma_inter_node_group_prob = reinterpret_cast<float*>(ptr + offset);
    offset += combine_rdma_inter_node_group_prob_sz;

    ep_group->ht_buffers.rdma_inter_node_group_flags = reinterpret_cast<uint64_t*>(ptr + offset);
    CUDACHECK_RET(cudaMemset(ep_group->ht_buffers.rdma_inter_node_group_flags, 0, flags_sz));
    offset += flags_sz;

    ep_group->ht_buffers.combine_rdma_inter_node_group_flags = reinterpret_cast<uint64_t*>(ptr + offset);
    CUDACHECK_RET(cudaMemset(ep_group->ht_buffers.combine_rdma_inter_node_group_flags, 0, flags_sz));
    offset += flags_sz;

    ep_group->ht_buffers.token_staging_buffer = reinterpret_cast<void*>(ptr + offset);
    offset += token_staging_sz;

    ep_group->ht_buffers.dense_prob_buffer = reinterpret_cast<float*>(ptr + offset);
    offset += dense_prob_sz;

    ep_group->ht_buffers.scaling_factor_staging_buffer = reinterpret_cast<float*>(ptr + offset);
    offset += scaling_factor_staging_sz;

    offset += rdma_send_staging_sz;

    // Calculate offsets for kernel mr_info
    size_t cur_offset = 0;
    ep_group->gin_config.rdma_intra_node_red_token_offset = cur_offset;
    cur_offset += rdma_intra_node_red_token_sz;

    ep_group->gin_config.combine_rdma_inter_node_group_token_offset = cur_offset;
    cur_offset += combine_rdma_inter_node_group_token_sz;

    ep_group->gin_config.rdma_intra_node_red_prob_offset = cur_offset;
    cur_offset += rdma_intra_node_red_prob_sz;

    ep_group->gin_config.combine_rdma_inter_node_group_prob_offset = cur_offset;
    cur_offset += combine_rdma_inter_node_group_prob_sz;

    cur_offset += flags_sz * 2;

    ep_group->gin_config.token_staging_offset = cur_offset;
    cur_offset += token_staging_sz;

    ep_group->gin_config.dense_prob_offset = cur_offset;
    cur_offset += dense_prob_sz;

    ep_group->gin_config.scaling_factor_staging_offset = cur_offset;
    cur_offset += scaling_factor_staging_sz;

    ep_group->gin_config.rdma_send_staging_offset = cur_offset;
    cur_offset += rdma_send_staging_sz;

    ep_group->gin_config.rdma_inter_node_group_packed_offset = cur_offset;
    cur_offset += rdma_recv_packed_sz;

    // =========================================================================
    // Phase 2: main_comm collective -- create split communicator
    // =========================================================================
    // Node concept: color = rank's position within its physical node (rank_in_node).
    // Groups same-node-position ranks across nodes into one split_comm for GIN/RDMA ops.
    // key = node_id orders ranks within each split_comm by their node index.
    int color = rank_in_node;
    int key = node_id;
    NCCLCHECK(ncclCommSplit(ep_group->comm, color, key, &ep_group->gin_config.split_comm, nullptr));

    int qps_per_rank = ep_group->config.num_qp_per_rank;
    int min_required_ctx = HYBRIDEP_DISPATCH_NUM_OF_BLOCKS * HYBRIDEP_DISPATCH_N2N_WARPS;
    if (qps_per_rank == 0) qps_per_rank = min_required_ctx;
    if (qps_per_rank < min_required_ctx) {
        fprintf(stderr, "[HT GIN] Error: num_qp_per_rank(%d) must be >= %d for dedicated N2N warp contexts\n",
                qps_per_rank, min_required_ctx);
        return ncclInvalidUsage;
    }
    ep_group->gin_config.qps_per_rank = qps_per_rank;
    ep_group->gin_config.num_comms = 1;
    ep_group->gin_config.num_ctx_per_comm = qps_per_rank;

    int max_chunks_per_rank = (MAX_SUPPORTED_TOKENS_PER_RANK + HT_OF_NUM_TOKENS_PER_CHUNK - 1) / HT_OF_NUM_TOKENS_PER_CHUNK;
    int dispatch_signals = n_ranks_per_node * nNodes * max_chunks_per_rank;
    int combine_signals = n_ranks_per_node * nNodes * max_chunks_per_rank;
    int streaming_tail_signals = nNodes * nNodes * n_ranks_per_node * max_chunks_per_rank;
    int streaming_head_signals = nNodes * nNodes * n_ranks_per_node;
    ep_group->gin_config.num_total_signals = dispatch_signals + combine_signals +
                                               streaming_tail_signals + streaming_head_signals + MAX_BARRIER_SESSIONS;
    ep_group->gin_config.signals_base = 0;
    ep_group->gin_config.combine_signal_offset = dispatch_signals;
    ep_group->gin_config.signals_tail_base = dispatch_signals + combine_signals;

    // =========================================================================
    // Phase 3: split_comm setup (DevCommCreate + WindowRegister)
    // =========================================================================
    ep_group->gin_config.num_dcomms = 1;
    ep_group->gin_config.dcomms = new ncclDevComm_t[1];

   {
        ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
        NCCL_CHECK_RESULT(ncclCommQueryProperties(ep_group->gin_config.split_comm, &props));
        if (props.ginType == NCCL_GIN_TYPE_NONE) {
            fprintf(stderr, "[HT GIN] Error: NCCL EP internode requires GIN, but GIN is not supported\n");
            return ncclInvalidUsage;
        }
    }

    {
        ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
        reqs.ginSignalCount = ep_group->gin_config.num_total_signals;
        reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
        reqs.ginContextCount = ep_group->gin_config.num_ctx_per_comm;
        reqs.ginQueueDepth = 3 * HT_OF_NUM_TOKENS_PER_CHUNK + 1;
        NCCLCHECK(ncclDevCommCreate(ep_group->gin_config.split_comm, &reqs, &ep_group->gin_config.dcomms[0]));
    }

    CUDACHECK_RET(cudaMalloc(reinterpret_cast<void**>(&ep_group->gin_config.d_dcomms),
                          sizeof(ncclDevComm_t) * ep_group->gin_config.num_dcomms));
    CUDACHECK_RET(cudaMemcpy(ep_group->gin_config.d_dcomms, ep_group->gin_config.dcomms,
                          sizeof(ncclDevComm_t) * ep_group->gin_config.num_dcomms, cudaMemcpyHostToDevice));

    // WindowRegister: depends on split_comm (Phase 2) and gin_base_ptr (Phase 1)
    NCCLCHECK(ncclCommWindowRegister(ep_group->gin_config.split_comm,
        ep_group->gin_config.gin_base_ptr,
        total_gin_buffer_size,
        &ep_group->gin_config.nccl_window, 0));

    ep_group->ht_buffers.internode_initialized = true;
    return ncclSuccess;
}

static ncclResult_t destroy_hybridep_internode(ncclEpGroup_t ep_group){
    if (!ep_group->ht_buffers.internode_initialized) return ncclSuccess;

    // =========================================================================
    // Cleanup using public NCCL APIs
    // =========================================================================

    // Destroy device communicator (created on split_comm)
    if (ep_group->gin_config.dcomms != nullptr) {
        if (ep_group->gin_config.split_comm) {
            ncclResult_t res = ncclDevCommDestroy(ep_group->gin_config.split_comm, &ep_group->gin_config.dcomms[0]);
            if (res != ncclSuccess) {
                fprintf(stderr, "[HT GIN] Warning: Failed to destroy device comm: %s\n",
                        ncclGetErrorString(res));
            }
        }
        delete[] ep_group->gin_config.dcomms;
        ep_group->gin_config.dcomms = nullptr;
    }
    // Free device memory for dcomms
    if (ep_group->gin_config.d_dcomms != nullptr) {
        cudaFree(ep_group->gin_config.d_dcomms);
        ep_group->gin_config.d_dcomms = nullptr;
    }

    // Deregister the single window on split_comm
    if (ep_group->gin_config.gin_base_ptr != nullptr) {
        ncclCommWindowDeregister(ep_group->gin_config.split_comm, ep_group->gin_config.nccl_window);
        ep_group->gin_config.nccl_window = {};
    }

    // Free the single GIN buffer (contains all RDMA regions)
    if (ep_group->gin_config.gin_base_ptr != nullptr) {
        ncclResult_t res = ncclMemFree(ep_group->gin_config.gin_base_ptr);
        if (res != ncclSuccess) {
            fprintf(stderr, "[HT GIN] Warning: Failed to free GIN memory: %s\n",
                    ncclGetErrorString(res));
        }
        ep_group->gin_config.gin_base_ptr = nullptr;

        // Clear buffer pointers (they pointed into gin_base_ptr)
        ep_group->ht_buffers.rdma_intra_node_red_token = nullptr;
        ep_group->ht_buffers.combine_rdma_inter_node_group_token = nullptr;
        ep_group->ht_buffers.rdma_intra_node_red_prob = nullptr;
        ep_group->ht_buffers.combine_rdma_inter_node_group_prob = nullptr;
        ep_group->ht_buffers.rdma_inter_node_group_flags = nullptr;
        ep_group->ht_buffers.combine_rdma_inter_node_group_flags = nullptr;
        ep_group->ht_buffers.token_staging_buffer = nullptr;
        ep_group->ht_buffers.dense_prob_buffer = nullptr;
        ep_group->ht_buffers.scaling_factor_staging_buffer = nullptr;
    }

    // Finalize and destroy the split communicator
    if (ep_group->gin_config.split_comm != nullptr) {
        ncclCommFinalize(ep_group->gin_config.split_comm);
        ncclCommDestroy(ep_group->gin_config.split_comm);
        ep_group->gin_config.split_comm = nullptr;
    }
    ep_group->gin_config.num_comms = 0;

    ep_group->ht_buffers.internode_initialized = false;
    return ncclSuccess;
}

ncclResult_t ncclEpCreateGroup(
    ncclEpGroup_t* out_ep_group,
    ncclComm_t comm,
    const ncclEpGroupConfig_t* in_config,
    cudaStream_t stream,
    ncclEpAllocFn_t alloc_fn,
    ncclEpFreeFn_t free_fn
) {
    // Parameter validation
    assert(out_ep_group != nullptr);
    assert(in_config != nullptr);
    int nRanks;
    assert(comm != nullptr && ncclCommCount(comm, &nRanks) == ncclSuccess && nRanks > 0);
    assert(in_config->version == 1 && "ncclEpCreateGroup: invalid config version (expected 1)");
    assert((in_config->algorithm == NCCL_EP_ALGO_LOW_LATENCY ||
            in_config->algorithm == NCCL_EP_ALGO_HIGH_THROUGHPUT ||
            in_config->algorithm == NCCL_EP_ALGO_FULLMESH) &&
           "ncclEpCreateGroup: invalid algorithm, supported: low_latency, high_throughput, fullmesh");

    bool low_latency_mode = (in_config->algorithm == NCCL_EP_ALGO_LOW_LATENCY);
    bool hybridep_mode = (in_config->algorithm == NCCL_EP_ALGO_HIGH_THROUGHPUT);
    bool fullmesh_mode = (in_config->algorithm == NCCL_EP_ALGO_FULLMESH);
    assert(in_config->num_experts > 0 && "ncclEpCreateGroup: num_experts must be greater than 0");
    assert(in_config->token_size_bytes > 0 && "ncclEpCreateGroup: token_size_bytes must be greater than 0");
    assert(!(low_latency_mode && in_config->max_tokens_per_rank == 0) &&
            "ncclEpCreateGroup: max_tokens_per_rank must be greater than 0 for low latency mode");
    assert(!(hybridep_mode && in_config->max_tokens_per_rank == 0) &&
             "ncclEpCreateGroup: max_tokens_per_rank must be set for HT backend");
    assert(!(hybridep_mode &&
             in_config->max_tokens_per_rank > MAX_SUPPORTED_TOKENS_PER_RANK) &&
             "ncclEpCreateGroup: HT max_tokens_per_rank exceeds build-time MAX_SUPPORTED_TOKENS_PER_RANK");
    assert(!(fullmesh_mode && in_config->max_tokens_per_rank == 0) &&
             "ncclEpCreateGroup: max_tokens_per_rank must be set for FULLMESH backend");
    // Query LSA team size: number of ranks reachable via NVLink/LSA from this rank.
    ncclTeam lsa_team = ncclTeamLsa(comm);
    const int lsa_team_size = lsa_team.nRanks;

    if (hybridep_mode) {
        assert((nRanks % lsa_team_size) == 0 &&
               "ncclEpCreateGroup: HT requires nRanks divisible by lsa_team_size");
    }
    // FULLMESH requires every rank reachable via CUmemFabricHandle peer mapping.
    // On NV72 / MNNVL this means lsa_team_size == nRanks. Reject multi-LSA topologies
    // explicitly so we fail fast instead of corrupting peer buffers later.
    if (fullmesh_mode) {
        assert(lsa_team_size == nRanks &&
               "ncclEpCreateGroup: FULLMESH requires lsa_team_size == nRanks "
               "(NV72 MNNVL full-coverage fabric)");
    }

    // Allocate EP group structure
    void* raw_memory = malloc(sizeof(ncclEpGroup));
    assert(raw_memory != nullptr && "Failed to malloc for ncclEpGroup");
    *out_ep_group = new (raw_memory) ncclEpGroup();
    ncclEpGroup_t ep_group = *out_ep_group;

    // Store configuration
    ep_group->comm = comm;
    ep_group->config = *in_config;
    // C-style cast required: cudaMalloc/cudaFree are overloaded and reinterpret_cast
    // cannot disambiguate overloaded functions.
    ep_group->alloc_fn = alloc_fn ? alloc_fn : (ncclEpAllocFn_t)cudaMalloc;
    ep_group->free_fn = free_fn ? free_fn : (ncclEpFreeFn_t)cudaFree;

    NCCL_CHECK_RESULT(ncclCommCount(comm, &ep_group->nRanks));
    NCCL_CHECK_RESULT(ncclCommUserRank(comm, &ep_group->rank));
    NCCL_CHECK_RESULT(ncclCommCuDevice(comm, &ep_group->cuda_device_id));

    // Determine number of nodes by gathering hostnames and counting unique ones
    constexpr size_t HOSTNAME_LEN = 256;
    std::vector<char> all_hostnames(HOSTNAME_LEN * ep_group->nRanks, 0);
    gethostname(all_hostnames.data() + ep_group->rank * HOSTNAME_LEN, HOSTNAME_LEN);
    ncclAllGatherHost(all_hostnames.data(), HOSTNAME_LEN, ep_group->rank, ep_group->nRanks, comm, stream);
    std::set<std::string> unique_hosts;
    for (int i = 0; i < ep_group->nRanks; ++i) {
        unique_hosts.insert(std::string(all_hostnames.data() + i * HOSTNAME_LEN));
    }
    ep_group->nNodes = static_cast<int>(unique_hosts.size());

    ep_group->num_local_experts = ep_group->config.num_experts / ep_group->nRanks;
    ep_group->max_recv_tokens = ep_group->nRanks *
        static_cast<int>(ep_group->config.max_tokens_per_rank);
    ep_group->hidden = ep_group->config.token_size_bytes / ncclTypeSize(ncclBfloat16);

    // Apply default values for auto-configured fields (when set to NCCL_EP_AUTO)
    if (ep_group->config.num_channels == NCCL_EP_AUTO) {
        ep_group->config.num_channels = 10;
    }

    if (ep_group->config.rdma_buffer_size == NCCL_EP_AUTO && ep_group->config.algorithm == NCCL_EP_ALGO_LOW_LATENCY) {
        ep_group->config.rdma_buffer_size = nccl_ep::get_low_latency_rdma_size_hint(ep_group->config.max_tokens_per_rank, ep_group->hidden, ep_group->nRanks, ep_group->config.num_experts);
    }

    if (ep_group->config.num_qp_per_rank == NCCL_EP_AUTO) {
        ep_group->config.num_qp_per_rank = HYBRIDEP_DISPATCH_NUM_OF_BLOCKS * HYBRIDEP_DISPATCH_N2N_WARPS;
    }

    // Physical node properties
    // Normally computed from hostname-based nNodes. When HT is on and the MNNVL
    // LSA domain covers ALL ranks (NVL72 pod intra-cluster), we treat the whole
    // fabric as a single "node" so HT can share buffers directly via NVSwitch
    // using fabric memory handles instead of falling through to RDMA. FULLMESH
    // is MNNVL-only by assertion above, so it always enters this branch.
    bool mnnvl_full_coverage = (hybridep_mode || fullmesh_mode) &&
                               (lsa_team_size == ep_group->nRanks) &&
                               (ep_group->nNodes > 1 || fullmesh_mode);
    if (mnnvl_full_coverage) {
        ep_group->nNodes         = 1;
        ep_group->gpus_per_node  = ep_group->nRanks;
        ep_group->rank_in_node   = ep_group->rank;
        ep_group->node_id        = 0;
        ep_group->lsa_rank_count = ep_group->nRanks;
        ep_group->ht_buffers.use_fabric_memory = true;
    } else {
        ep_group->gpus_per_node  = ep_group->nRanks / ep_group->nNodes;
        ep_group->rank_in_node   = ep_group->cuda_device_id;
        ep_group->node_id        = ep_group->rank / ep_group->gpus_per_node;
        ep_group->lsa_rank_count = ep_group->gpus_per_node;
        ep_group->ht_buffers.use_fabric_memory = false;
    }

    // LSA domain properties (used by LL path)
    ep_group->lsa_team_size  = lsa_team_size;
    ep_group->rdma_ranks     = std::max(1, ep_group->nRanks / lsa_team_size);
    ep_group->rdma_rank      = ep_group->rank / lsa_team_size;
    ep_group->rdma_buffer    = nullptr;

    // [NV72-ADAPT] Phase 0 topology log: verify MNNVL full-coverage path and LSA sizing.
    // Prints once at group-create time from rank 0 so we can tell which HT path is active
    // (cudaIpc vs fabric-memory) and what LSA_TEAM_SIZE the scan kernel will end up using.
    // Phase 2 extends the log with the algorithm name so FULLMESH traces are
    // distinguishable from HT in the same codebase.
    if (ep_group->rank == 0) {
        const char* algo_str = low_latency_mode ? "LL"
                             : (hybridep_mode    ? "HT"
                             : (fullmesh_mode    ? "FULLMESH" : "UNKNOWN"));
        fprintf(stderr,
                "[NV72-ADAPT] topology: algo=%s lsa_team_size=%d nRanks=%d nNodes=%d "
                "mnnvl_full_coverage=%d -> lsa_rank_count=%d gpus_per_node=%d "
                "rank_in_node=%d node_id=%d use_fabric_memory=%d\n",
                algo_str, lsa_team_size, ep_group->nRanks, ep_group->nNodes,
                (int)mnnvl_full_coverage, ep_group->lsa_rank_count,
                ep_group->gpus_per_node, ep_group->rank_in_node, ep_group->node_id,
                (int)ep_group->ht_buffers.use_fabric_memory);
        fflush(stderr);
    }

    CUDA_CHECK(cudaSetDevice(ep_group->cuda_device_id));
    cudaDeviceProp device_prop = {};
    CUDA_CHECK(cudaGetDeviceProperties(&device_prop, ep_group->cuda_device_id));
    ep_group->device_sm_count = device_prop.multiProcessorCount;

    CUDA_CHECK(ep_group->alloc_fn(&ep_group->ep_workspace, NUM_WORKSPACE_BYTES));
    CUDA_CHECK(cudaMemsetAsync(ep_group->ep_workspace, 0, NUM_WORKSPACE_BYTES, stream));

    ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
    NCCL_CHECK_RESULT(ncclCommQueryProperties(ep_group->comm, &props));
    if (!props.deviceApiSupport) {
        fprintf(stderr, "Error: NCCL EP requires NCCL Device API support, but Device API is not supported\n");
        return ncclInvalidUsage;
    }

    // Initialize HT intranode buffers (IPC handles, completion flags, etc.)
    if (hybridep_mode) {
        NCCL_CHECK_RESULT(init_hybridep_intranode(ep_group, in_config, stream));
        NCCL_CHECK_RESULT(init_hybridep_internode(ep_group, in_config, stream));
    }

    // [FULLMESH] Phase 2 commit 2: real fabric allocations.
    //   - recv_buf[nRanks][max_tokens][meta_bytes + token_size_bytes]
    //   - counter_row[nRanks] int32
    // Both exported via CUmemFabricHandle; every peer maps both so commit-4
    // dispatch can atomicAdd on the dest counter and push its payload TMA
    // to the dest recv buffer in a single kernel. Asserted MNNVL coverage
    // earlier (lsa_team_size == nRanks) so the import step cannot fail for
    // topology reasons at this point.
    if (fullmesh_mode) {
        NCCL_CHECK_RESULT(init_fullmesh_intranode_fabric(ep_group, in_config, stream));
    }

    if (ep_group->config.rdma_buffer_size > 0 && low_latency_mode) {
        // Allocate RDMA buffer
        ncclBarrier(ep_group->comm, stream, ep_group->ep_workspace);
        NCCL_CHECK_RESULT(ncclMemAlloc(&ep_group->rdma_buffer, ep_group->config.rdma_buffer_size));

        // Clean buffer (mainly for low-latency mode)
        CUDA_CHECK(cudaMemset(ep_group->rdma_buffer, 0, ep_group->config.rdma_buffer_size));
        // NCCL related setup - use ep_group->comm directly with all GIN contexts
        // (like DeepEP: 1 comm, N contexts, no split needed)
        ep_group->num_nccl_comms = 0;  // no split comms created

        // Cleaning up any pending CUDA error
        CUDA_CHECK(cudaGetLastError());

        // Create device communicator on ep_group->comm with all GIN contexts
        ncclDevComm_t* nccl_dev_comms_host = new ncclDevComm_t[1];
        nccl_dev_comms_host[0] = ncclDevComm_t{};
        ep_group->num_dispatch_signals = ep_group->num_local_experts * ep_group->nRanks;
        int num_total_signals = ep_group->num_dispatch_signals;

        ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
        NCCLCHECK(ncclCommQueryProperties(ep_group->comm, &props));
        if (props.nLsaTeams > 1 && props.ginType == NCCL_GIN_TYPE_NONE) {
            fprintf(stderr, "[LL] Error: NCCL EP requires GIN, but GIN is not supported\n");
            return ncclInvalidUsage;
        }

        ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
        if (props.nLsaTeams > 1) {
            reqs.ginContextCount = ep_group->config.num_qp_per_rank;  // all contexts in single comm
            // Signal layout: combine uses [0, num_total_signals), dispatch uses [num_total_signals, 2*num_total_signals)
            reqs.ginSignalCount = 2 * num_total_signals;
            reqs.ginForceEnable = true;
            reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
        }
        NCCL_CHECK_RESULT(ncclDevCommCreate(ep_group->comm, &reqs, &nccl_dev_comms_host[0]));

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ep_group->nccl_dev_comms),
                                sizeof(ncclDevComm_t)));
        CUDA_CHECK(cudaMemcpy(ep_group->nccl_dev_comms, nccl_dev_comms_host,
                                sizeof(ncclDevComm_t), cudaMemcpyHostToDevice));

        // Register RDMA buffer with single NCCL window
        ncclWindow_t* nccl_wins_host = new ncclWindow_t[1];
        NCCL_CHECK_RESULT(ncclCommWindowRegister(ep_group->comm, ep_group->rdma_buffer,
                                                    ep_group->config.rdma_buffer_size, &nccl_wins_host[0], 0));

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ep_group->nccl_wins),
                                sizeof(ncclWindow_t)));
        CUDA_CHECK(cudaMemcpy(ep_group->nccl_wins, nccl_wins_host,
                                sizeof(ncclWindow_t), cudaMemcpyHostToDevice));

        // Cleanup host memory for NCCL windows and devcomms
        delete[] nccl_wins_host;
        nccl_wins_host = nullptr;

        delete[] nccl_dev_comms_host;
        nccl_dev_comms_host = nullptr;

        ncclBarrier(ep_group->comm, stream, ep_group->ep_workspace);

        CUDA_CHECK(cudaDeviceSynchronize());
    }


    return ncclSuccess;
}

ncclResult_t ncclEpGroupDestroy(
    ncclEpGroup_t ep_group,
    cudaStream_t stream
) {
    if (ep_group == nullptr) {
        return ncclSuccess;
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // Clean up HT intranode resources
    if (ep_group->config.algorithm == NCCL_EP_ALGO_HIGH_THROUGHPUT &&
        ep_group->ht_buffers.initialized) {
        destroy_hybridep_intranode(ep_group);
    }
    // Clean up HT internode resources (GIN deregistration must happen before ncclCommDestroy)
    if (ep_group->config.algorithm == NCCL_EP_ALGO_HIGH_THROUGHPUT &&
        ep_group->ht_buffers.internode_initialized) {
        destroy_hybridep_internode(ep_group);
    }
    // Clean up FULLMESH fabric resources (Phase 2). Same MNNVL-full-coverage
    // precondition as init, so every rank must call this if it called init.
    if (ep_group->config.algorithm == NCCL_EP_ALGO_FULLMESH &&
        ep_group->fullmesh_buffers.initialized) {
        destroy_fullmesh_intranode_fabric(ep_group);
    }
    // Clean up workspace memory
    if (ep_group->ep_workspace != nullptr) {
        CUDA_CHECK(ep_group->free_fn(ep_group->ep_workspace));
    }

    // Clean up RDMA resources (single-comm path: 1 window, 1 devcomm on ep_group->comm)
    if (ep_group->config.rdma_buffer_size > 0 && NCCL_EP_ALGO_LOW_LATENCY == ep_group->config.algorithm) {
        CUDA_CHECK(cudaDeviceSynchronize());
        NCCL_CHECK_RESULT(ncclBarrier(ep_group->comm, stream));

        // Deregister single NCCL window (copy back from device, deregister on ep_group->comm)
        ncclWindow_t win_host;
        CUDA_CHECK(cudaMemcpy(&win_host, ep_group->nccl_wins,
                                sizeof(ncclWindow_t), cudaMemcpyDeviceToHost));
        NCCL_CHECK_RESULT(ncclCommWindowDeregister(ep_group->comm, win_host));
        CUDA_CHECK(cudaFree(ep_group->nccl_wins));
        ep_group->nccl_wins = nullptr;

        // Free RDMA buffer (after window deregistered)
        if (ep_group->rdma_buffer) {
            NCCL_CHECK_RESULT(ncclMemFree(ep_group->rdma_buffer));
            ep_group->rdma_buffer = nullptr;
        }

        // Destroy single NCCL device communicator (copy back from device, destroy on ep_group->comm)
        ncclDevComm_t dc_host;
        CUDA_CHECK(cudaMemcpy(&dc_host, ep_group->nccl_dev_comms,
                                sizeof(ncclDevComm_t), cudaMemcpyDeviceToHost));
        NCCL_CHECK_RESULT(ncclDevCommDestroy(ep_group->comm, &dc_host));
        CUDA_CHECK(cudaFree(ep_group->nccl_dev_comms));
        ep_group->nccl_dev_comms = nullptr;

        // No split comms to destroy (using ep_group->comm directly)
    }
    // Invoke destructor explicitly (placement new was used)
    ep_group->~ncclEpGroup();

    // Free the group structure
    free(ep_group);

    return ncclSuccess;
}

ncclResult_t ncclEpTensorCreate(
    ncclEpGroup_t ep_group,
    ncclNDTensor_t* tensor,
    unsigned int ndim,
    ncclDataType_t datatype,
    ncclEpTensorTag_t tag,
    void* data,
    unsigned int size0,
    unsigned int size1,
    unsigned int size2,
    unsigned int size3,
    unsigned int size4
) {
    assert(ep_group != nullptr);
    assert(tensor != nullptr);
    assert(ndim > 0 && ndim <= 5);

    unsigned int dim_sizes[] = {size0, size1, size2, size3, size4};

    struct ncclNDTensor* t = new struct ncclNDTensor();
    t->version = 1;
    t->ndim = ndim;
    t->datatype = datatype;
    t->tag = tag;
    t->flags = NCCL_EP_TENSOR_FLAG_NONE;

    t->sizes = new unsigned int[ndim];
    t->strides = new unsigned int[ndim];

    unsigned int total_size = 1;
    for (unsigned int i = 0; i < ndim; i++) {
        t->sizes[i] = dim_sizes[i];
        t->strides[i] = 1;
        total_size *= dim_sizes[i];
    }

    if (data != nullptr) {
        t->data = data;
        t->owns_data = false;
    } else {
        CUDA_CHECK(ep_group->alloc_fn(&t->data, total_size * ncclTypeSize(datatype)));
        t->owns_data = true;
    }
    *tensor = t;
    return ncclSuccess;
}

ncclResult_t ncclEpTensorDestroy(
    ncclEpGroup_t ep_group,
    ncclNDTensor_t tensor
) {
    if (tensor == nullptr) return ncclSuccess;

    if (tensor->owns_data && tensor->data && ep_group) {
        CUDA_CHECK(ep_group->free_fn(tensor->data));
    }
    delete[] tensor->sizes;
    delete[] tensor->strides;
    delete tensor;

    return ncclSuccess;
}

ncclResult_t ncclEpTensorGetData(
    ncclNDTensor_t tensor,
    void** data
) {
    assert(tensor != nullptr);
    assert(data != nullptr);
    *data = tensor->data;
    return ncclSuccess;
}

ncclResult_t ncclEpTensorGetSizes(
    ncclNDTensor_t tensor,
    const unsigned int** sizes,
    unsigned int* ndim
) {
    assert(tensor != nullptr);
    if (sizes != nullptr) *sizes = tensor->sizes;
    if (ndim != nullptr) *ndim = tensor->ndim;
    return ncclSuccess;
}

struct ncclEpHandle {
    ncclEpGroup_t group;

    bool use_fp8;

    // tensor that is owned by the user, do not free this tensor!
    ncclNDTensor_t topk_idx;
    int num_tokens, num_topk;

    bool cached_mode;
    int num_scales;
    int hidden_int4;

    union {
        struct {
            // packed tensors for LL
            ncclNDTensor_t expert_recv_source_indices;
            ncclNDTensor_t expert_dispatch_layout;

            int buffer_idx = 0;

            std::function<void(unsigned int)> continue_fn;
            nccl_ep::LowLatencyLayout layout;
        } ll;
        struct {

             // Global routing map: allgathered routing decisions from ALL ranks (bitmap format).
             // Contains complete routing information needed for preprocessing.
             // dtype: uint8_t (bitmap: 1 bit per expert, 8 experts per byte)
             // layout: [total_tokens, ceil(num_experts / 8)]
             // lifetime: valid after ncclAllGather in ncclEpCreateHandle
            uint8_t* global_routing_map;

             // =================================================================================
             // PREPROCESSING OUTPUTS - Computed once per iteration, used by dispatch & combine
             // =================================================================================

             // Sparse-to-dense map: maps each (token, source_rank) pair to its position in
             // the destination rank's expert buffer. Used by NVLink (intra-node) warps.
             // Value of -1 indicates token is not routed to that rank.
             // dtype: int32_t
             // layout: [num_nodes * max_tokens_per_rank, num_ranks_per_node]
             // usage: dispatch S2G warp group, combine G2S warp group
             // lifetime: valid after metadata_preprocessing, constant within iteration
             // vs NCCL HT: no direct equivalent.
             //   - HT: uses is_token_in_rank (local tokens only) + atomics to compute positions
             //   - HT: precomputes positions for ALL nodes' tokens, -1 sentinel for not routed
            int32_t* sparse_to_dense_map;

             // RDMA-to-attention map: boolean mask indicating which tokens this node needs
             // to RECEIVE from RDMA (inter-node). Indexed by [node_id, token_id].
             // Primarily used during combine to know which remote tokens to wait for.
             // dtype: bool
             // layout: [num_nodes, max_tokens_per_rank_padded_to_16]
             //        (padding to 16 required for TMA alignment)
             // usage: dispatch G2S warp (polling), combine inter-node G2S/reduction warps
             // lifetime: valid after metadata_preprocessing, constant within iteration
             // vs NCCL HT: inverse perspective of is_token_in_rank.
             //   - is_token_in_rank: outbound - "where do MY tokens go?" [my_token, dest_rank]
             //   - rdma_to_attn_map: inbound - "which remote tokens do I receive?" [src_node, token]
            bool* rdma_to_attn_map;

             // Attention-to-RDMA map: boolean mask indicating which local tokens need to be
             // SENT via RDMA (inter-node) to each remote node.
             // Only allocated when num_nodes > 1.
             // dtype: bool
             // layout: [max_tokens_per_rank, num_nodes - 1]
             // usage: dispatch N2N (RDMA) warp group
             // lifetime: valid after metadata_preprocessing, constant within iteration
             // vs NCCL HT: closest equivalent to is_token_in_rank for inter-node RDMA.
             //   - is_token_in_rank: per-rank granularity [num_tokens, num_ranks]
             //   - attn_to_rdma_map: per-node granularity [num_tokens, num_nodes-1] (RDMA only)
            bool* attn_to_rdma_map;

            // Per-token per-rank bitmask cache produced during preprocessing.
            // dtype: uint32_t (was uint8_t; widened so each bit still maps to
            // one rank for LSA_TEAM_SIZE up to 32; EP16/EP32 HT combine was
            // silently dropping rank>=8 with uint8_t, see 1bfa83d for the
            // [VALIDATE-DEBUG] evidence).
            // layout: [num_nodes * max_tokens_per_rank * ranks_per_node]
            uint32_t* token_rank_mask;

             // Local expert routing map: per-expert routing for tokens in this rank's buffer.
             // Used by subsequent expert MLP layers to route tokens to correct experts.
             // dtype: bool
             // layout: [max_recv_tokens, experts_per_rank]
             //        where max_recv_tokens = num_ranks * max_tokens_per_rank
             // usage: passed to expert computation layers (not used by dispatch/combine directly)
             // lifetime: valid after metadata_preprocessing, constant within iteration
             // vs NCCL HT: similar purpose to num_tokens_per_expert but more detailed
            bool* local_expert_routing_map;

             // Number of tokens routed to local experts (total across all local experts).
             // Each token counted once even if routed to multiple local experts.
             // dtype: int32_t
             // layout: [1]
             // usage: buffer sizing, iteration control
             // lifetime: valid after metadata_preprocessing
            int32_t* num_tokens_for_experts;

            // Host-pinned mirror of num_tokens_for_experts (single scalar).
            // Filled once during handle creation on the same stream as preprocessing.
            int32_t* num_tokens_for_experts_host;

             // =================================================================================
             // CONVERSION BUFFERS - Pre-allocated to avoid dispatch/combine-time malloc
             // =================================================================================

             // Dense prob buffer: shared scratch for sparse↔dense conversions
             // dtype: float
             // layout: [max_tokens_per_rank, num_experts]
             // usage:
             //   - dispatch forward: sparse→dense input topk_weights conversion
             //   - combine backward: dense→sparse output prob conversion
             // lifetime: allocated at handle creation, freed at handle destroy
             // note: dispatch and combine are sequential, so one buffer suffices
            // For multi-node: points to group-level pre-registered buffer
            // For single-node: handle-owned buffer
            float* dense_prob_buffer;

            // Token staging buffer: pre-registered buffer to avoid GIN registration during dispatch
            // User tokens are copied here during dispatch, then this buffer is used for RDMA
            // dtype: uint16_t (bf16) or uint8_t (fp8)
            // layout: [max_tokens_per_rank, hidden]
            // usage: copy user tokens → use for inter-node RDMA
            // lifetime: group-owned (allocated in Group Create, freed in Group Destroy)
            void* token_staging_buffer;  // Pointer to group-level buffer (not handle-owned)

            // Scaling factor staging buffer: pre-registered buffer for FP8 scaling factors
            // User scaling factors are copied here during dispatch, then this buffer is used for RDMA
            // dtype: float
            // layout: [max_tokens_per_rank]
            // usage: copy user scaling factors → use for inter-node RDMA
            // lifetime: group-owned (allocated in Group Create, freed in Group Destroy)
            float* scaling_factor_staging_buffer;  // Pointer to group-level buffer (not handle-owned)

            // RDMA inter-node group flags: atomic completion flags for each remote node.
            // Remote ranks increment via RDMA atomic fetch-add to signal chunk completion.
            // Only allocated when num_nodes > 1.
            // dtype: uint64_t
            // layout: [num_nodes - 1]
            // usage: dispatch N2N warp (signaling), dispatch G2S warp (polling)
            // lifetime: reset to 0 at init, incremented by remote RDMA atomics
            //uint64_t* rdma_inter_node_group_flags;

            // Per-handle preprocessing block (single allocation for all preprocessing buffers)
            void* preprocessing_block;
            size_t preprocessing_zero_region_size;
            size_t preprocessing_s2d_size;
            void* preprocessing_scan_tmp;
            int32_t* per_expert_counts_tmp;
        } hybridep;
    };

    ncclEpHandle()
        : group(nullptr),
          topk_idx(nullptr),
          num_tokens(0),
          num_topk(0),
          cached_mode(false),
          num_scales(0),
          hidden_int4(0) {
        constexpr size_t union_size = std::max(sizeof(ll), sizeof(hybridep));
        memset(static_cast<void*>(&ll), 0, union_size);
    }

    ~ncclEpHandle() {
    }
};

static bool tensor_is_contiguous(ncclNDTensor_t tensor) {
    for (unsigned int i = 0; i < tensor->ndim; i++)
        if (tensor->strides[i] != 1)
            return false;
    return true;
}

// static const ncclNDTensor_t* find_tensor_by_tag(const ncclNDTensor_t* const* tensors, int num_tensors, ncclEpTensorTag_t tag) {
//     for (int i = 0; i < num_tensors; i++) {
//         if (tensors[i]->tag == tag)
//             return tensors[i];
//     }
//     return nullptr;
// }

static ncclNDTensor_t find_tensor_by_tag(const ncclNDTensor_t* tensors, int num_tensors, ncclEpTensorTag_t tag) {
    for (int i = 0; i < num_tensors; i++) {
        if (tensors[i]->tag == tag)
            return tensors[i];
    }
    return nullptr;
}

static bool is_internode_available(ncclEpGroup_t ep_group) {
    // True when there are multiple physical nodes. HT always uses node-based RDMA across nodes,
    // including MNNVL (which has NVLink fabric but is treated as regular multi-node for HT).
    return ep_group->nNodes > 1;
}


static void tensor_free(ncclEpGroup_t group, ncclNDTensor_t t) {
    if (t == nullptr) return;
    if (t->owns_data && t->data)
        group->free_fn(t->data);
    if (t->strides)
        delete[] t->strides;
    if (t->sizes)
        delete[] t->sizes;
    delete t;
}

ncclResult_t ncclEpCreateHandle(
    ncclEpHandle_t* out_handle,
    ncclEpGroup_t ep_group,
    ncclNDTensor_t topk_idx,
    const ncclNDTensor_t* local_tensors,
    unsigned int num_local_tensors,
    const ncclEpHandleConfig_t* config,
    cudaStream_t stream,
    bool use_fp8
) {
    assert(topk_idx != nullptr);
    assert(ep_group != nullptr);
    assert(out_handle != nullptr);
    assert(config == nullptr);

    // Validate communicator
    int nRanks = 0;
    assert(ep_group->comm != nullptr);
    assert(ncclCommCount(ep_group->comm, &nRanks) == ncclSuccess);
    assert(nRanks > 0);

    // Validate tensor properties
    assert(topk_idx->ndim == 2);
    assert(topk_idx->datatype == ncclInt64);
    assert(topk_idx->tag == NCCL_EP_TENSOR_TAG_TOPK_IDX);
    assert(tensor_is_contiguous(topk_idx));

    // Validate expert configuration
    assert(ep_group->config.num_experts > 0);
    assert(ep_group->config.num_experts % ep_group->nRanks == 0);

    // Create and initialize handle
    *out_handle = new ncclEpHandle();
    ncclEpHandle_t handle = *out_handle;

    handle->group = ep_group;
    handle->use_fp8 = use_fp8;
    handle->topk_idx = topk_idx;

    // Extract tensor dimensions
    const int num_tokens = static_cast<int>(topk_idx->sizes[0]);
    const int num_topk = static_cast<int>(topk_idx->sizes[1]);
    handle->num_tokens = num_tokens;
    handle->num_topk = num_topk;

    if (ep_group->config.algorithm == NCCL_EP_ALGO_LOW_LATENCY) {
        // LL mode does not accept local tensors
        assert(num_local_tensors == 0 && "LL mode does not accept local tensors in ncclEpCreateHandle");

        // Allocate packed tensors
        // packed_recv_x is the input tensor in the dispatch
        ncclEpTensorCreate(ep_group, &handle->ll.expert_recv_source_indices, 2, ncclInt32, NCCL_EP_TENSOR_TAG_NONE, nullptr, static_cast<unsigned int>(handle->group->num_local_experts), static_cast<unsigned int>(ep_group->nRanks * ep_group->config.max_tokens_per_rank));
        ncclEpTensorCreate(ep_group, &handle->ll.expert_dispatch_layout, 2, ncclInt64, NCCL_EP_TENSOR_TAG_NONE, nullptr, static_cast<unsigned int>(handle->group->num_local_experts), static_cast<unsigned int>(ep_group->nRanks));

        assert((ep_group->config.max_tokens_per_rank * handle->group->num_local_experts) % 4 == 0 and "TMA requires the number of tokens to be multiple of 4");

        handle->ll.layout = nccl_ep::LowLatencyLayout(handle->group->rdma_buffer, handle->group->config.max_tokens_per_rank, handle->group->hidden, handle->group->nRanks, handle->group->config.num_experts);

        assert(handle->ll.layout.total_bytes <= handle->group->config.rdma_buffer_size);
    } else if (ep_group->config.algorithm == NCCL_EP_ALGO_FULLMESH) {
        // [FULLMESH] Phase 2 commit 1 stub: zero-preprocess handle.
        // Real preprocess (counter reset, per-iter topk cache, optional local slot
        // table for the src-side path) arrives with the dispatch kernel in commit 4.
        // For now we only record num_tokens/num_topk and leave every union field
        // at its default so ncclEpHandleDestroy has nothing to free.
        assert(num_local_tensors == 0 && "FULLMESH commit-1 stub does not accept local tensors");
        assert(handle->num_tokens <= static_cast<int>(ep_group->config.max_tokens_per_rank) &&
               "FULLMESH: token count exceeds buffer capacity");
        if (ep_group->rank == 0) {
            fprintf(stderr,
                    "[FULLMESH] ncclEpCreateHandle stub: num_tokens=%d num_topk=%d "
                    "num_experts=%u nRanks=%d (no preprocessing, commit-1 skeleton)\n",
                    handle->num_tokens, handle->num_topk,
                    ep_group->config.num_experts, ep_group->nRanks);
            fflush(stderr);
        }
    } else { // HT
        assert(ep_group->config.max_tokens_per_rank > 0 && "HT requires max_tokens_per_rank > 0");
        assert(handle->num_tokens <= static_cast<int>(ep_group->config.max_tokens_per_rank) && "Token count exceeds HT buffer capacity");

        // Optional: per-expert token counts output
        ncclNDTensor_t recv_expert_counter = nullptr;
        if (num_local_tensors > 0) {
            recv_expert_counter = find_tensor_by_tag(local_tensors, num_local_tensors, NCCL_EP_TENSOR_TAG_RECV_EXPERT_COUNTER_HOST);
            if (recv_expert_counter == nullptr) {
                recv_expert_counter = find_tensor_by_tag(local_tensors, num_local_tensors, NCCL_EP_TENSOR_TAG_RECV_EXPERT_COUNTER_DEVICE);
            }
        }

        const int nRanks = ep_group->nRanks;
        const int num_experts = ep_group->config.num_experts;
        const int max_tokens = ep_group->config.max_tokens_per_rank;
        const int n_ranks_per_node = ep_group->lsa_rank_count;
        const int nNodes = ep_group->nNodes;
        const int experts_per_rank = ep_group->num_local_experts;

        // Allocate preprocessing buffers per-handle (single block allocation)
        {
            auto align256 = [](size_t s) -> size_t { return (s + 255) & ~size_t(255); };

            const int total_send_tokens = nRanks * max_tokens;
            const int padded_max_tokens = ((max_tokens + 15) / 16) * 16;
            const int num_experts_packed = (num_experts + 7) / 8;

            size_t sz_routing   = align256(static_cast<size_t>(total_send_tokens) * num_experts_packed);
            size_t sz_r2a       = align256(static_cast<size_t>(nNodes) * padded_max_tokens * sizeof(bool));
            size_t sz_a2r       = (nNodes > 1) ? align256(static_cast<size_t>(max_tokens) * (nNodes - 1) * sizeof(bool)) : 0;
            size_t sz_ler       = align256(static_cast<size_t>(ep_group->max_recv_tokens) * experts_per_rank * sizeof(bool));
            size_t sz_ntfe      = align256(sizeof(int32_t));
            size_t zero_region  = sz_routing + sz_r2a + sz_a2r + sz_ler + sz_ntfe;

            size_t sz_s2d       = align256(static_cast<size_t>(nNodes) * max_tokens * n_ranks_per_node * sizeof(int32_t));
            size_t sz_rank_mask = align256(static_cast<size_t>(nNodes) * max_tokens * n_ranks_per_node * sizeof(uint32_t));

            size_t sz_scan_tmp  = align256(nccl_ep::hybridep::get_preprocessing_scan_tmp_size(n_ranks_per_node));
            size_t sz_prob      = !is_internode_available(ep_group) ? align256(static_cast<size_t>(max_tokens) * num_experts * sizeof(float)) : 0;
            size_t sz_per_expert_counts_tmp = align256(static_cast<size_t>(experts_per_rank) * sizeof(int32_t));
            size_t no_memset_region = sz_rank_mask + sz_scan_tmp + sz_prob + sz_per_expert_counts_tmp;

            size_t total_size = zero_region + sz_s2d + no_memset_region;

            CUDA_CHECK(ep_group->alloc_fn(&handle->hybridep.preprocessing_block, total_size));
            handle->hybridep.preprocessing_zero_region_size = zero_region;
            handle->hybridep.preprocessing_s2d_size = sz_s2d;

            char* ptr = static_cast<char*>(handle->hybridep.preprocessing_block);
            size_t offset = 0;

            handle->hybridep.global_routing_map = reinterpret_cast<uint8_t*>(ptr + offset);
            offset += sz_routing;

            handle->hybridep.rdma_to_attn_map = reinterpret_cast<bool*>(ptr + offset);
            offset += sz_r2a;

            handle->hybridep.attn_to_rdma_map = (nNodes > 1) ? reinterpret_cast<bool*>(ptr + offset) : nullptr;
            offset += sz_a2r;

            handle->hybridep.local_expert_routing_map = reinterpret_cast<bool*>(ptr + offset);
            offset += sz_ler;

            handle->hybridep.num_tokens_for_experts = reinterpret_cast<int32_t*>(ptr + offset);
            offset += sz_ntfe;

            handle->hybridep.sparse_to_dense_map = reinterpret_cast<int32_t*>(ptr + offset);
            offset += sz_s2d;

            handle->hybridep.token_rank_mask = reinterpret_cast<uint32_t*>(ptr + offset);
            offset += sz_rank_mask;

            handle->hybridep.preprocessing_scan_tmp = reinterpret_cast<void*>(ptr + offset);
            offset += sz_scan_tmp;

            if (!is_internode_available(ep_group)) {
                // Single physical node: allocate handle-local buffer
                handle->hybridep.dense_prob_buffer = reinterpret_cast<float*>(ptr + offset);
                offset += sz_prob;
            } else {
                handle->hybridep.dense_prob_buffer = nullptr;
            }

            handle->hybridep.per_expert_counts_tmp = reinterpret_cast<int32_t*>(ptr + offset);
            offset += sz_per_expert_counts_tmp;
        }

        // For multi-node: dense_prob_buffer is the group-level GIN-registered buffer
        if (is_internode_available(ep_group)) {
            handle->hybridep.dense_prob_buffer = ep_group->ht_buffers.dense_prob_buffer;
        }

        // Staging buffers (group-level, GIN-registered)
        if (is_internode_available(ep_group)) {
            handle->hybridep.token_staging_buffer = ep_group->ht_buffers.token_staging_buffer;
            handle->hybridep.scaling_factor_staging_buffer = ep_group->ht_buffers.scaling_factor_staging_buffer;
        } else {
            handle->hybridep.token_staging_buffer = nullptr;
            handle->hybridep.scaling_factor_staging_buffer = nullptr;
        }

        // Cache actual received token count on host once per handle.
        CUDA_CHECK(cudaHostAlloc(
            reinterpret_cast<void**>(&handle->hybridep.num_tokens_for_experts_host),
            sizeof(int32_t),
            cudaHostAllocDefault));
        *handle->hybridep.num_tokens_for_experts_host = -1;

        // Initialize only the regions touched by this handle's num_tokens.
        // Keep allocation capacities unchanged to preserve buffer layout/contracts.
        const int num_experts_packed = (num_experts + 7) / 8;
        const int padded_num_tokens = ((handle->num_tokens + 15) / 16) * 16;  // rdma_to_attn_map uint4 loads
        const size_t routing_send_bytes = static_cast<size_t>(handle->num_tokens) * num_experts_packed;
        const size_t rdma_to_attn_bytes = static_cast<size_t>(nNodes) * padded_num_tokens * sizeof(bool);
        const size_t attn_to_rdma_bytes = (nNodes > 1) ?
            static_cast<size_t>(handle->num_tokens) * (nNodes - 1) * sizeof(bool) : 0;
        const size_t local_expert_routing_bytes =
            static_cast<size_t>(ep_group->max_recv_tokens) * experts_per_rank * sizeof(bool);
        const size_t sparse_to_dense_bytes =
            static_cast<size_t>(nNodes) * handle->num_tokens * n_ranks_per_node * sizeof(int32_t);

        // convert_topk_to_routing_map uses bitwise-OR into this buffer, so local send rows must be pre-zeroed.
        uint8_t* local_routing_send_ptr =
            handle->hybridep.global_routing_map + (max_tokens * num_experts_packed) * ep_group->rank;
        if (routing_send_bytes > 0) {
            CUDA_CHECK(cudaMemsetAsync(local_routing_send_ptr, 0, routing_send_bytes, stream));
        }
        if (rdma_to_attn_bytes > 0) {
            CUDA_CHECK(cudaMemsetAsync(handle->hybridep.rdma_to_attn_map, 0, rdma_to_attn_bytes, stream));
        }
        if (attn_to_rdma_bytes > 0) {
            CUDA_CHECK(cudaMemsetAsync(handle->hybridep.attn_to_rdma_map, 0, attn_to_rdma_bytes, stream));
        }
        // Keep full clear for local_expert_routing_map to preserve current dispatch output behavior
        // when callers allocate outputs at max receive capacity.
        CUDA_CHECK(cudaMemsetAsync(
            handle->hybridep.local_expert_routing_map, 0, local_expert_routing_bytes, stream));
        CUDA_CHECK(cudaMemsetAsync(
            handle->hybridep.num_tokens_for_experts, 0, sizeof(*handle->hybridep.num_tokens_for_experts), stream));
        if (sparse_to_dense_bytes > 0) {
            CUDA_CHECK(cudaMemsetAsync(
                handle->hybridep.sparse_to_dense_map, 0xFF, sparse_to_dense_bytes, stream));
        }

        // ===== Step 1: Convert sparse topk_idx to bitmap routing map =====
        nccl_ep::hybridep::convert_topk_to_routing_map(
            static_cast<const int64_t*>(topk_idx->data),
            local_routing_send_ptr,
            handle->num_tokens,
            handle->num_topk,
            num_experts_packed,
            stream);

        // ===== Step 2: Allgather bitmap routing maps =====
        NCCL_CHECK_RESULT(ncclAllGather(
            local_routing_send_ptr,
            handle->hybridep.global_routing_map,
            static_cast<size_t>(handle->num_tokens) * num_experts_packed,
            ncclUint8,
            ep_group->comm,
            stream));

        // ===== Step 3: Run metadata_preprocessing =====
        int32_t* per_expert_counts_device = nullptr;
        bool copy_per_expert_counts_to_host = false;
        if (recv_expert_counter != nullptr) {
            if (recv_expert_counter != nullptr) {
                assert(recv_expert_counter->ndim == 1 && "recv_expert_counter must be 1D");
                assert(recv_expert_counter->datatype == ncclInt32 && "recv_expert_counter must be ncclInt32");
                assert(recv_expert_counter->sizes[0] >= static_cast<unsigned int>(ep_group->num_local_experts) &&
                    "recv_expert_counter size must be >= num_local_experts");
                assert(recv_expert_counter->data != nullptr && "recv_expert_counter data must not be null");
            }

            if (recv_expert_counter->tag == NCCL_EP_TENSOR_TAG_RECV_EXPERT_COUNTER_HOST) {
                // Fused preprocessing uses atomics; avoid atomics directly into mapped host memory.
                per_expert_counts_device = handle->hybridep.per_expert_counts_tmp;
                copy_per_expert_counts_to_host = true;
            } else {
                per_expert_counts_device = static_cast<int32_t*>(recv_expert_counter->data);
            }
        }

        // [NV72-ADAPT] Phase 0: log template params about to be instantiated.
        // The pair (nNodes, n_ranks_per_node) is what HYBRIDEP_SWITCH_NUM_LSA_TEAMS +
        // HYBRIDEP_SWITCH_LSA_TEAM_SIZE switch on, so this tells us exactly which
        // template specialization the scan kernel will execute.
        if (ep_group->rank == 0) {
            fprintf(stderr,
                    "[NV72-ADAPT] call_metadata_preprocessing: "
                    "nNodes=%d n_ranks_per_node=%d experts_per_rank=%d hidden=%d tokens=%d "
                    "-> LSA_TEAM_SIZE=%d NUM_LSA_TEAMS=%d\n",
                    nNodes, n_ranks_per_node, experts_per_rank, ep_group->hidden,
                    handle->num_tokens, n_ranks_per_node, nNodes);
            fflush(stderr);
        }

        nccl_ep::hybridep::call_metadata_preprocessing(
            handle->hybridep.global_routing_map,
            handle->hybridep.sparse_to_dense_map,
            handle->hybridep.rdma_to_attn_map,
            handle->hybridep.attn_to_rdma_map,
            handle->hybridep.token_rank_mask,
            handle->hybridep.num_tokens_for_experts,
            handle->hybridep.local_expert_routing_map,
            per_expert_counts_device,
            handle->hybridep.preprocessing_scan_tmp,
            ep_group->node_id,
            ep_group->rank_in_node,
            handle->num_tokens,
            ep_group->hidden,
            nNodes,
            n_ranks_per_node,
            experts_per_rank,
            stream);

        if (copy_per_expert_counts_to_host) {
            CUDA_CHECK(cudaMemcpyAsync(
                recv_expert_counter->data,
                per_expert_counts_device,
                static_cast<size_t>(experts_per_rank) * sizeof(int32_t),
                cudaMemcpyDeviceToHost,
                stream));
        }

        // Mirror the actual recv-token count to host once; getters can read this cached value.
        CUDA_CHECK(cudaMemcpyAsync(
            handle->hybridep.num_tokens_for_experts_host,
            handle->hybridep.num_tokens_for_experts,
            sizeof(*handle->hybridep.num_tokens_for_experts_host),
            cudaMemcpyDeviceToHost,
            stream));
    }

    return ncclSuccess;
    }

ncclResult_t ncclEpHandleDestroy(
    ncclEpHandle_t handle
) {
    if (!handle)
        return ncclSuccess;

    if (handle->group->config.algorithm == NCCL_EP_ALGO_LOW_LATENCY) {
        tensor_free(handle->group, handle->ll.expert_recv_source_indices);
        tensor_free(handle->group, handle->ll.expert_dispatch_layout);
    } else if (handle->group->config.algorithm == NCCL_EP_ALGO_HIGH_THROUGHPUT) {
        if (handle->hybridep.num_tokens_for_experts_host) {
            CUDA_CHECK(cudaFreeHost(handle->hybridep.num_tokens_for_experts_host));
            handle->hybridep.num_tokens_for_experts_host = nullptr;
        }
        if (handle->hybridep.preprocessing_block) {
            handle->group->free_fn(handle->hybridep.preprocessing_block);
            handle->hybridep.preprocessing_block = nullptr;
        }
    }
    // FULLMESH commit-1 stub allocates nothing in ncclEpCreateHandle, so destroy
    // is a no-op. Per-handle resources (my_slot_at_dest cache, counter reset
    // state) arrive with commit 3 and will be cleaned up here.

    delete handle;
    return ncclSuccess;
}

    // EP Operations

ncclResult_t ncclEpDispatch(
    ncclEpHandle_t handle,
    const ncclNDTensor_t* inputs,
    unsigned int num_inputs,
    const ncclNDTensor_t* outputs,
    unsigned int num_outputs,
    const ncclNDTensor_t* local_tensors,
    unsigned int num_local_tensors,
    unsigned int send_only,
    const ncclEpDispatchConfig_t* config,
    cudaStream_t stream
) {
        ncclEpGroup_t group = handle->group;
    if (group->config.algorithm == NCCL_EP_ALGO_LOW_LATENCY) {
        ncclNDTensor_t x = find_tensor_by_tag(
            inputs, num_inputs, NCCL_EP_TENSOR_TAG_TOKENS
        );
        assert(x != nullptr);
        assert(x->ndim == 2);
        assert(tensor_is_contiguous(x));
        assert(x->datatype == ncclBfloat16);
        assert(x->sizes[0] == handle->num_tokens);
        assert(x->sizes[0] <= group->config.max_tokens_per_rank);
        assert(x->sizes[1] % sizeof(int4) == 0);
        assert(x->sizes[1] % 128 == 0);
        assert(x->sizes[1] * ncclTypeSize(x->datatype) == group->config.token_size_bytes);

        // Find and validate output tensors
        ncclNDTensor_t recv_x = find_tensor_by_tag(
            outputs, num_outputs, NCCL_EP_TENSOR_TAG_TOKENS
        );
        ncclNDTensor_t scales = find_tensor_by_tag(
            outputs, num_outputs, NCCL_EP_TENSOR_TAG_SCALES
        );
        assert(recv_x != nullptr);
        assert(recv_x->ndim == 3);
        assert(tensor_is_contiguous(recv_x));
        assert(recv_x->sizes[0] == group->num_local_experts);
        assert(recv_x->sizes[1] == group->nRanks * group->config.max_tokens_per_rank);
        assert(recv_x->sizes[2] == group->hidden);

        if (scales != nullptr) {
            constexpr int scale_block_size = 128;
            assert(group->hidden % 512 == 0);
            assert(scales->ndim == 3);
            assert(tensor_is_contiguous(scales));
            assert(scales->datatype == ncclFloat32);
            assert(scales->sizes[0] == group->num_local_experts);
            assert(scales->sizes[1] == group->config.max_tokens_per_rank * group->nRanks);
            assert(scales->sizes[2] == group->hidden / scale_block_size);
        }

        ncclNDTensor_t recv_count = find_tensor_by_tag(
            local_tensors, num_local_tensors, NCCL_EP_TENSOR_TAG_RECV_EXPERT_COUNTER_DEVICE
        );

        assert(recv_count != nullptr);
        assert(recv_count->ndim == 1);
        assert(tensor_is_contiguous(recv_count));
        assert(recv_count->datatype == ncclInt32);
        assert(recv_count->sizes[0] == group->num_local_experts);

        const auto& buffer = handle->ll.layout.buffers[handle->ll.buffer_idx];
        auto& next_buffer = handle->ll.layout.buffers[handle->ll.buffer_idx ^= 1];
        const auto next_clean_meta = next_buffer.clean_meta();

        unsigned signal_base = group->num_dispatch_signals;
        auto dispatch_fn = [=](int phases) {
            // Prepare data pointers
            auto* recv_x_data = recv_x->data;
            auto* scales_data = scales ? scales->data : nullptr;
            auto* expert_recv_source_indices_data = static_cast<int*>(handle->ll.expert_recv_source_indices->data);
            auto* expert_dispatch_layout_data = static_cast<int64_t*>(handle->ll.expert_dispatch_layout->data);
            auto* recv_count_data = static_cast<int*>(recv_count->data);
            auto* x_data = x->data;
            auto* topk_idx_data = static_cast<int64_t*>(handle->topk_idx->data);

            const bool use_fp8 = (scales != nullptr);
            const bool round_scale = config->round_scales;
            const bool use_ue8m0 = false;

            nccl_ep::internode_ll::dispatch(
                x_data,
                topk_idx_data,
                recv_x_data,
                scales_data,
                expert_recv_source_indices_data,
                expert_dispatch_layout_data,
                recv_count_data,
                buffer.dispatch_rdma_send_buffer,
                buffer.dispatch_rdma_recv_data_buffer,
                buffer.dispatch_rdma_recv_count_buffer,
                (reinterpret_cast<uint64_t>(buffer.dispatch_rdma_send_buffer) - reinterpret_cast<uint64_t>(group->rdma_buffer)),
                (reinterpret_cast<uint64_t>(buffer.dispatch_rdma_recv_data_buffer) - reinterpret_cast<uint64_t>(group->rdma_buffer)),
                (reinterpret_cast<uint64_t>(buffer.dispatch_rdma_recv_count_buffer) - reinterpret_cast<uint64_t>(group->rdma_buffer)),
                next_clean_meta.first,
                next_clean_meta.second,
                nullptr, /*recv_send_sizes=*/
                nullptr, /*recv_send_offsets=*/
                handle->num_tokens,
                group->hidden,
                group->config.max_tokens_per_rank,
                handle->num_topk,
                group->config.num_experts,
                group->rank,
                group->nRanks,
                use_fp8,
                round_scale,
                use_ue8m0,
                phases,
                group->num_nccl_comms,
                group->nccl_dev_comms,
                group->nccl_wins,
                signal_base,
                group->ep_workspace,
                group->device_sm_count,
                stream
            );
        };

        // Execute dispatch with appropriate phase flags
        const int dispatch_phases = send_only ?
            LOW_LATENCY_SEND_PHASE :
            (LOW_LATENCY_SEND_PHASE | LOW_LATENCY_RECV_PHASE);
        dispatch_fn(dispatch_phases);

        if (send_only) {
            handle->ll.continue_fn = dispatch_fn;
        }
    } else if (group->config.algorithm == NCCL_EP_ALGO_FULLMESH) {
        // [FULLMESH] Phase 2 commit 3: real dispatch path.
        //   (a) cudaMemsetAsync on this rank's counter_row so peers start
        //       every iter from slot 0. On-stream; ncclBarrier below forces
        //       cross-rank ordering.
        //   (b) ncclBarrier (Q1=C host wrapper): cudaDeviceSynchronize inside
        //       makes sure every rank has finished its counter reset before
        //       any rank issues an atomicAdd at a peer.
        //   (c) launch_dispatch_kernel: per (token, k) pair, atomicAdd on
        //       peer_counter_vas_dev[dest][myRank] for a slot, then store
        //       (meta, payload) at peer_recv_vas_dev[dest] entry.
        //   (d) ncclBarrier (Q1=C host wrapper): kernel completion + cross-
        //       rank readiness before this rank reads its own recv_buf.
        //   (e) launch_compact_to_output: cudaMemcpy2D strips 8B per-entry
        //       meta and writes the payload into the user's dense output
        //       tensor ([nRanks * max_tokens_per_rank, hidden]).
        //
        // Two ncclBarrier calls per dispatch is the commit-3 correctness-first
        // design; each contains a cudaDeviceSynchronize so CUDA Graph capture
        // is NOT supported yet. If profiling shows this to be the dominant
        // cost, commit 6 can revisit with ncclBarrierSession (Q1=C2) which is
        // stream-only. Functional correctness first.
        assert(num_local_tensors == 0 && "FULLMESH dispatch does not accept local_tensors");
        assert(group->fullmesh_buffers.initialized && "FULLMESH buffers not init'd");

        ncclNDTensor_t x       = find_tensor_by_tag(inputs,  num_inputs,  NCCL_EP_TENSOR_TAG_TOKENS);
        ncclNDTensor_t topk    = find_tensor_by_tag(inputs,  num_inputs,  NCCL_EP_TENSOR_TAG_TOPK_IDX);
        ncclNDTensor_t recv_x  = find_tensor_by_tag(outputs, num_outputs, NCCL_EP_TENSOR_TAG_TOKENS);
        assert(x      != nullptr && "FULLMESH dispatch missing input TOKENS tensor");
        assert(topk   != nullptr && "FULLMESH dispatch missing input TOPK_IDX tensor");
        assert(recv_x != nullptr && "FULLMESH dispatch missing output TOKENS tensor");
        assert(x->ndim    == 2 && tensor_is_contiguous(x));
        assert(topk->ndim == 2 && tensor_is_contiguous(topk) && topk->datatype == ncclInt64);
        assert(recv_x->ndim == 2 && tensor_is_contiguous(recv_x));
        assert(x->sizes[0] == handle->num_tokens);
        assert(x->sizes[0] <= group->config.max_tokens_per_rank);

        auto& fb = group->fullmesh_buffers;
        const int nRanks           = group->nRanks;
        const int myRank           = group->rank;
        const int max_tokens       = static_cast<int>(group->config.max_tokens_per_rank);
        const int hidden_bytes     = static_cast<int>(group->config.token_size_bytes);
        const int num_local_exps   = group->num_local_experts;

        // (a) Reset this rank's counter row on the group stream.
        CUDA_CHECK(cudaMemsetAsync(reinterpret_cast<void*>(fb.counter_local_va),
                                   0, fb.bytes_per_counter_row, stream));

        // (b) Cross-rank barrier so no peer atomicAdds our counter before reset finishes.
        NCCL_CHECK_RESULT(ncclBarrier(group->comm, stream, group->ep_workspace));

        // (c) Fused atomicAdd-slot + payload push kernel.
        nccl_ep::fullmesh::launch_dispatch_kernel(
            x->data,
            static_cast<const int64_t*>(topk->data),
            fb.peer_recv_vas_dev,
            fb.peer_counter_vas_dev,
            handle->num_tokens,
            handle->num_topk,
            num_local_exps,
            myRank,
            nRanks,
            max_tokens,
            hidden_bytes,
            static_cast<int>(fb.bytes_per_entry),
            static_cast<int>(fb.meta_bytes),
            stream);

        // (d) Cross-rank barrier so every peer has finished writing into my recv_buf.
        NCCL_CHECK_RESULT(ncclBarrier(group->comm, stream, group->ep_workspace));

        // (e) Strip per-entry meta and project payload to the dense output tensor.
        nccl_ep::fullmesh::launch_compact_to_output(
            reinterpret_cast<const void*>(fb.recv_local_va),
            recv_x->data,
            nRanks,
            max_tokens,
            hidden_bytes,
            static_cast<int>(fb.bytes_per_entry),
            static_cast<int>(fb.meta_bytes),
            stream);

        (void)config; (void)send_only;
    } else { // HT

        bool is_single_node = !is_internode_available(group);

        assert(num_local_tensors == 0 && "HT dispatch does not accept local_tensors");

        ncclNDTensor_t x = find_tensor_by_tag(inputs, num_inputs, NCCL_EP_TENSOR_TAG_TOKENS);
        ncclNDTensor_t topk_idx = find_tensor_by_tag(inputs, num_inputs, NCCL_EP_TENSOR_TAG_TOPK_IDX);
        ncclNDTensor_t topk_weights = find_tensor_by_tag(inputs, num_inputs, NCCL_EP_TENSOR_TAG_TOPK_WEIGHTS);
        ncclNDTensor_t scales = find_tensor_by_tag(inputs, num_inputs, NCCL_EP_TENSOR_TAG_SCALES);

        assert(x != nullptr);
        assert(x->ndim == 2 && tensor_is_contiguous(x));
        assert(x->sizes[0] == handle->num_tokens);
        assert(x->sizes[0] <= group->config.max_tokens_per_rank);
        assert(x->sizes[1] == group->hidden &&
               "HT dispatch token hidden size must match group configuration");

        // For multi-node: copy user buffers to pre-registered staging buffers
        // The staging buffers were allocated and GIN-registered during Group Create
        // This avoids ~60ms GIN registration overhead on the dispatch hot path
        void* token_ptr = x->data;  // Default: use user buffer directly
        if (!is_single_node && handle->hybridep.token_staging_buffer != nullptr) {
            // Copy user tokens to pre-registered staging buffer (D2D copy is ~0.1ms vs ~30ms GIN registration)
            size_t token_size = x->sizes[0] * x->sizes[1] * ncclTypeSize(x->datatype);
            CUDA_CHECK(cudaMemcpyAsync(handle->hybridep.token_staging_buffer, x->data, token_size, cudaMemcpyDeviceToDevice, stream));
            token_ptr = handle->hybridep.token_staging_buffer;
        }
        // Detect FP8 mode based on datatype
        bool use_fp8 = (x->datatype == ncclFloat8e4m3 || x->datatype == ncclFloat8e5m2);

        // For FP8: copy user scaling factors to pre-registered staging buffer
        float* scales_ptr = use_fp8 ? static_cast<float*>(scales->data) : nullptr;  // Default: use user buffer directly
        if (use_fp8 && !is_single_node && handle->hybridep.scaling_factor_staging_buffer != nullptr) {
            // Copy user scaling factors to pre-registered staging buffer (D2D copy is ~0.1ms vs ~30ms GIN registration)
            size_t scales_size = x->sizes[0] * sizeof(float);  // One scale per token
            CUDA_CHECK(cudaMemcpyAsync(handle->hybridep.scaling_factor_staging_buffer, scales->data, scales_size, cudaMemcpyDeviceToDevice, stream));
            scales_ptr = handle->hybridep.scaling_factor_staging_buffer;
        }
        if (!use_fp8) {
            assert(x->datatype == ncclBfloat16);
        } else {
            assert(scales != nullptr && "FP8 tokens require scales input");
            assert(scales->ndim == 2 && tensor_is_contiguous(scales));
            assert(scales->datatype == ncclFloat32);
        }

        // HT dispatch kernel uses TMA for token/prob/scaling-factor payloads.
        // Keep these constraints at API-entry to fail fast on unsupported shapes.
        const int experts_per_node = group->num_local_experts * group->lsa_rank_count;
        assert((experts_per_node * static_cast<int>(sizeof(float))) % 16 == 0 &&
               "HT dispatch requires experts_per_node to be multiple of 4 (16B prob TMA alignment)");

        const size_t token_bytes_per_token =
            static_cast<size_t>(group->hidden) * ncclTypeSize(x->datatype);
        assert((token_bytes_per_token % 16) == 0 &&
               "HT dispatch requires token bytes per token to be 16B aligned for TMA");

        if (use_fp8) {
            assert((group->hidden % 128) == 0 &&
                   "HT dispatch FP8 requires hidden_dim multiple of 128");
            assert((((group->hidden / 128) * static_cast<int>(sizeof(float))) % 16) == 0 &&
                   "HT dispatch FP8 requires scaling-factor bytes per token to be 16B aligned");
        }

        // Output tensors
        ncclNDTensor_t recv_topk_weights = find_tensor_by_tag(outputs, num_outputs, NCCL_EP_TENSOR_TAG_TOPK_WEIGHTS);
        ncclNDTensor_t recv_topk_idx = find_tensor_by_tag(outputs, num_outputs, NCCL_EP_TENSOR_TAG_TOPK_IDX);

        // Detect forward/backward mode
        bool forward_dispatch = (topk_idx != nullptr);

         // Validate topk inputs
        if (forward_dispatch) {
            assert(topk_weights != nullptr);
            assert(topk_idx->ndim == 2 && tensor_is_contiguous(topk_idx) && topk_idx->datatype == ncclInt64);
            assert(topk_weights->ndim == 2 && tensor_is_contiguous(topk_weights) && topk_weights->datatype == ncclFloat32);
            assert(topk_weights->sizes[0] == handle->num_tokens);
            assert(topk_weights->sizes[1] == handle->num_topk);
        } else {
            assert(topk_weights == nullptr);
            assert(recv_topk_weights == nullptr);
            assert(recv_topk_idx == nullptr);
        }

        /* ===== Convert sparse topk_weights → dense prob (sparse→dense format) ===== */
        // NCCL HT uses sparse format: topk_idx[token][k] = expert_id, topk_weights[token][k] = weight
        // HT uses dense format: prob[token][expert] = weight (0 if not routed)
        float* dense_prob = handle->hybridep.dense_prob_buffer;
        if (forward_dispatch) {
            size_t dense_prob_size = static_cast<size_t>(handle->num_tokens) * group->config.num_experts * sizeof(float);
            CUDA_CHECK(cudaMemsetAsync(dense_prob, 0, dense_prob_size, stream));

            nccl_ep::hybridep::sparse_to_dense_prob(
                static_cast<const int64_t*>(topk_idx->data),
                static_cast<const float*>(topk_weights->data),
                dense_prob,
                handle->num_tokens,
                handle->num_topk,
                group->config.num_experts,
                stream);
        }

        /* ===== Build DispatchParams ===== */
        // DispatchParams encapsulates all buffers and metadata needed by HT dispatch kernel:
        //   - Input buffers: attn_input_token, attn_input_prob, attn_input_scaling_factor
        //   - IPC output buffers: expert_output_token_ptrs, expert_output_prob_ptrs (per-rank pointers)
        //   - RDMA staging buffers: rdma_inter_node_group_* (for multi-node only)
        //   - Metadata: sparse_to_dense_map, rdma_to_attn_map, attn_to_rdma_map
        //   - Sync flags: expected_*_flag_value, intra_node_write_completion_flags
        nccl_ep::hybridep::DispatchParams params;
        params.hidden_dim = group->hidden;
        params.experts_per_rank = group->num_local_experts;
        params.num_ranks_per_node = group->lsa_rank_count;
        params.attn_input_token = token_ptr;
        params.attn_input_prob = forward_dispatch ? dense_prob : nullptr;
        params.attn_input_scaling_factor = use_fp8 ? static_cast<const float*>(scales_ptr) : nullptr;
        // Use HOST pointer arrays - these get copied into the kernel param struct for fast __grid_constant__ access
        params.expert_output_token_ptrs = group->ht_buffers.dispatch_expert_output_token_buffer_ptrs;
        params.expert_output_prob_ptrs = group->ht_buffers.dispatch_expert_output_prob_buffer_ptrs;
        params.expert_output_scaling_factor_ptrs = use_fp8 ? group->ht_buffers.dispatch_expert_output_scaling_factor_buffer_ptrs : nullptr;
        params.rdma_to_attn_map = handle->hybridep.rdma_to_attn_map;
        params.attn_to_rdma_map = handle->hybridep.attn_to_rdma_map;
        params.sparse_to_dense_map = handle->hybridep.sparse_to_dense_map;
        group->ht_buffers.host_dispatch_expected_rdma += 1;
        group->ht_buffers.host_dispatch_expected_intra += group->lsa_rank_count;
        params.expected_rdma_flag_value = is_single_node ? 0 : group->ht_buffers.host_dispatch_expected_rdma;
        params.rdma_inter_node_group_flags = is_single_node ? nullptr : group->ht_buffers.rdma_inter_node_group_flags;
        params.expected_intra_node_flag_value = group->ht_buffers.host_dispatch_expected_intra;
        params.intra_node_write_completion_flags = group->ht_buffers.intra_node_write_completion_flags;
        params.dispatch_grid_barrier_counter = group->ht_buffers.dispatch_grid_barrier_counter;
        // Pass device communicators and windows
        params.dcomms = is_single_node ? nullptr : group->gin_config.d_dcomms;
        params.nccl_window = is_single_node ? ncclWindow_t{} : group->gin_config.nccl_window;
        params.num_gin_comms = is_single_node ? 0 : group->gin_config.num_comms;
        params.num_ctx_per_comm = is_single_node ? 0 : group->gin_config.num_ctx_per_comm;
        params.gin_base_ptr = is_single_node ? nullptr : group->gin_config.gin_base_ptr;
        params.signals_base = group->gin_config.signals_base;
        // Use offsets relative to gin_base_ptr
        // All buffers are part of one large registered window
        // Calculate bytes_per_entry for batched staging
        size_t bytes_per_token_entry = group->hidden * sizeof(uint16_t);  // token data
        size_t bytes_per_prob_entry = (group->num_local_experts * group->lsa_rank_count) * sizeof(float);  // prob data
        size_t bytes_per_sf_entry = (group->hidden / 128) * sizeof(float);  // scaling factor (FP8)
        size_t bytes_per_entry = bytes_per_token_entry + bytes_per_prob_entry + bytes_per_sf_entry;

        params.mr_info = {
            .attn_input_token_offset = is_single_node ? 0 : group->gin_config.token_staging_offset,
            .attn_input_prob_offset = is_single_node ? 0 : group->gin_config.dense_prob_offset,
            .attn_input_scaling_factor_offset = is_single_node ? 0 : group->gin_config.scaling_factor_staging_offset,
            // Batched staging parameters (packed layout)
            .rdma_send_staging_offset = is_single_node ? 0 : group->gin_config.rdma_send_staging_offset,
            .rdma_inter_node_group_packed_offset = is_single_node ? 0 : group->gin_config.rdma_inter_node_group_packed_offset,
            .bytes_per_entry = bytes_per_entry,
            .max_tokens_per_dest = static_cast<size_t>(group->config.max_tokens_per_rank),
            // Streaming signal parameters
            .signals_tail_base = is_single_node ? 0 : static_cast<unsigned>(group->gin_config.signals_tail_base),
            .num_max_rdma_chunked_send_tokens = is_single_node ? 0 : group->gin_config.num_max_rdma_chunked_send_tokens,
        };
        params.local_rank = group->rank_in_node;
        params.node_rank = group->node_id;
        params.num_tokens_per_rank = handle->num_tokens;

        // Call dispatch kernel
        nccl_ep::hybridep::call_dispatch(
            params,
            group->config.max_tokens_per_rank,
            group->nNodes,
            use_fp8,
            forward_dispatch,
            stream
        );

        /* ===== Copy IPC staging → caller outputs ===== */
        // HT kernel writes to IPC-mapped buffers (dispatch_expert_output_*_buffer_ptrs)
        // Copy results to user-provided output tensors
        ncclNDTensor_t recv_x = find_tensor_by_tag(outputs, num_outputs, NCCL_EP_TENSOR_TAG_TOKENS);
        if (recv_x != nullptr) {
            assert(recv_x->ndim == 2 && tensor_is_contiguous(recv_x));
            size_t copy_size = static_cast<size_t>(recv_x->sizes[0]) * recv_x->sizes[1] * ncclTypeSize(recv_x->datatype);

            CUDA_CHECK(cudaMemcpyAsync(recv_x->data,
                group->ht_buffers.dispatch_expert_output_token_buffer_ptrs[group->rank_in_node],
                copy_size,
                cudaMemcpyDeviceToDevice,
                stream));
        }

        /* ===== Convert dense output → sparse format ===== */
        if (forward_dispatch) {
            // Convert outputs - both recv_topk_weights and recv_topk_idx must be provided together
            if (recv_topk_weights != nullptr && recv_topk_idx != nullptr) {
                // Validate formats for both outputs
            assert(recv_topk_weights->ndim == 2 && tensor_is_contiguous(recv_topk_weights));
            assert(recv_topk_weights->datatype == ncclFloat32);
            assert(recv_topk_idx->ndim == 2 && tensor_is_contiguous(recv_topk_idx));
            assert(recv_topk_idx->datatype == ncclInt64);
            assert(recv_topk_weights->sizes[0] == recv_topk_idx->sizes[0]);

            int num_recv_tokens = static_cast<int>(recv_topk_weights->sizes[0]);
            int experts_per_node = group->num_local_experts * group->lsa_rank_count;

            nccl_ep::hybridep::dense_to_sparse_prob(
                group->ht_buffers.dispatch_expert_output_prob_buffer_ptrs[group->rank_in_node],
                handle->hybridep.local_expert_routing_map,
                static_cast<float*>(recv_topk_weights->data),
                static_cast<int64_t*>(recv_topk_idx->data),
                num_recv_tokens,
                handle->num_topk,
                group->num_local_experts,
                experts_per_node,
                group->rank_in_node,
                stream);
             } else {
                 // Both outputs must be provided together or neither
                 assert(recv_topk_weights == nullptr && recv_topk_idx == nullptr);
             }
        }

        // Copy FP8 scales output
        if (use_fp8) {
            ncclNDTensor_t recv_scales = find_tensor_by_tag(outputs, num_outputs, NCCL_EP_TENSOR_TAG_SCALES);
            if (recv_scales != nullptr) {
                assert(recv_scales->ndim == 2 && tensor_is_contiguous(recv_scales));
                size_t copy_size = static_cast<size_t>(recv_scales->sizes[0]) * recv_scales->sizes[1] * ncclTypeSize(recv_scales->datatype);

                CUDA_CHECK(cudaMemcpyAsync(recv_scales->data,
                    group->ht_buffers.dispatch_expert_output_scaling_factor_buffer_ptrs[group->rank_in_node],
                    copy_size,
                    cudaMemcpyDeviceToDevice,
                    stream));
            }
        }
    }
    return ncclSuccess;
}

ncclResult_t ncclEpCombine(
    ncclEpHandle_t handle,
    const ncclNDTensor_t* inputs,
    unsigned int num_inputs,
    const ncclNDTensor_t* outputs,
    unsigned int num_outputs,
    const ncclNDTensor_t* local_tensors,
    unsigned int num_local_tensors,
    unsigned int send_only,
    const ncclEpCombineConfig_t* config,
    cudaStream_t stream
) {
    if (handle->group->config.algorithm == NCCL_EP_ALGO_LOW_LATENCY) {
        // Find and validate input tensors
        ncclNDTensor_t x = find_tensor_by_tag(
            inputs, num_inputs, NCCL_EP_TENSOR_TAG_TOKENS
        );
            assert(x != nullptr);

        ncclNDTensor_t topk_idx = handle->topk_idx;
        ncclNDTensor_t src_info = handle->ll.expert_recv_source_indices;
        ncclNDTensor_t layout_range = handle->ll.expert_dispatch_layout;

        // Find and validate local tensors
        ncclNDTensor_t topk_weights = find_tensor_by_tag(
            local_tensors, num_local_tensors, NCCL_EP_TENSOR_TAG_TOPK_WEIGHTS
        );
            assert(topk_weights != nullptr);

        // Extract configuration values
        const int num_experts = handle->group->config.num_experts;
        const int num_ranks = handle->group->nRanks;
        const int num_max_dispatch_tokens_per_rank = handle->group->config.max_tokens_per_rank;

        // Validate input tensor x
        assert(x->ndim == 3);
        assert(tensor_is_contiguous(x));
        assert(x->datatype == ncclBfloat16);
            assert(x->sizes[0] == num_experts / num_ranks);
            assert(x->sizes[1] == num_ranks * num_max_dispatch_tokens_per_rank);
        assert(x->sizes[2] % sizeof(int4) == 0);
        assert(x->sizes[2] % 128 == 0);

            // Validate topk_idx tensor
        assert(topk_idx->ndim == 2);
        assert(tensor_is_contiguous(topk_idx));
        assert(topk_idx->datatype == ncclInt64);

            // Validate src_info tensor
        assert(src_info->ndim == 2);
        assert(tensor_is_contiguous(src_info));
        assert(src_info->datatype == ncclInt32);
        assert(x->sizes[0] == src_info->sizes[0]);

            // Validate topk_weights tensor
        assert(topk_weights->ndim == 2);
        assert(tensor_is_contiguous(topk_weights));
        assert(topk_weights->sizes[0] == topk_idx->sizes[0]);
        assert(topk_weights->sizes[1] == topk_idx->sizes[1]);
        assert(topk_weights->datatype == ncclFloat32);
        assert(topk_weights->sizes[0] <= num_max_dispatch_tokens_per_rank);

        // Extract dimensions
        const int hidden = static_cast<int>(x->sizes[2]);
        const int num_topk = static_cast<int>(topk_weights->sizes[1]);
        const int num_combined_tokens = static_cast<int>(topk_weights->sizes[0]);

        // Manage double-buffering
        const auto& buffer = handle->ll.layout.buffers[handle->ll.buffer_idx];
        auto& next_buffer = handle->ll.layout.buffers[handle->ll.buffer_idx ^= 1];
        const auto next_clean_meta = next_buffer.clean_meta();

        // Validate buffer layout
        assert(handle->ll.layout.total_bytes <= handle->group->config.rdma_buffer_size);

        // Find and validate output tensor
        ncclNDTensor_t out = find_tensor_by_tag(
            outputs, num_outputs, NCCL_EP_TENSOR_TAG_TOKENS
        );

        assert(out != nullptr);
        assert(out->ndim == 2);
        assert(tensor_is_contiguous(out));
        assert(out->sizes[0] == num_combined_tokens);
        assert(out->sizes[1] == hidden);
        assert(out->datatype == x->datatype);

        // Define combine lambda
        unsigned signal_base = handle->ll.buffer_idx * (handle->group->num_dispatch_signals / 2);
        auto combine_fn = [=](int phases) {
            // Prepare data pointers
            auto* out_data = out->data;
            auto* x_data = x->data;
            auto* topk_idx_data = static_cast<int64_t*>(topk_idx->data);
            auto* topk_weights_data = static_cast<float*>(topk_weights->data);
            auto* src_info_data = static_cast<int*>(src_info->data);
            auto* layout_range_data = static_cast<int64_t*>(layout_range->data);

            const bool use_fp8 = false;
            const bool zero_copy = false;

            nccl_ep::internode_ll::combine(
                x_data,
                src_info_data,
                layout_range_data,
                topk_idx_data,
                topk_weights_data,
                out_data,
                buffer.combine_rdma_send_buffer,
                buffer.combine_rdma_recv_data_buffer,
                buffer.combine_rdma_recv_flag_buffer,
                (reinterpret_cast<uint64_t>(buffer.combine_rdma_send_buffer) - reinterpret_cast<uint64_t>(handle->group->rdma_buffer)),
                (reinterpret_cast<uint64_t>(buffer.combine_rdma_recv_data_buffer) - reinterpret_cast<uint64_t>(handle->group->rdma_buffer)),
                (reinterpret_cast<uint64_t>(buffer.combine_rdma_recv_flag_buffer) - reinterpret_cast<uint64_t>(handle->group->rdma_buffer)),
                next_clean_meta.first,
                next_clean_meta.second,
                nullptr, /*recv_topk_idx=*/
                num_combined_tokens,
                hidden,
                num_max_dispatch_tokens_per_rank,
                num_topk,
                num_experts,
                handle->group->rank,
                handle->group->nRanks,
                use_fp8,
                phases,
                zero_copy,
                handle->group->num_nccl_comms,
                handle->group->nccl_dev_comms,
                handle->group->nccl_wins,
                signal_base,
                handle->group->ep_workspace,
                handle->group->device_sm_count,
                stream
            );
        };

        // Execute combine with appropriate phase flags
        const int combine_phases = send_only ?
            LOW_LATENCY_SEND_PHASE : (LOW_LATENCY_SEND_PHASE | LOW_LATENCY_RECV_PHASE);
        combine_fn(combine_phases);

        if (send_only) {
                handle->ll.continue_fn = combine_fn;
            }
    } else if (handle->group->config.algorithm == NCCL_EP_ALGO_FULLMESH) {
        // [FULLMESH] Phase 2 commit 4: real combine path.
        //   (a) cudaMemsetAsync on combine_local_va so contributions that the
        //       current handle's num_topk leaves past are zero (the reduce
        //       kernel always sums 0..num_topk entries, so unused slots must
        //       read back as zero, not last-iter garbage).
        //   (b) ncclBarrier: every rank finishes its memset before any peer
        //       starts pushing into it.
        //   (c) launch_combine_push_kernel: dest rank reads its own meta +
        //       counter_row, then pushes ffn_output[src*max+slot] to
        //       peer_combine_vas_dev[src_rank][src_token_id][k_in_topk].
        //   (d) ncclBarrier: all pushes from every peer complete before we
        //       read our own combine_local_va.
        //   (e) launch_combine_reduce_kernel: src-local weighted sum across
        //       the k dimension using topk_weights -> combined_output.
        ncclEpGroup_t group = handle->group;
        assert(group->fullmesh_buffers.initialized && "FULLMESH buffers not init'd");

        // Inputs: expert (FFN) output tokens (required) + topk_weights (optional).
        // topk_weights is absent in ep_bench's HT tensor setup (it puts them in
        // combine_inputs[1] but sets num_combine_inputs=1, so find_tensor_by_tag
        // only sees inputs[0] = expert_outputs). Allow nullptr and fall back to
        // uniform 1/num_topk in the reduce kernel -- matches HT's forward
        // combine semantics and keeps FULLMESH testable without touching the
        // benchmark.
        ncclNDTensor_t expert_out  = find_tensor_by_tag(inputs, num_inputs, NCCL_EP_TENSOR_TAG_TOKENS);
        ncclNDTensor_t topk_w_in   = find_tensor_by_tag(inputs, num_inputs, NCCL_EP_TENSOR_TAG_TOPK_WEIGHTS);
        ncclNDTensor_t combined_x  = find_tensor_by_tag(outputs, num_outputs, NCCL_EP_TENSOR_TAG_TOKENS);
        assert(expert_out != nullptr && "FULLMESH combine missing input TOKENS");
        assert(combined_x != nullptr && "FULLMESH combine missing output TOKENS");
        assert(expert_out->ndim == 2 && tensor_is_contiguous(expert_out));
        assert(combined_x->ndim == 2 && tensor_is_contiguous(combined_x));
        assert(expert_out->datatype == ncclBfloat16 && "FULLMESH combine only supports bf16 payload");
        assert(combined_x->datatype == ncclBfloat16);
        assert(combined_x->sizes[0] == handle->num_tokens);
        if (topk_w_in != nullptr) {
            assert(topk_w_in->ndim == 2 && tensor_is_contiguous(topk_w_in));
            assert(topk_w_in->datatype == ncclFloat32);
            assert(topk_w_in->sizes[0] == handle->num_tokens);
            assert(topk_w_in->sizes[1] == handle->num_topk);
        }
        assert(num_local_tensors == 0 && "FULLMESH combine does not accept local_tensors");

        auto& fb = group->fullmesh_buffers;
        const int nRanks       = group->nRanks;
        const int myRank       = group->rank;
        const int max_tokens   = static_cast<int>(group->config.max_tokens_per_rank);
        const int hidden_bytes = static_cast<int>(group->config.token_size_bytes);
        const int max_topk     = static_cast<int>(fb.max_topk_for_combine);
        assert(handle->num_topk > 0 && handle->num_topk <= max_topk &&
               "FULLMESH combine: handle num_topk exceeds max_topk_for_combine");

        // (a) Zero only the ACTIVE num_topk columns of combine_buf so stale
        // data from past iterations doesn't leak into the weighted sum.
        //
        // Why 2D instead of flat memset of the full buffer: combine_local_va
        // is sized to max_topk_for_combine (=32) columns per token as a
        // group-level upper bound, but the current handle's num_topk is
        // normally <= 8. push_kernel only writes columns [0, num_topk),
        // reduce_kernel only reads columns [0, num_topk) (loops 0..num_topk-1).
        // Columns [num_topk, max_topk) are completely untouched by the kernels,
        // so zeroing them every call is 100% waste.
        //
        // Why memset at all (can't just delete it): dispatch_kernel early-exits
        // on topk_idx[t][k] == -1, so masked (t, k) pairs never reach the
        // push_kernel, and the corresponding combine_buf[t][k] is not written
        // by anyone this iteration. Without memset, reduce_kernel reads stale
        // residue from last iteration. generateTopkIndicesHT never emits -1
        // (HT mode is the dense case), but LL's generateRandomTopkIndicesLL
        // and real production topk_idx can, so Phase 3 keeps the memset and
        // only shrinks its width.
        //
        // Shape: combine_local_va layout is [max_tokens][max_topk][hidden_bytes].
        // Row pitch = max_topk * hidden_bytes. We want to zero the first
        // num_topk * hidden_bytes of every row, for max_tokens_per_rank rows.
        {
            const size_t row_pitch_bytes = fb.max_topk_for_combine
                                         * static_cast<size_t>(hidden_bytes);
            const size_t active_width_bytes = static_cast<size_t>(handle->num_topk)
                                            * static_cast<size_t>(hidden_bytes);
            CUDA_CHECK(cudaMemset2DAsync(
                reinterpret_cast<void*>(fb.combine_local_va),
                row_pitch_bytes,
                0,
                active_width_bytes,
                static_cast<size_t>(max_tokens),
                stream));
        }

        // (b) Pre-push barrier.
        NCCL_CHECK_RESULT(ncclBarrier(group->comm, stream, group->ep_workspace));

        // (c) Dest-initiated push kernel.
        nccl_ep::fullmesh::launch_combine_push_kernel(
            expert_out->data,                                        // ffn output
            reinterpret_cast<const void*>(fb.recv_local_va),         // for meta
            reinterpret_cast<const int32_t*>(fb.counter_local_va),   // counter_row
            fb.peer_combine_vas_dev,
            nRanks, myRank, max_tokens, max_topk,
            hidden_bytes,
            static_cast<int>(fb.bytes_per_entry),
            static_cast<int>(fb.meta_bytes),
            stream);

        // (d) Post-push barrier: peers finished writing into my combine_local_va.
        NCCL_CHECK_RESULT(ncclBarrier(group->comm, stream, group->ep_workspace));

        // (e) Src-local weighted reduce across k -> combined_output.
        nccl_ep::fullmesh::launch_combine_reduce_kernel(
            reinterpret_cast<const void*>(fb.combine_local_va),
            topk_w_in ? static_cast<const float*>(topk_w_in->data) : nullptr,
            combined_x->data,
            handle->num_tokens, handle->num_topk, max_topk,
            hidden_bytes,
            stream);

        (void)config; (void)send_only;
    } else if (handle->group->config.algorithm == NCCL_EP_ALGO_HIGH_THROUGHPUT) {
        // ===== HT mode =====
        // Combine: gather expert outputs back to original token positions
        // Forward combine: just tokens (inference)
        // Backward combine: tokens + gradients (training, topk_weights provided)
        ncclEpGroup_t group = handle->group;
        bool is_single_node = !is_internode_available(group);
             //assert(is_single_node && "HT mode only supports single-node");

        /* ===== Inputs validation ===== */
        ncclNDTensor_t x = find_tensor_by_tag(inputs, num_inputs, NCCL_EP_TENSOR_TAG_TOKENS);
        assert(x != nullptr);
        assert(x->ndim == 2 && tensor_is_contiguous(x));
        assert(x->datatype == ncclBfloat16); // HT combine only supports BF16

        // Get dimensions from input tensor
        auto num_tokens = static_cast<int>(x->sizes[0]);
        auto hidden = static_cast<int>(x->sizes[1]);

        // Validate int4 alignment for TMA
        assert((hidden * ncclTypeSize(x->datatype)) % sizeof(int4) == 0);

        // Number of tokens to combine back (original token count from this rank)
        auto num_combined_tokens = handle->num_tokens;

        // Top-k checks (for backward mode)
        int num_topk = 0;
        ncclNDTensor_t topk_weights = find_tensor_by_tag(inputs, num_inputs, NCCL_EP_TENSOR_TAG_TOPK_WEIGHTS);
        ncclNDTensor_t combined_topk_weights = find_tensor_by_tag(outputs, num_outputs, NCCL_EP_TENSOR_TAG_TOPK_WEIGHTS);

        // Determine if this is backward mode (topk_weights provided = backward combine)
        bool backward_combine = (topk_weights != nullptr);

        if (backward_combine) {
            num_topk = static_cast<int>(topk_weights->sizes[1]);

            assert(combined_topk_weights != nullptr);
            assert(combined_topk_weights->ndim == 2 && tensor_is_contiguous(combined_topk_weights));
            assert(combined_topk_weights->sizes[0] == num_combined_tokens);
            assert(combined_topk_weights->sizes[1] == num_topk);
            assert(combined_topk_weights->datatype == ncclFloat32);
        }

        // HT combine doesn't use local tensors, so hybridep should not use local tensors either
        assert(num_local_tensors == 0);

        /* ===== Output tensors ===== */
        ncclNDTensor_t combined_x = find_tensor_by_tag(outputs, num_outputs, NCCL_EP_TENSOR_TAG_TOKENS);
        assert(combined_x != nullptr);
        assert(combined_x->ndim == 2 && tensor_is_contiguous(combined_x));
        assert(combined_x->sizes[0] == num_combined_tokens); // Output should match original token count
        assert(combined_x->sizes[1] == hidden);              // Should match input hidden dimension

        /* ===== Copy input to IPC staging buffers ===== */
        // Expert MLP output needs to be in IPC buffer so other ranks can read it
        size_t token_copy_size = static_cast<size_t>(num_tokens) * hidden * sizeof(uint16_t); // BF16 = uint16_t
        CUDA_CHECK(cudaMemcpyAsync(
            group->ht_buffers.expert_input_token,
            x->data,
            token_copy_size,
            cudaMemcpyDeviceToDevice,
            stream));

        /* ===== Convert sparse topk_weights to dense prob for backward combine ===== */
        // For backward combine, convert sparse input weights to dense format for HT kernel
        if (backward_combine) {
            int experts_per_node = group->num_local_experts * group->lsa_rank_count;
            size_t dense_prob_size = static_cast<size_t>(num_tokens) * experts_per_node * sizeof(float);

            // Zero-initialize the dense prob buffer before scattering
            CUDA_CHECK(cudaMemsetAsync(
                group->ht_buffers.combine_expert_input_prob_buffer_ptrs[group->rank_in_node],
                0, dense_prob_size, stream));

            // Convert sparse [num_tokens, topk] to dense [num_tokens, experts_per_node]
            // Uses local_expert_routing_map to determine expert positions (matches dispatch output order)
            nccl_ep::hybridep::sparse_to_dense_prob_combine(
                static_cast<const float*>(topk_weights->data),
                handle->hybridep.local_expert_routing_map,
                group->ht_buffers.combine_expert_input_prob_buffer_ptrs[group->rank_in_node],
                num_tokens,
                num_topk,
                group->num_local_experts, // experts_per_rank
                experts_per_node,
                group->rank_in_node,
                stream);
        }

        // Use pre-allocated dense prob buffer for backward combine
        float* dense_output_prob = handle->hybridep.dense_prob_buffer;
        if (backward_combine) {
            size_t dense_output_prob_size = static_cast<size_t>(num_combined_tokens) * group->config.num_experts * sizeof(float);
            CUDA_CHECK(cudaMemsetAsync(dense_output_prob, 0, dense_output_prob_size, stream));
        }

        /* ===== Build CombineParams ===== */
        // CombineParams encapsulates all buffers and metadata needed by HT combine kernel:
        //   - IPC input buffers: expert_input_token_ptrs, expert_input_prob_ptrs (per-rank pointers)
        //   - Output buffers: attn_output_token, attn_output_prob (user-provided)
        //   - RDMA buffers: rdma_intra_node_red_*, combine_rdma_inter_node_group_* (for multi-node)
        //   - Metadata: sparse_to_dense_map, rdma_to_attn_map, attn_to_rdma_map, local_expert_routing_map
        //   - Sync flags: combine_expected_*_flag_value, combine_intra_node_write_completion_flags
        nccl_ep::hybridep::CombineParams params;
        params.hidden_dim = group->hidden;
        params.experts_per_rank = group->num_local_experts;
        params.num_ranks_per_node = group->lsa_rank_count;
        // Use HOST pointer arrays - these get copied into the kernel param struct for fast __grid_constant__ access
        params.expert_input_token_ptrs = group->ht_buffers.combine_expert_input_token_buffer_ptrs;
        params.expert_input_prob_ptrs = backward_combine ? group->ht_buffers.combine_expert_input_prob_buffer_ptrs : nullptr;
        params.attn_output_token = combined_x->data;
        params.attn_output_prob = backward_combine ? dense_output_prob : nullptr;
        params.rdma_intra_node_red_token = is_single_node ? nullptr : group->ht_buffers.rdma_intra_node_red_token;
        params.rdma_intra_node_red_prob = (!is_single_node && backward_combine) ? group->ht_buffers.rdma_intra_node_red_prob : nullptr;
        params.combine_rdma_inter_node_group_token = is_single_node ? nullptr : group->ht_buffers.combine_rdma_inter_node_group_token;
        params.combine_rdma_inter_node_group_prob = (!is_single_node && backward_combine) ? group->ht_buffers.combine_rdma_inter_node_group_prob : nullptr;
        params.sparse_to_dense_map = handle->hybridep.sparse_to_dense_map;
        params.rdma_to_attn_map = handle->hybridep.rdma_to_attn_map;
        params.attn_to_rdma_map = handle->hybridep.attn_to_rdma_map;
        params.local_expert_routing_map = handle->hybridep.local_expert_routing_map;
        group->ht_buffers.host_combine_expected_rdma += 1;
        group->ht_buffers.host_combine_expected_intra += group->lsa_rank_count;
        params.combine_expected_rdma_flag_value = is_single_node ? 0 : group->ht_buffers.host_combine_expected_rdma;
        params.combine_rdma_inter_node_group_flags = is_single_node ? nullptr : group->ht_buffers.combine_rdma_inter_node_group_flags;
        params.combine_expected_intra_node_flag_value = group->ht_buffers.host_combine_expected_intra;
        params.combine_intra_node_write_completion_flags = group->ht_buffers.combine_intra_node_write_completion_flags;
        // Pass device communicators and windows
        params.dcomms = is_single_node ? nullptr : group->gin_config.d_dcomms;
        params.nccl_window = is_single_node ? ncclWindow_t{} : group->gin_config.nccl_window;
        params.num_gin_comms = is_single_node ? 0 : group->gin_config.num_comms;
        params.num_ctx_per_comm = is_single_node ? 0 : group->gin_config.num_ctx_per_comm;
        params.gin_base_ptr = is_single_node ? nullptr : group->gin_config.gin_base_ptr;
        params.signals_base = group->gin_config.signals_base;
        params.combine_signal_offset = group->gin_config.combine_signal_offset;
        // Use offsets relative to gin_base_ptr
        params.mr_info = {
            .rdma_intra_node_red_token_offset = is_single_node ? 0 : group->gin_config.rdma_intra_node_red_token_offset,
            .combine_rdma_inter_node_group_token_offset = is_single_node ? 0 : group->gin_config.combine_rdma_inter_node_group_token_offset,
            .rdma_intra_node_red_prob_offset = is_single_node ? 0 : group->gin_config.rdma_intra_node_red_prob_offset,
            .combine_rdma_inter_node_group_prob_offset = is_single_node ? 0 : group->gin_config.combine_rdma_inter_node_group_prob_offset,
        };
        params.local_rank = group->rank_in_node;
        params.node_rank = group->node_id;
        params.num_tokens_per_rank = num_combined_tokens;
        params.num_recv_tokens = num_tokens;

        /* ===== Call combine kernel ===== */
        nccl_ep::hybridep::call_combine(
            params,
            group->config.max_tokens_per_rank, // max_tokens_per_rank
            group->nNodes, // num_nodes (RDMA domain size)
            backward_combine, // backward mode flag
            stream
        );

        /* ===== Convert dense output prob to sparse format ===== */
        // For backward combine, convert kernel's dense output to sparse format
        // HT outputs dense [num_tokens, num_experts], NCCL expects sparse [num_tokens, topk]
        if (backward_combine && combined_topk_weights != nullptr) {
            nccl_ep::hybridep::dense_to_sparse_prob_combine(
                dense_output_prob,
                handle->hybridep.global_routing_map + (num_combined_tokens * ((group->config.num_experts + 7) / 8))*group->rank,
                static_cast<float*>(combined_topk_weights->data),
                nullptr,  // No need to output topk_idx for combine
                num_combined_tokens,
                num_topk,
                group->config.num_experts,
                stream);
        }

        handle->cached_mode = true;
    }

    return ncclSuccess;
}

ncclResult_t ncclEpComplete(
        ncclEpHandle_t handle,
    const ncclEpCompleteConfig_t* config,
    cudaStream_t stream
) {
    if (handle->group->config.algorithm == NCCL_EP_ALGO_LOW_LATENCY) {
        if (handle->ll.continue_fn) {
                handle->ll.continue_fn(LOW_LATENCY_RECV_PHASE);
                handle->ll.continue_fn = nullptr;
        }
    } else if (handle->group->config.algorithm == NCCL_EP_ALGO_HIGH_THROUGHPUT) {
        // HT mode - no continue needed (synchronous)
    } else if (handle->group->config.algorithm == NCCL_EP_ALGO_FULLMESH) {
        // FULLMESH Phase 2 commit 1 stub: dispatch is synchronous (single kernel),
        // no deferred continue_fn. ncclBarrier (Q1=C) will be added in commit 4
        // and called directly from ncclEpDispatch/ncclEpCombine for cross-rank
        // completion sync, not here.
    }
    (void)config; (void)stream;
    return ncclSuccess;
}

ncclResult_t ncclEpHandleGetNumRecvTokens(
        ncclEpHandle_t handle,
    unsigned int* num_recv_tokens
) {
    if (handle->group->config.algorithm == NCCL_EP_ALGO_HIGH_THROUGHPUT) {
        int32_t actual_recv_tokens =
            (handle->hybridep.num_tokens_for_experts_host != nullptr) ?
            *handle->hybridep.num_tokens_for_experts_host : -1;

        // Fallback for callers that query before stream synchronization.
        if (actual_recv_tokens < 0) {
            CUDA_CHECK(cudaMemcpy(
                &actual_recv_tokens,
                handle->hybridep.num_tokens_for_experts,
                sizeof(actual_recv_tokens),
                cudaMemcpyDeviceToHost));
            if (handle->hybridep.num_tokens_for_experts_host != nullptr) {
                *handle->hybridep.num_tokens_for_experts_host = actual_recv_tokens;
            }
        }

        assert(actual_recv_tokens >= 0);
        assert(actual_recv_tokens <= handle->group->max_recv_tokens);
        *num_recv_tokens = static_cast<unsigned int>(actual_recv_tokens);
    } else if (handle->group->config.algorithm == NCCL_EP_ALGO_FULLMESH) {
        // FULLMESH Phase 2 commit 1: dynamic num_recv_tokens is not supported.
        // Callers compute the static upper bound nRanks * max_tokens_per_rank
        // directly (same as HT non-dynamic path in ep_bench). Return invalid
        // usage so accidental dynamic-mode wiring fails fast during phase 2
        // rollout instead of producing a silently-zero counter.
        return ncclInvalidUsage;
    } else { // LL
        return ncclInvalidUsage;
    }
    return ncclSuccess;
}
