# MORI EP vs NCCL EP: 方案与性能数据对比

> 范围: 纯代码 + 公开文档分析, 本次不跑环境验证. 所有结论标注证据出处. AMD MI300/MI355 公开数值不直接外推到 NV GB300.
>
> 完成时间: 2026-05-06
> 对比对象: ROCm/mori main 分支 vs 本仓库 contrib/nccl_ep 当前 working tree (含未提交的 HT NV72 调参改动)

## 0. TL;DR

- **MORI EP 的工程优势不是某一个 kernel 技巧, 而是 "5 种 kernel 类型 + per-(EP, dtype, tokens, hidden) 的 JSON tuning table + 低 token 专用 LL/AsyncLL 路径" 这套自动选择机制**.
- **NCCL EP 当前是 "HT/HybridEP 与 FULLMESH 两个手选后端 + 少量 env 旋钮"**, 已经在朝 MORI 方向走 (你未提交的 `NCCL_EP_HT_NV72_NUM_SMS / _CHUNK` 是第一步), 但还没有 tuning table 与低 token 专用路径.
- 公开 BW 数字不能跨硬件对比. **本仓库内最有价值的对比是 NCCL EP 自己的 HT vs FULLMESH**, 已在 `ep_summary.py` 中支持.

---

## 1. 架构对比 (一句话)

| 维度 | MORI EP | NCCL EP (本仓库) |
|---|---|---|
| 后端数 | 5 类 kernel (`IntraNode`/`InterNode`/`InterNodeV1`/`InterNodeV1LL`/`AsyncLL`) | 2 个后端 (HT/HybridEP, FULLMESH) + LL stub |
| Kernel 选择 | 框架按 (EP, tokens, dtype) 自动选 | 用户通过 `--algorithm` 指定 |
| Launch params | JSON keep-best 表, 运行时查表 | env 手选; HT NV72 路径新加 3 档 `NUM_SMS/CHUNK` |
| Low-token path | `InterNodeV1LL` / `AsyncLL` 专攻 tokens/rank < 256 | 无, HT/FM 同一路径处理所有 tokens |
| Token dedup | intra+inter 都做 | HT 内部有, FULLMESH 不做 (atomic slot) |
| Multi-QP | `numQpPerPe` 可调, PR #108 双 QP 已合并 | 不适用 (FULLMESH 是 NVLink fabric peer store) |
| Per-stage profile | MORI-VIZ + Perfetto | `NCCL_EP_FULLMESH_PROFILE` 5 段 cudaEvent + CUPTI ktimer |

---

## 2. MORI EP 实现要点 (证据: ROCm/mori main)

### 2.1 5 种 kernel 类型

来源: [`include/mori/ops/dispatch_combine/dispatch_combine.hpp:48`](https://github.com/ROCm/mori/blob/main/include/mori/ops/dispatch_combine/dispatch_combine.hpp#L48)

```cpp
enum KernelType { IntraNode = 0, InterNode = 1, InterNodeV1 = 2,
                  InterNodeV1LL = 3, AsyncLL = 4 };
```

| Kernel | 拓扑 | 用途 | 选择依据 (来自 MORI guide §1) |
|---|---|---|---|
| `IntraNode` | 单节点 XGMI/P2P | EP8 | EP fits in node |
| `InterNode` | 多节点 XGMI+RDMA | baseline / debug | 兼容性 |
| `InterNodeV1` | 多节点 XGMI+RDMA | EP16/EP32 大 batch | tokens/rank > 256 |
| `InterNodeV1LL` | 多节点 XGMI+RDMA | tokens/rank < 256 | SGLang PR #18437 实测 dispatch 1.52x / combine 1.82x |
| `AsyncLL` | 多节点 XGMI+RDMA | split send/recv | 最低延迟, 可与计算 overlap |

### 2.2 InterNodeV1: 一条 dispatch 用 2 个 kernel, 一条 combine 用 4 个

来源: [`python/mori/ops/dispatch_combine.py:511-546, 733-764`](https://github.com/ROCm/mori/blob/main/python/mori/ops/dispatch_combine.py)

dispatch:

```text
EpDispatchCopyToStaging  (mp 个 block, 每 block 8*64 thread)
EpDispatchInterNodeV1Kernel  (block_num 个 block; blockId<rdmaBlockNum 跑 RDMA send/recv,
                              其余跑 intra-node routing)
```

combine:

```text
EpCombineSync  (拷输入到 shmem registered buffer)
EpCombineSyncBarrier  (cross-device barrier, 仅 1 block warpSize 线程)
EpCombineInterNodeV1Kernel  (RDMA combine + IntraNode combine)
EpCombineAll  (mp 个 block, 跨 node 累加)
```

关键: `EpDispatchInterNodeV1Kernel_body` 用 `if (blockId < args.rdmaBlockNum)` 直接在同一 grid 内分 RDMA block 与 XGMI block, 见 [`src/ops/dispatch_combine/internode_v1.cpp:594-604`](https://github.com/ROCm/mori/blob/main/src/ops/dispatch_combine/internode_v1.cpp).

### 2.3 AsyncLL: split-phase 流水

来源: [`src/ops/dispatch_combine/low_latency_async.cpp:115-285`](https://github.com/ROCm/mori/blob/main/src/ops/dispatch_combine/low_latency_async.cpp)

dispatch 拆 3 个 kernel:

1. `EpDispatchLowLatencyAsyncSendCopySlotAssign` - warp shuffle 做 dedup, `numExpertPerToken` 个线程一组协作
2. `EpDispatchLowLatencyAsyncSendCopyMultiBlock` - 多 warp 协作拷一个 token 的 hidden, sub-warp 拷 metadata
3. `EpDispatchLowLatencyAsyncSendTransfer` - 每 PE x 每 QP 一次 IBGDA `ShmemPutMemNbiThread`

recv 端 `EpDispatchLowLatencyAsyncRecvCopyMultiBlock` 用 **warp prefix-sum 算 per-PE offset**, 避免全局 `atomicAdd(totalRecvTokenNum)` 瓶颈 (注释里明说 "Prefix-sum variant: eliminates atomicAdd").

### 2.4 调参是 first-class

来源: [`python/mori/ops/tuning_config.py`](https://github.com/ROCm/mori/blob/main/python/mori/ops/tuning_config.py) + `python/mori/ops/tuning_configs/*.json`

文件命名: `{arch}_{model}_{kernel}_ep{n}_{dispatch|combine}.json`

例如 `gfx950_mi355x_InterNodeV1_ep16_dispatch.json` 含 (fp4/fp8) x (64..262144 tokens) x (3584/7168 hidden) 的最优 `(block_num, rdma_block_num, warp_per_block)`. 查表 key:

- dispatch: `(dtype, num_tokens, hidden_dim)` -> `(block_num, rdma_block_num, warp_per_block)`
- combine: `(dtype, num_tokens, hidden_dim, zero_copy, quant_type)` -> 同上

查不到才用代码 fallback (来自 `dispatch_combine.py:303-326`):

| Kernel | block_num | rdma_block_num | warp_per_block |
|---|---:|---:|---:|
| `InterNodeV1` | 96 | 64 | 8 |
| `InterNodeV1LL` | 256 | 128 | 8 |
| 其他 | 128 | 0 | 16 |

仓库已有 gfx942 mi308x 与 gfx950 mi355x 多 EP/dtype 的 JSON, 可看出最优 `block_num` 随 token 数增长 (从 32 到 256), `rdma_block_num` 通常是 `block_num/2`.

### 2.5 显式 dedup

来源: [`src/ops/dispatch_combine/internode_v1.cpp:104-127, 153-208`](https://github.com/ROCm/mori/blob/main/src/ops/dispatch_combine/internode_v1.cpp)

- intra: 同一 token 的 top-k 命中同一 PE -> 仅发送 1 次, `dispDestTokIdMap[i] = NullFlatTokenIndex` 标记重复
- inter: 同一 token 命中同一 dest node -> 仅 RDMA 一次
- combine 端用 `dispDestTokIdMap` / `interNodeDispSendMap` 反推回原 token

InterNodeV1 还分 `DEDUP=true/false` 两套 kernel template (PR #105 加的, 用 warp ballot mask 做 dedup compaction).

### 2.6 Multi-QP

来源: [PR #108 "Feat: add multi-qp support for EPV1 kernel"](https://github.com/ROCm/mori/pull/108)

PR 描述: "Dual-port NIC need more than 1 qp to saturate bandwidth. CX7 single-port NIC 有微小回归, warps 限 8, blocks 翻倍补偿".

代码上每个 token 按 `tokenId / warpSize % numQpPerPe` 分到一个 QP, 见 `internode_v1.cpp:200, 236, 307`.

---

## 3. NCCL EP 当前方案 (证据: 本仓库)

### 3.1 HT / HybridEP

固定常数 (`device/hybridep_configs.cuh:17-48`):

```cpp
HYBRIDEP_MAX_NUM_SMS_PER_RANK   16
HYBRIDEP_DISPATCH_NUM_OF_STAGES         12
HYBRIDEP_DISPATCH_NUM_OF_IN_FLIGHT_S2G  4
HYBRIDEP_DISPATCH_NUM_OF_BLOCKS         16   // == MAX_NUM_SMS
HYBRIDEP_DISPATCH_NUM_OF_PIPELINES_PER_BLOCK 2
HYBRIDEP_DISPATCH_N2N_WARPS             2
HYBRIDEP_DISPATCH_RDMA_BATCH_SIZE       4
HYBRIDEP_COMBINE_SINGLENODE_NUM_OF_STAGES_G2S 12
HYBRIDEP_COMBINE_SINGLENODE_NUM_OF_STAGES_S2G  2
HYBRIDEP_COMBINE_MULTINODE_NUM_OF_STAGES_G2S   4
HYBRIDEP_COMBINE_MULTINODE_NUM_OF_STAGES_S2G   2
HYBRIDEP_COMBINE_NUM_OF_TOKENS_PER_GROUP 4
HYBRIDEP_COMBINE_NUM_OF_BLOCKS          16
HYBRIDEP_COMBINE_NUM_OF_ADDITIONAL_IN_FLIGHT_S2G 2
HYBRIDEP_COMBINE_RDMA_STREAMING_BATCH   8
```

未提交改动:

- `device/hybridep_adapter.cuh +166-191`: 新增 `HYBRIDEP_SWITCH_NUM_BLOCKS({16,32,64})` 与 `HYBRIDEP_SWITCH_CHUNK_TOKENS({64,128,256})`
- `nccl_ep.cc +1885-1912`: 仅在 `hybridep_mode && use_fabric_memory` (NV72 MNNVL full-coverage) 路径下读 `NCCL_EP_HT_NV72_NUM_SMS / NCCL_EP_HT_NV72_CHUNK` 选择 template 实例; RDMA / 普通 intra-node 行为不变
- `device/hybridep_adapter.cu +473-507, +624-655`: `dispatch_impl/combine_impl` 仅当 `NUM_LSA_TEAMS==1` 时走 tuned switch, 其他保持 production 常数

**评估**: 这是朝 MORI tuning 方向的第一步 (从 1 档变成 3x3 = 9 档); 但还没到 MORI 的 (dtype, tokens, hidden) keyed JSON.

### 3.2 FULLMESH dispatch

来源: [`nccl_ep.cc:2912-3115`](../nccl_ep.cc), [`device/fullmesh.cu:189-341`](../device/fullmesh.cu)

5 段 stream op, `NCCL_EP_FULLMESH_PROFILE=1` 可拆段:

```text
(a) cudaMemsetAsync(counter_row)
(b) ncclBarrier  (含 cudaDeviceSynchronize, 不支持 graph capture)
(c) launch_dispatch_kernel  (kCoop 或 kTma, 由 NCCL_EP_FULLMESH_DISPATCH_PATH 选)
(d) ncclBarrier
(e) launch_compact_to_output  (cudaMemcpy2DAsync 剥 8B meta, 写到用户 dense recv_x)
```

dispatch kernel 两版:

- `kCoop` (legacy): 每 (token,k) 一个 warp, lane-0 atomicAdd slot, 协作 uint4 store
- `kTma` (默认): 1 producer warp + top_k consumer warp, 2-stage ring buffer + mbarrier producer/consumer; producer 协作 gmem->smem load (fence 防 smem 重叠), consumer lane-0 issue `tma_store_1d` 到 peer fabric memory

### 3.3 FULLMESH combine

来源: [`nccl_ep.cc:3501-3757`](../nccl_ep.cc), [`device/fullmesh.cu:475-795`](../device/fullmesh.cu)

两条路径, env `NCCL_EP_FULLMESH_COMBINE_PATH={fused, push_reduce}` 选择, 默认 `push_reduce`:

```text
push_reduce (默认):
  (a) cudaMemset2DAsync(combine_buf 的 num_topk 列)
  (b) ncclBarrier
  (c) launch_combine_push_kernel   (dest 把 ffn_output 推到 src.combine_buf[token,k])
  (d) ncclBarrier
  (e) launch_combine_reduce_kernel (src 本地按 weight 累加 -> combined_output)

fused (实测慢, 仅保留 A/B):
  (a) cudaMemset2DAsync(combine_buf 仅 col 0)
  (b) ncclBarrier
  (c) launch_combine_fused_kernel (dest 把 weight*ffn 直接 atomicAdd(bf162) 到 src.combine_buf[token,0])
  (d) ncclBarrier
  (e) cudaMemcpy2DAsync (col 0 写到 combined_output)
```

注释明确 (`nccl_ep.cc:3597-3608`): 在 GB300 上 EP4/8/16/32 t=8192 fused 都比 push_reduce 慢. EP8 fused `combine_kernel_us=2284us` vs push_reduce `1795us`, 推断原因是 num_topk 个 src rank 同时向同一 col-0 region atomicAdd(bf162) 在 fabric memory 上序列化 HBM 控制器写.

---

## 4. 方案差异 (行级对照)

| 维度 | MORI EP | NCCL EP |
|---|---|---|
| Kernel 选择 | enum + 自动选 | 手指定 algorithm |
| Launch params | JSON 表, key=(dtype,tokens,hidden[,zc,qt]) | env, 9 档 (HT NV72) / 3 档 (FM) |
| Low-token path | V1LL/AsyncLL 专门优化 | 无 |
| Dedup | intra + inter 都做, dispatch kernel 内 warp ballot | HT 内部有, FULLMESH 不做 |
| RDMA blocks | `rdmaBlockNum` 显式分; 一个 grid 内 if/else 分工 | HT N2N_WARPS=2, FM 无 RDMA |
| Multi-QP | `numQpPerPe`, dual-port NIC 加 BW | 不适用 |
| Combine 数据流 | warp accum + IBGDA put 多段 kernel | FM: memset2D + barrier + atomicAdd/push + reduce |
| Cross-device barrier | shmem `crossDeviceBarrier` 数组, 每 rank 累加 | HT 用 named barrier; FM 用 ncclBarrier (cudaDeviceSynchronize) |
| Profiling | MORI-VIZ Perfetto + per-kernel slot 表 | FM: 5 段 cudaEvent (env 控制) + CUPTI ktimer (默认开) |
| Token compaction | recv 端 prefix-sum 算 offset | FM: cudaMemcpy2DAsync 剥 meta |

---

## 5. 公开性能数据

### 5.1 MORI README headline (DeepSeek V3: 4096 tokens, hidden 7168, top-8, FP8 dispatch + BF16 combine)

| 平台 | Kernel | Dispatch XGMI | Dispatch RDMA | Combine XGMI | Combine RDMA |
|---|---|---:|---:|---:|---:|
| MI300X+CX7 | EP8 IntraNode | 307 | - | 330 | - |
| MI300X+CX7 | EP16-V1 | 171 | 52 | 219 | 67 |
| MI300X+CX7 | EP32-V1 (stale) | 103 | 57 | 91 | 50 |
| MI355X+AINIC | EP8 | 345 | - | 420 | - |
| MI355X+AINIC | EP16-V1 | 179 | 54 | 234 | 71 |
| MI355X+AINIC | EP32-V1 (stale) | 85 | 46 | 110 | 61 |

128 tokens latency (DeepSeek V3, FP8 dispatch + BF16 combine):

| 平台 | Kernel | Dispatch us | Combine us |
|---|---|---:|---:|
| MI300X+CX7 | EP8 | 35 | 47 |
| MI300X+CX7 | EP16-V1-LL | 76 | 122 |
| MI355X+AINIC | EP8 | 31 | 36 |
| MI355X+AINIC | EP16-V1-LL | 84 | 108 |

注意 1: MORI guide §9 给出过同组别 EP16-V1=208/63 与 161/49, 不同 commit/node 拓扑下波动很大.
注意 2: MORI README 自标 EP32 数据 stale.
注意 3: AMD MI 系 vs NV GB300 NIC/fabric/cu 数完全不同, **不能跨平台外推**.

### 5.2 NCCL EP 仓库内可见数

来源 1: [`README.md:91-96`](../README.md), 128 tokens

| GPU | Nodes | Dispatch GB/s | Combine GB/s |
|---:|---:|---:|---:|
| 8 | 1 | 224.3 | 185.2 |
| 16 | 2 | 76.7 | 73.0 |
| 32 | 4 | 53.6 | 50.0 |
| 64 | 8 | 48.8 | 43.8 |

来源 2: FULLMESH 注释 (EP8 t=8192, GB300, sm_103)

| 路径 | dispatch_kernel_us | dispatch BW |
|---|---:|---:|
| HT (实测) | - | ~599 GB/s |
| FM C3 (实测) | 1881 | ~499 GB/s (-17% vs HT) |
| FM D1 (设计预期) | ~1170 | ~803 GB/s (+34% vs HT) |

来源: [`device/fullmesh.cu:142-188`](../device/fullmesh.cu) - **D1 是设计目标, 不是本轮实测**.

来源 3: FULLMESH combine 注释 (EP8 t=8192)

| 路径 | combine_kernel_us | 备注 |
|---|---:|---|
| push_reduce | 1795 | 当前默认 |
| fused | 2284 | bf162 atomic 在 fabric 上序列化 |

来源: [`nccl_ep.cc:3597-3608`](../nccl_ep.cc).

### 5.3 对比口径必须统一

要做公平对比必须固定:

- tokens, hidden, top-k
- dtype (MORI 默认 FP8 dispatch + BF16 combine; NCCL 默认 BF16+BF16, FP8 走 `ht_fp8`)
- kernel-only 还是 total time (MORI 报 LL bandwidth 含 send+recv 双向)
- 是否包含 copy/barrier (MORI 用 IBGDA put, NCCL FM 包含 cudaMemcpy2D)
- BW 分项 (XGMI/RDMA vs total/nvl/rdma)

**唯一公平比法**: 在同一台机器上, 固定上述维度, 看 NCCL EP 内部 HT vs FULLMESH; MORI 数据仅作 "工程组织能达到这个量级" 的参考, 不作绝对胜负结论.

---

## 6. NCCL EP 改造优先级 (按 ROI, 不在本次范围内执行)

### P1: HT NV72 调参 -> MORI 式 tuning table

现状: env (`NCCL_EP_HT_NV72_NUM_SMS / _CHUNK`) 仅 9 档手选.

做法:

1. 复用 `contrib/nccl_ep/sweep/ep_sweep.sh` 跑全网格 (EP x tokens x dtype x hidden x num_sms x chunk)
2. 写入 `contrib/nccl_ep/tuning_configs/{arch}_{model}_HT_ep{n}.json`
3. 启动时按 (dtype, tokens, hidden) 查表; 查不到 fallback 到当前 env / production 常数

触动文件: `nccl_ep.cc` (init 时加载 JSON), `hybridep_adapter.cu` (查表后选 template); 不需要改 kernel 本身.

### P2: NCCL EP low-token path (对标 InterNodeV1LL/AsyncLL)

现状: 无, HT/FM 同一路径处理所有 tokens; FM 在 t=128 等小 token 上的 barrier/copy 占比偏高.

做法 (思路, 不直接照搬 RDMA API):

- 借鉴 AsyncLL 的 split slot-assign + warp prefix-sum offset 减少 `atomicAdd(totalRecvTokenNum)` 全局原子
- 借鉴 V1LL 在 tokens/rank < 256 的 single-shot transfer 思路 (NCCL EP 在 GB300 NVL72 fabric memory 上做对应实现)
- 不动 HT/FULLMESH 现有路径; 单独加 LL 后端

触动文件: 新增 `contrib/nccl_ep/device/lowlatency_fm.cu`; `nccl_ep.cc` 加路径选择.

### P3: FULLMESH stream-op 砍刀

现状: `NCCL_EP_FULLMESH_PROFILE=1` 已能拆段, 但还没基于实际占比下手.

做法 (按 `benchmark-breakdown-first.mdc` 规则):

- 先量 EP32 t=8192 下 `cudaMemcpy2DAsync` 与 `ncclBarrier` 的实际 us 占比
- 高占比的先动 (如把 `memset+barrier` 合并, 或换 stream-only barrier `ncclBarrierSession`)
- 不要先动 kernel

触动文件: `nccl_ep.cc` FULLMESH 段; 可能加 `NCCL_EP_FULLMESH_BARRIER_MODE` env.

### P4: 统一性能口径

现状: `ep_summary.py` 已能输出 HT vs FM dispatch/combine BW.

做法:

- 每行附 (tokens, hidden, topk, dtype, kernel-only?, include-copy?)
- 新增 RDMA/XGMI/NVL 分项列, 与 MORI 公开数据对齐字段
- 加 latency-only 视图 (avg/min/max + p99 if available)

触动文件: `sweep/ep_summary.py`, `sweep/ep_parse.py`.

---

## 7. 本次明确不做

- 不跑任何 build/sweep/profile (无环境)
- 不修改 C++/CUDA 源码
- 不基于 "AMD MORI 报了 X GB/s" 推 NCCL EP 应该到 X GB/s
- 不改 NCCL EP 默认 algorithm (HT/FULLMESH 选择维持现状)
- 不动 FULLMESH combine 默认路径 (push_reduce, 实测最优)

---

## 8. 证据索引

MORI 上游:

- 仓库主页与 README BW 表: <https://github.com/ROCm/mori>
- MORI-EP guide: <https://github.com/ROCm/mori/blob/main/docs/MORI-EP-GUIDE.md>
- enum / config: <https://github.com/ROCm/mori/blob/main/include/mori/ops/dispatch_combine/dispatch_combine.hpp>
- IntraNode kernel: <https://github.com/ROCm/mori/blob/main/src/ops/dispatch_combine/intranode.hpp>
- InterNodeV1 kernel: <https://github.com/ROCm/mori/blob/main/src/ops/dispatch_combine/internode_v1.cpp>
- AsyncLL kernel: <https://github.com/ROCm/mori/blob/main/src/ops/dispatch_combine/low_latency_async.cpp>
- Tuning manager: <https://github.com/ROCm/mori/blob/main/python/mori/ops/tuning_config.py>
- Multi-QP PR: <https://github.com/ROCm/mori/pull/108>
- LL kernel PR: <https://github.com/ROCm/mori/pull/105>
- SGLang inter kernel switch PR: <https://github.com/sgl-project/sglang/pull/18437>

本仓库 (相对路径):

- [`device/hybridep_configs.cuh`](../device/hybridep_configs.cuh) - HT 常数
- [`device/hybridep_adapter.cuh`](../device/hybridep_adapter.cuh) - HT NV72 SWITCH 宏 (未提交)
- [`device/hybridep_adapter.cu`](../device/hybridep_adapter.cu) - dispatch/combine_impl 选 template (未提交)
- [`device/fullmesh.cuh`](../device/fullmesh.cuh) - FULLMESH 接口与设计注释
- [`device/fullmesh.cu`](../device/fullmesh.cu) - FULLMESH dispatch/combine 内核
- [`nccl_ep.cc`](../nccl_ep.cc) - HT NV72 env 解析 + FULLMESH dispatch/combine host (含未提交)
- [`ep_bench.cu`](../ep_bench.cu) - CUPTI ktimer + cudaEvent loop
- [`sweep/ep_sweep.sh`](../sweep/ep_sweep.sh) - sweep 驱动
- [`sweep/ep_summary.py`](../sweep/ep_summary.py) - HT vs FM BW 对比
- [`README.md`](../README.md) - 老 BW 表 (128 tokens)
