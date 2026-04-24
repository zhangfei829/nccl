# Phase 3 FULLMESH latency 极致化

**当前位置**：Commit A 已落地（3c5aee1），Commit P3-diag 在路上，Commit B
架构上受阻（见下），Commit C/D 等 P3-diag 的精确 breakdown 数据再决定顺序。

## Commit A 实测结果（4/24 sweep @ 20260424_055024）

对比 phase2-done baseline (20260423_171353) 的 FULLMESH Combine total_us:

| (EP, t) | baseline | Commit A | 实测节省 | 原预测区间 |
|---|---|---|---|---|
| EP4 t=4096  | 937   | 747   | 190   | — |
| EP4 t=8192  | 1715  | 1332  | 383   | — |
| EP8 t=4096  | 1275  | 1090  | 185   | ~940 |
| EP8 t=8192  | 2387  | 2043  | 344   | ~1880 |
| EP16 t=4096 | 1377  | 1209  | 168   | ~940 |
| EP16 t=8192 | 2567  | 2175  | 392   | ~1880 |
| EP32 t=4096 | 1518  | 1425  | 93    | ~940 |
| **EP32 t=8192** | **2737**  | **2438**  | **299**   | **500-2000** ❌ 低于下限 |

实测 299 us < 预测下限 500 us，按 benchmark-breakdown-first.mdc 规则必须
**回头重新分析**。反推：如果 Commit A 砍 75% memset 字节节省 299 us，原
memset **绝对耗时 ≈ 399 us**，不是预测的 1880 us。原估 HBM3e memset rate
1500 GB/s 高估过低；实际大段 memset rate 在 3-5 TB/s 区间。combine non-
kernel 的 740-1740 us 里，memset 只占 400 us，另外 340-1340 us 是 2x
barrier + stream op 串行化 + 其他未拆出的开销。

**Dispatch total_us 基本无变化**（Commit A 不碰 dispatch）：EP32 t=8192
baseline 3809 → Commit A 3815，+6 us 噪声内，符合预期。

**D+C total_dc_avg_us**:
  EP32 t=8192: 6556 → 6265 (-291)
  EP16 t=8192: 5290 → 4904 (-386)
  EP8  t=8192: 4423 → 4084 (-339)

## Commit B 架构约束（新发现，阻止原计划直接实施）

原 NEXT_TASKS.md 写的是"dispatch kernel 同时写两份，meta 到 peer recv_buf
payload 直接到 local output tensor"。读 ncclEpTensorCreate (nccl_ep.cc:
2011-2056) 后发现：output tensor 通过 `ep_group->alloc_fn` (默认 cudaMalloc)
分配，**不是 fabric-mapped**。远程 src rank 的 dispatch kernel 没有该
output 的 peer mapping，要 push 必须经 fabric。

解决需要其中之一：
  (B-alt-1) 扩展 ncclEpTensorCreate API 增加 "peer_accessible" flag，
            让 output 通过 cuMemCreate(FABRIC) 分配并 export + import 到
            所有 peer。侵入 public API + tensor lifecycle 管理。
  (B-alt-2) 把 recv_buf 的 payload 区作为 output tensor alias，在
            ncclEpDispatch 内部 swap tensor->data 指针指向 fb.recv_local_va
            的 payload offset。破坏 owns_data 契约 + destroy 时不能 cudaFree
            那段内存（要特殊处理）。
  (B-alt-3) 分离 meta 和 payload 到两块独立 fabric 分配（meta_buf 4MB,
            payload_buf 3.76 GB），payload_buf dense 后 cudaMemcpy2DAsync
            → cudaMemcpyAsync（flat），估节省 350-700 us @ EP32 t=8192
            （rate 从 ~1688 GB/s 升到 ~2000+ GB/s）。预期收益远低于原
            "消除 memcpy2D 2000 us"。

**三个 alt 的 ROI 都不突出**。决定：Commit B 先搁置，等 Phase 3 确实卡
在 memcpy2D 上且有时间做 API 扩展时再动。

## 当前在做：Commit P3-diag — FULLMESH 内部 per-stage profile event

**目的**：不再基于推算评估 memset / barrier / memcpy / kernel 的 absolute
时间，用 cudaEvent 在 FULLMESH dispatch + combine 的每个 stream op 之间
打点，拿精确 per-stage 耗时。Commit A 教训：推算和实测差 5-6×，继续猜
下去做的优化会全部打偏。

**实现**：在 FULLMESH 分支里（不影响 HT/LL path）插 event，env var
`NCCL_EP_FULLMESH_PROFILE=1` 触发每次调用打印:

```
[FM-PROFILE] dispatch EP=32 t=8192 iter=N memset=X.X barrier1=Y.Y kernel=Z.Z barrier2=W.W memcpy2d=V.V us
[FM-PROFILE] combine  EP=32 t=8192 iter=N memset=... barrier1=... push=... barrier2=... reduce=... us
```

每次调用做 cudaStreamSynchronize 会让 total_us **变慢**（预期 +100-300us
per call 的 sync overhead），所以 profile 模式的 total_us 不能直接和 base
line 对比。但 per-stage 数字是精确的，拿到数据就关 profile 再跑正常 sweep。

**改动规模**：~80 行（dispatch 分支 +40，combine 分支 +40）；全部在
`contrib/nccl_ep/nccl_ep.cc` 的 FULLMESH 分支内，不动 device/fullmesh.*
也不影响 HT/LL。

## 基于 P3-diag 数据的后续决策树

拿到 breakdown 后，按实测优先级排 Commit C/D/E:

1. **如果 memcpy2d > 1000 us @ EP32 t=8192** → 优先做 Commit B-alt-3
   （分离 meta/payload，flat memcpy）。预期节省 350-700 us。
2. **如果 barrier1+barrier2 > 300 us @ 小 workload** → Commit C (device
    barrier session)，前提是 GIN 能在 MNNVL 下 bootstrap（见下）。
3. **如果 dispatch kernel 本身 > fabric peak 80%** → Commit D (TMA bulk
    async) 收益很小，不做。
4. **如果 combine push kernel 单独很慢** → 独立分析。

## Commit C (device barrier) 的已知依赖

需要 `ncclBarrierSession<ncclCoopCta>`（docs/examples/06_device_api/03_
alltoall_hybrid/main.cu:82）。API 签名带 `ncclGin const&` (bindings/ir/
nccl_device_wrapper__impl.h:96)，要 GIN 初始化过。FULLMESH 当前 MNNVL
full-coverage 下 nLsaTeams=1，现有代码 (nccl_ep.cc:1780) 不 init GIN
devcomm。Commit C 前要确认 MNNVL 下 GIN 能 bootstrap。

## Commit D (TMA) 先跳过

Hopper/Blackwell `cp.async.bulk` 多数 API 为 global-to-shared 或
shared-to-global。FULLMESH dispatch kernel 是 global-to-global peer push
（stride row）。直接用 bulk async 未必 supported，需查 CUDA 13 docs。
收益预估 < 200 us，优先级低于 B-alt-3 和 C。

## 瓶颈已钉（基于 EP32 t=8192 CSV 数据，**不是猜测**）

| 序号 | 阶段 | 来源 | us @ EP32 t=8192 | 分类 |
|---|---|---|---|---|
| #1 | `cudaMemcpy2DAsync`（dispatch 末尾 compact recv_buf → output）| 3.76 GB, ~1688 GB/s | **~2100** | size-proportional |
| #2 | `cudaMemsetAsync(combine_local_va)`（combine 开头清 max_tokens × 32 × hidden）| 3.76 GB | **~500–2500** | size-proportional |
| #3 | 4× `ncclBarrier`（2 per dispatch + 2 per combine，每次含 `cudaDeviceSynchronize`）| fixed | **~120–250** | fixed |
| #4 | `fullmesh_dispatch_kernel` 本身 | 262144 blocks × 14336 B | 1586 | kernel |
| #5 | `fullmesh_combine_kernel_push`/`_reduce` | 同上 | ~998 avg/launch | kernel |

数据来源：`~/fizhang/nccl-sweep-20260423_171353/all_results.csv`，重算 `dispatch_avg_us - dispatch_kernel_us` 得 non-kernel 时间，对比 t=128 vs t=8192 两组得 fixed vs variable。

Phase 2 tag message (`git show phase2-done`) 里已记录完整 24 行数据；这里不复述。

## Phase 3 commit 计划（按 ROI 从高到低）

### Commit A — 缩小 combine memset 范围（ROI 最高，先做）

**昨晚错误结论更正**：原本以为"reduce kernel 只 loop 0..num_topk-1 所以可以删
memset"。实际上 **dispatch kernel 对 `topk_idx[t][k] == -1` 会 early-exit 不
carve slot 不 push** (`device/fullmesh.cu:35` 的 `if (eid < 0) return;`)。对
masked 场景（LL 模式 `generateRandomTopkIndicesLL` 会产生 -1，生产 API 也可
能有），token t 的某些 k 根本没 dest 往 combine_buf[t][k] push，reduce kernel
loop 到 k 时会**读上次 iter 残留值**。必须 memset。

**真正的修复**：保留 memset，但只清**当前 handle 实际用到的 `num_topk` 列**，
不清 `max_topk_for_combine=32` 的 padding 列。layout 不变 (`[max_tokens]
[max_topk][hidden]`)，memset 改成 `cudaMemset2DAsync`：

```cpp
// Old (full):
cudaMemsetAsync(combine_local_va, 0, max_tokens * 32 * hidden);

// New (只清活跃 num_topk 列):
cudaMemset2DAsync(
    combine_local_va,                      // dst
    32 * hidden_bytes,                     // dst_pitch  (max_topk * hidden)
    0,                                     // value
    handle->num_topk * hidden_bytes,       // width  (只清前 num_topk 列)
    max_tokens);                           // height
```

**改动位置**：
- `contrib/nccl_ep/nccl_ep.cc` ncclEpCombine FULLMESH 分支：`cudaMemsetAsync`
  → `cudaMemset2DAsync`，用 `handle->num_topk` 计算 width
- `contrib/nccl_ep/device/fullmesh.cuh`:
  `fullmesh_combine_kernel_reduce` doc 从 "Assumes the user cudaMemsetAsync'd
  combine_local_va to zero" 改成 "Caller must zero slots `[t, 0..num_topk-1,
  :]` for all t; slots at k >= num_topk are never touched by push nor read
  by reduce".
- `contrib/nccl_ep/device/fullmesh.cu`: **reduce/push kernel 无需改**，它们
  本来就只访问 `k < num_topk` 区域。

**预测节省**（memset 量从 `max_topk=32` 列 -> `num_topk=8` 列，**4× 缩减**）:
EP4 num_topk=8, max_topk_for_combine=32:
- EP32 t=8192: 3.76 GB → 0.94 GB memset，按 ~1500 GB/s 估节省 **1880 us**
- EP32 t=4096: 1.88 GB → 0.47 GB，节省 **940 us**
- EP4 t=128: 58 MB → 14 MB，节省 **30 us**

**落地后必须实测**（rule `benchmark-breakdown-first.mdc` 要求）：
- 预期实测节省区间 500–2000 us @ EP32 t=8192（区间宽因为 HBM3e effective
  rate 受 memset 粒度影响，1500 GB/s 是估计不是测过）
- 如果实测 < 500 us 或 > 3000 us，说明模型错，回头再分析 memset throughput

**防回退的 hard check**：reduce kernel 在 `if (k_in_topk < 0 || k_in_topk >=
max_topk_for_combine)` 已 early-exit，push kernel 也有同样 guard。所以
num_topk > max_topk_for_combine 的路径已被拒绝，不存在越界风险。

### Commit B — Dispatch fused-write，干掉 memcpy2D（ROI 第二高，改动中等）

**问题**：dispatch kernel 写 `recv_buf[meta + payload]`，然后 `cudaMemcpy2D` 把 payload 拷到 user output tensor。HBM-to-HBM 纯浪费 3.76 GB / 次。

**修复**：dispatch kernel 同时写两份 —
- meta (16 B) → `peer_recv_buf[dest][src][slot][0..16]` (combine push kernel 读这里的 src_rank/src_token_id/k_in_topk)
- payload (hidden_bytes) → **直接写 local output tensor**[`src * max_tokens + slot`][`0..hidden_bytes`]

这样：
- recv_buf 可以 **只存 meta**（16 B × nRanks × max_tokens），大小从 3.76 GB 砍到 4.2 MB
- init_fullmesh_intranode_fabric 需要重算 bytes_per_entry = meta_bytes only，buffer 更小，fabric handle 不变
- combine push 读 meta from recv_buf，读 payload **from user's output tensor**（而不是 recv_buf）

**改动位置**：
- `device/fullmesh.cu`:
  - `fullmesh_dispatch_kernel` 签名加 `void* output`，kernel 里 warp cooperative store payload 到 `output[i*hidden_u4]`，meta 还是写 peer recv
  - 删 `launch_compact_to_output` 实现（留 wrapper empty 或直接删）
  - `fullmesh_combine_kernel_push` 签名加 `const void* ffn_output`（已经有），读 payload from output 而不是 recv
- `device/fullmesh.cuh`: 更新 contract
- `nccl_ep.cc`:
  - init_fullmesh_intranode_fabric：`bytes_per_entry = meta_bytes`（删 `+ token_bytes`）
  - ncclEpDispatch FULLMESH 分支：kernel launch 加 `output` 指针，删 `launch_compact_to_output` 调用
  - ncclEpCombine FULLMESH 分支：launch_combine_push 的 ffn_output 参数不变（它本来就是 user 的 output tensor，commit 4 已这样）

**预测节省**：
- EP32 t=8192: 2000 us (dispatch memcpy2D 全没)
- + recv_buf 从 3.76 GB 砍到 4.2 MB，group_create 也变快 ~50 ms

### Commit C — ncclBarrier → ncclBarrierSession (Q1=C2)

预期节省 ~150 us fixed + 解锁 CUDA Graph capture。改动 ~100 行。参考 `docs/examples/06_device_api/03_alltoall_hybrid/main.cu:82` 的 `ncclBarrierSession<ncclCoopCta>` 用法。

### Commit D — TMA bulk async copy

Hopper/Blackwell 的 `cuda::memcpy_async` / `cp.async.bulk.tensor.1d` 替换 uint4 cooperative store。预期 kernel 10–20% 提升。放最后因为 kernel 已接近 fabric 上限，收益最小。

## 目标

Commit A + B 落地后 EP32 t=8192 D+C 从 **6546 us** 降到 **~3746 us**（**比 HT 5341 us 快 30%**）。
Commit C + D 再压到 **~3000 us**（比 HT 快 40%）。

## 明天开工第一步（唯一要做的事）

开 Commit A。只改 2 个文件（`nccl_ep.cc` 删 1 行 + `fullmesh.cuh` 改 1 段 doc），commit + push + 让用户重跑 sweep。实测节省贴回来后看和预测差多少，决定是否进 Commit B。

不用事先问"要不要开工"；`phase2-done` tag 上用户已明确同意继续优化（"做到极致"）。

---

**Session 入口提示**（给下次 session 的 agent）：
1. 读 `phase2-done` tag message 获得完整 Phase 2 历史
2. 读 `.cursor/rules/benchmark-breakdown-first.mdc` 的约束
3. 读 `.cursor/rules/one-step-at-a-time.mdc` 的聚合 / 单步原则
4. 直接开 Commit A，不重复问用户同一件事
