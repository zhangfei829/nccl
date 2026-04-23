# 明天继续 — Phase 3 FULLMESH latency 极致化

**当前位置**：`phase2-done` tag (commit d6c4307) + cursor rule `benchmark-breakdown-first.mdc` (commit 100947d)
已推送到 origin/master。Phase 2 功能全绿，24/24 sweep PASS。

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

### Commit A — 删 combine memset（ROI 最高，改动最小，先做）

**问题**：`ncclEpCombine` FULLMESH 分支开头 `cudaMemsetAsync(combine_local_va, 0, max_tokens*32*hidden)`。原设计是让 reduce kernel 无脑 loop `0..max_topk-1` 求和，slot `k >= num_topk` 必须是 0。

**修复**：reduce kernel 只 loop `0..num_topk-1`。push kernel 本来就只写 `k < num_topk` 的 slot（已是此行为，commit 4 里的 defensive guard 保证）。**删 memset 不影响正确性**。

**改动位置**：
- `contrib/nccl_ep/nccl_ep.cc` ncclEpCombine FULLMESH 分支：删 `cudaMemsetAsync(combine_local_va, ...)` 一行
- `contrib/nccl_ep/device/fullmesh.cu` `fullmesh_combine_kernel_reduce` 已经是 `for (int k = 0; k < num_topk; ++k)`，**无需改 kernel**
- `contrib/nccl_ep/device/fullmesh.cuh` 更新 doc：从 "Assumes the user cudaMemsetAsync'd combine_local_va to zero" 改成 "reduce kernel bounds its loop by num_topk, no memset required"

**预测节省**：
- EP32 t=8192: 500–2500 us
- EP32 t=4096: 250–1200 us
- EP4 t=128: ~70 us

**commit message 必须记录**：预测节省区间 + 落地后实测差值（规则 `benchmark-breakdown-first.mdc` 要求）。

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
