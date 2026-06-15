<!-- Copyright 2026 The xLLM Authors. All Rights Reserved. -->

# Hierarchy → Composite 迁移设计（device + host group 化）

> **范围与状态。** 本文是把现状 **继承式** `HierarchyBlockManagerPool`（device/host 双 `KVCacheState` 仍用扁平 `blocks_` + 扁平 index 配对）迁移到 composite 形态的**落地 playbook**，**只出设计、不写代码**。它衔接两份既有文档：
> - 《Block Manager 架构重构方案》(`block_manager_architecture_design.md`)——目标分层（`HierarchyBlockManagerPool` 组合 device pool + host pool，两侧都跑 `CompositeBlockManager`）。该文 `HierarchyBlockManagerPool` 一节标注「实现状态（2026-06）：后置，未开始。当前实现保持继承式」。
> - 《Prefix Host Cache Transfer》(`prefix_host_cache_transfer_design.md`)——transfer/offload/prefetch 的 envelope 与 hash 语义。
>
> 本文不重新推导目标架构（见上述两文），只给出**从当前继承式实现出发**、删除 `KVCacheState::blocks_` 所需的**逐调用点改造方案**。本文是 block-manager 重构总计划 Phase C 的产物；Phase A（xtensor→composite C1 leaf）、Phase B（标注 hierarchy-only 面）已落地，Phase D（真正删除 `blocks_`）在本文设计落地后的未来一轮执行。

---

## 1. 为什么 `blocks_` 还不能删

block-manager 重构的终态是 `KVCacheState` 只保留 `std::vector<CacheGroupState> groups_`，删掉扁平字段 `blocks_` / `src_blocks_` / `num_owned_shared_blocks_`，以及 `kv_blocks()` / `mutable_kv_blocks()` / `c1_view_group()` 等读写桥。

Phase A 把 xtensor 折叠进 composite 后，唯一仍走非 composite 路径的「孤岛」是 **hierarchy host 模式**：

- 选路在 `llm_engine.cpp:567`：`host_blocks_factor() > 1.0 || enable_kvcache_store()` ⇒ `HierarchyBlockManagerPool`，否则普通 `BlockManagerPool`。
- 一条 sequence 同时持有 **两份** `KVCacheState`：device `kv_state_`（`sequence.h:463` 区）和 host `host_kv_state_`（`sequence.h:465`）。两者**都**用扁平 `blocks_`。
- `HierarchyBlockManagerPool`（`hierarchy_block_manager_pool.cpp`）的每个方法都依赖**扁平 index 配对不变量**：device `blocks_[i]` ↔ host `blocks_[i]`，`OffloadBlockPair{src,dst}` 把 device block 和 host block 并排携带。

因此删 `blocks_` 会**同时**打断 device 端和 host 端两条扁平表。Phase B 已把所有「仅 hierarchy」面标注为 `// hierarchy-only; delete in Phase D`，本文负责设计这些面在 hierarchy 上 composite 化后的替代方案，使 Phase D 能安全删除。

---

## 2. 当前扁平 index 配对不变量（要被替换的东西）

`HierarchyBlockManagerPool` 现状全部建立在「device 与 host 的 `blocks_` 按下标一一对应」之上：

| 调用点（`hierarchy_block_manager_pool.cpp`） | 现状依赖的扁平不变量 |
|---|---|
| `deallocate` (53–103) | `mutable_kv_blocks()` 取 device/host 两个扁平 vector；`blocks->at(i)` 与 `host_blocks->at(i)` 同下标配对，按 `ref_count()==2` 判定 offload，构造 `OffloadBlockPair{device[i], host[i]}` |
| `allocate(num_tokens, max_copy_in)` (105–157) | `kv_blocks()` 取 device/host 扁平 slice，按 `i`（block 下标）配 `BlockTransferInfo{host[i].id, hbm[i].id, H2D}` |
| `allocate(num_tokens)` (159–190) | 同上，按下标配 H2D load 计划 |
| `allocate_shared` (192–198) / `allocate_host_shared` (200–207) | host `allocate_shared(tokens)` 直接产出扁平 host shared blocks，`add_shared_host_kv_blocks` 写 host `blocks_` |
| `prefetch_from_storage` (209–264) | host `blocks_` 扁平下标 + `shared_blocks_num` 偏移构造 G2H `BlockTransferInfo` |
| `update_prefetch_result` (266–291) | host `kv_blocks()` 扁平 slice `cache(...)` |
| `transfer_blocks(batches)` (293–307) | 把 `load_block_transfer_infos_`（已是 id 级 envelope）下发，本身不依赖扁平下标，但其内容由上面几处按扁平下标构造 |
| `transfer_blocks()` (309–355) | 从 `offload_block_pair_queues_` 取出 `OffloadBlockPair`，按 id 构造 D2G envelope；offload 完成回调里 `block_managers_[i]`/`host_block_managers_[i]` 直接 deallocate device/host block |

关键观察：

1. **id 级 envelope 已经存在。** `BlockTransferInfo`（`model_input_params.h:634`）和 `OffloadBlockPair` 携带的是 **block id**，不是下标。真正用扁平下标的只是**配对选取**那一步（决定哪个 device block 配哪个 host block）。
2. **`BlockTransferInfo` 已带 `PrefixCacheGroup group` 字段**（`model_input_params.h:638`，默认 `INVALID`）。per-group 化所需的 group 标识**不需要新增字段**，只需在构造点把 `group` 填成 `PrefixCacheGroup::C1`（phase-1 host 只镜像 C1）。
3. **worker 侧不读 host 扁平表。** worker forward 只读 device `kv_state()` 的 block table（H2D 完成后 device 视图已恢复，见架构文档 line 1034）。host 状态只服务 offload/prefetch/host prefix cache。⇒ host composite 化对 worker **零影响**，耦合面只在 `HierarchyBlockManagerPool` 内部。

所以迁移的本质：**把「按扁平下标配对」换成「按 `state_id`（C1）配对，组内按位次配对」**，envelope 结构和下发链路不变。

---

## 3. 目标设计

### 3.1 host 侧 composite

- host 端也用 `GroupCompositeBlockManager`：host C1 group 的 leaf allocator 是按 `host_num_blocks` 定容的 host `BlockManagerImpl` / `ConcurrentBlockManagerImpl`（沿用现状 `host_block_managers_` 的并发选择：`enable_disagg_pd || enable_kvcache_store` 时用 Concurrent）。
- `host_kv_state_` 改持 `groups_`（单 C1 group），不再用扁平 `blocks_`。
- **host 是否需要 SINGLE_RES：不需要。** host 只镜像 C1 attention KV（offload/prefetch 的对象），不承载 per-sequence linear/embedding 资源——那是 device 侧 SINGLE_RES 的职责。host composite 因此是**单 C1 group**。在 host spec builder 入口显式断言 host 只构造 C1，不追加 SINGLE_RES。
- host composite 的 prefix cache：沿用现状 host `BlockManagerImpl::allocate_shared` 的语义，但收敛为 C1 group-local PrefixCache（与 device C1 同构）。**注意**现状 host 是直接调 allocator 的 `allocate_shared(tokens)`（leaf 自带 prefix cache），而 device composite 的 leaf 是纯池（`enable_prefix_cache(false)`）+ 独立 group-local PrefixCache。host composite 必须对齐 device 形态：leaf 纯池 + group-local PrefixCache，match/insert 经 composite 的 `match_prefix_cache` / 内部 insert，不再直接打 leaf 的 `allocate_shared`。

### 3.2 per-group offload 配对取代扁平 index 配对

- device C1 group ↔ host C1 group **按 `CacheStateId::C1` 匹配**，组内按位次（block 在该 group 内的序号）配对。
- phase-1 **仅 C1 offload**：host 没有 SWA / compressed / SINGLE_RES 镜像，因此不存在 C4/C128/SWA/SINGLE_RES 的 offload。文档与代码都应显式写出这一约束（host composite 只有 C1，遍历 offload 计划时只枚举 C1 group）。
- 「组内按位次配对」替代「`blocks_[i]` ↔ `host_blocks_[i]`」：从 device C1 `CacheGroupState::blocks` 和 host C1 `CacheGroupState::blocks` 各取第 `k` 个 block 配成一对。两侧 group 的 block 数可能不同（host 落后于 device 时按 `cached_host_block_num` 区间配，沿用现状区间逻辑，只是数组来源从 `blocks_` 换成 C1 group 的 `blocks`）。

### 3.3 `OffloadBlockPair` —— 携带 group，配对方式改 per-group

- 结构 `{Block src; Block dst;}` **保留不变**；改的是**选取方式**：`src` / `dst` 从 device/host **C1 group** 的 `blocks` 取，而非扁平 `blocks_`。
- 可选增强（非必须）：给 `OffloadBlockPair` 加一个 `CacheStateId state_id = CacheStateId::C1` 字段，便于将来扩展多 group offload 时区分。phase-1 因恒为 C1，可不加，留待未来。本文建议**先不加**，保持 diff 最小；多 group offload 引入时再加。

### 3.4 `BlockTransferInfo` —— 复用既有 `group` 字段

- `BlockTransferInfo` **已有** `PrefixCacheGroup group`（`model_input_params.h:638`）。per-group 化**不新增字段**：在 hierarchy 构造 H2D / G2H / D2G envelope 的每一处，把 `group` 显式填 `PrefixCacheGroup::C1`（现状构造点用的是不带 group 的重载，group 默认 `INVALID`）。
- 这与《Prefix Host Cache Transfer》line 1462 的要求一致：「新的 transfer/offload 路径必须保证 group != INVALID」。
- **跨进程影响**：`BlockTransferInfo` 经 `engine_->transfer_kv_blocks` / `prefetch_from_storage` 下发到 worker / KVCache store。若该路径跨进程序列化，填充 `group` 字段需同步检查 proto / 序列化定义是否已含 `group`（`PrefixCacheGroup` 是 enum，序列化为整型）。落地前**必须核对** `engine_->transfer_kv_blocks` 签名与其 proto，确认 `group` 字段已在序列化集合内或需补充——这是本设计唯一的跨进程风险点。

### 3.5 prefetch / update_prefetch / transfer 改 per-group

- 所有「按 host 扁平下标 + `shared_blocks_num` 偏移」构造的描述符，改为「枚举 host C1 group 的 `blocks`，按 group 内位次 + group 的 `shared_blocks_num` 偏移」。
- `load_block_transfer_infos_` / `offload_block_pair_queues_` 的索引维度保持「per DP rank」，phase-1 不需要再按 group 细分（只有 C1）；但注释应写明「当前仅 C1，多 group offload 时按 (DP rank, group) 标记」，为未来留接口语义。

---

## 4. `HierarchyBlockManagerPool` 各方法改造（文档级）

下面把 §2 表格里的每个调用点映射到 per-group 替代方案。**数组来源**统一从「device/host `KVCacheState` 的扁平 `blocks_`」换成「device/host C1 group 的 `CacheGroupState::blocks`」；**配对**统一从「扁平同下标」换成「C1 group 内同位次」。

### 4.1 `deallocate` (53–103)

- `cache(sequence)`：device 侧已走 composite 内部 insert（Phase A/B 后 `BlockManagerPool::cache` 在 composite 路径是 no-op），host 侧改为经 host composite 的内部 insert（C1 group-local PrefixCache）。
- device/host block 数组：`sequence->kv_state().group_blocks(C1)` 与 `sequence->host_kv_state().group_blocks(C1)`，取代 `mutable_kv_blocks()`。
- offload 判定 `ref_count()==2`、区间 `[cached_host_block_num, host C1 size)`：逻辑不变，只是下标作用在 C1 group 的 `blocks` 上。
- 追加 host block：`host composite allocate` C1 group（或直接 host C1 leaf `allocate`，沿用现状「按需补块」语义），写入 host C1 group state，取代 `host_kv_state().add_kv_blocks(...)`。
- 构造 `OffloadBlockPair`：`src` = device C1 `blocks[i]`，`dst` = host C1 `blocks[i]`，填 `group=C1`。
- 释放：device/host composite 的 `deallocate`（释放各自所有 group），取代 `block_managers_[dp]->deallocate(kv_blocks())` + `deallocate_single_block`（SINGLE_RES 由 device composite 内部释放，host 无 SINGLE_RES）。

### 4.2 `allocate(num_tokens, max_copy_in_blocks_num)` (105–157) 与 `allocate(num_tokens)` (159–190)

- 先 `BlockManagerPool::allocate`（device composite，已是 composite 路径）。
- host shared：首次调度且非 DECODE 时 `allocate_host_shared`（见 4.3）。
- H2D load 计划：device/host block 改取 C1 group 的 `blocks`；`for i in [hbm_cache_token_num/block_size, ...)` 区间逻辑不变，构造 `BlockTransferInfo{host_C1[i].id, hbm_C1[i].id, H2D, group=C1}`。
- `incr_kv_cache_tokens_num` 不变（token 计数与 group 无关）。

### 4.3 `allocate_shared` (192–198) / `allocate_host_shared` (200–207)

- device 侧 `BlockManagerPool::allocate_shared` 已 composite（首次调度经 `composite_match_shared` 命中 device C1 group）。
- host 侧改为：经 host composite 的 `match_prefix_cache`（host context，指向 `sequence->host_kv_state()`）命中 host C1 group，再 attach 到 host C1 group state，取代现状直接 `host_block_managers_[dp]->allocate_shared(tokens)` + `add_shared_host_kv_blocks`。
- host context 构造复用架构文档 §（`BlockManagerContext` 指向 `host_kv_state()`）的形态。

### 4.4 `prefetch_from_storage` (209–264)

- host shared match 同 4.3（host composite match）。
- 补块 `allocate(num_additional_blocks)` → host C1 group 分配；`PrefixCache::compute_hash_keys` 作用在 host C1 group 的 `blocks` 上（取代 `mutable_kv_blocks()`）。
- G2H envelope：`BlockTransferInfo{-1, host_C1[shared+i].id, hash, G2H, group=C1}`；下发 `engine_->prefetch_from_storage` 不变。

### 4.5 `update_prefetch_result` (266–291)

- host block 数组取 host C1 group `blocks`；`cache(slice(...))` 改为 host composite 的 C1 group-local PrefixCache insert（取代 host leaf 直接 `cache`）。

### 4.6 `transfer_blocks(batches)` (293–307) 与 `transfer_blocks()` (309–355)

- `transfer_blocks(batches)`：仅下发 `load_block_transfer_infos_`，不依赖扁平下标，**无需改动**（其内容已由 4.2/4.4 按 per-group 构造）。
- `transfer_blocks()`：从 `offload_block_pair_queues_` 取 `OffloadBlockPair` 构造 D2G envelope，填 `group=C1`；offload 完成回调里的 deallocate 改为打 device/host **C1 group 的 leaf allocator**（通过 composite 暴露的 group-leaf 释放路径，或直接持有 leaf 指针——见下「释放路径」注）。

**释放路径注。** 现状回调直接持 `block_managers_[i].get()` / `host_block_managers_[i].get()` 调 `deallocate`。composite 化后，Block 析构本就直达 owner leaf allocator 的 `free()`（架构文档 line 282 的并发不变量）——offload 完成后让持有的 `OffloadBlockPair` 的 `Block` 析构自然回收，**优于**显式 deallocate，可去掉回调里对 manager 指针的依赖。落地时确认 host C1 leaf 的 `free()` 跨线程安全（folly global executor 回调线程），与架构文档 line 282 一致。

---

## 5. 终态声明（Phase D 删除集合）

hierarchy 完成 composite 化后，可在 Phase D 删除：

- `KVCacheState`：`blocks_`、`src_blocks_`、`num_owned_shared_blocks_`、`c1_view_group()`、`kv_blocks()` / `mutable_kv_blocks()` 的扁平 fallback（坍缩为 groups_-only 读）、`add_kv_blocks` / `add_shared_kv_blocks` / `incr_shared_kv_blocks_num` 等仅扁平的写入口（均已在 Phase B 标 `// hierarchy-only; delete in Phase D`）。
- `BlockManagerPool`：`block_managers_`、`single_block_managers_`、`composite_` 标志及全部 `composite_ ? ... : ...` 三元（坍缩为 composite 分支）、`allocate_single_block` / `deallocate_single_block`、扁平 `process_beam_search`、`reserve_xtensor_padding_blocks` 的 legacy 死分支。
- `HierarchyBlockManagerPool`：`OffloadBlockPair` / 队列改 per-group（源自本文 §3–4）；删残余扁平 index 辅助；`host_block_managers_` 收敛进 host composite。

`groups_` 成为 `KVCacheState` 唯一 KV 成员。

---

## 6. 落地 checklist（逐扁平配对调用点 → per-group 替代）

| 调用点（行号） | 现状扁平依赖 | per-group 替代（命名方案） | 风险 |
|---|---|---|---|
| `deallocate` (53–103) | device/host `mutable_kv_blocks()` 同下标，`OffloadBlockPair{blocks_[i], host_blocks_[i]}` | device/host `group_blocks(C1)` 同位次配对，`group=C1` | host 补块改 host composite 分配 |
| `allocate(num_tokens,max_copy_in)` (105–157) | device/host `kv_blocks()` 同下标 H2D | device/host C1 group `blocks` 同位次，`BlockTransferInfo{...,H2D,group=C1}` | 无（worker 不读 host） |
| `allocate(num_tokens)` (159–190) | 同上 | 同上 | 无 |
| `allocate_shared` (192–198) | host leaf `allocate_shared(tokens)` 扁平 | host composite `match_prefix_cache`(host ctx) attach host C1 | host leaf 改纯池+group PrefixCache |
| `allocate_host_shared` (200–207) | `add_shared_host_kv_blocks` 写 host `blocks_` | attach 到 host C1 group state | 同上 |
| `prefetch_from_storage` (209–264) | host `blocks_` 扁平下标 + `shared_blocks_num` 偏移 G2H | host C1 group `blocks` 位次 + group `shared_blocks_num`，`group=C1` | **`group` 字段跨进程序列化需核对** |
| `update_prefetch_result` (266–291) | host `kv_blocks()` 扁平 slice `cache` | host C1 group `blocks` slice → host composite C1 insert | 无 |
| `transfer_blocks(batches)` (293–307) | 无扁平依赖 | 不改 | 无 |
| `transfer_blocks()` (309–355) | `OffloadBlockPair` id 级 + 回调显式 deallocate manager | 同结构 + `group=C1`；释放改走 Block 析构直达 leaf | leaf `free()` 跨线程安全核对 |

**落地前置核对项（仅两处真正风险）：**
1. `engine_->transfer_kv_blocks` / `engine_->prefetch_from_storage` 的下发路径是否序列化 `BlockTransferInfo::group`（若跨进程则补 proto）。
2. host C1 leaf allocator 的 `free()` 在 folly global executor 回调线程下的并发安全（与架构文档 §并发一节对齐）。

其余调用点均为「数组来源 + 配对方式」的机械替换，无新增结构、无 worker 侧改动。

---

## 7. 与既有文档的关系

- **目标架构**（device pool + host pool 组合、两侧 Composite、`BlockManagerContext` 指向 device/host `kv_state`）：见 `block_manager_architecture_design.md`，本文不重复。
- **transfer/offload/prefetch envelope 与 hash 语义**（`group != INVALID`、Block 携带 `prefix_hash/token_end`、`CompositeMatchResult.matched_tokens` 比较 device/host 命中长度）：见 `prefix_host_cache_transfer_design.md`，本文的 per-group envelope 构造遵循其约定。
- 本文新增的内容仅限：**从现状继承式扁平实现出发**的逐调用点迁移映射，以及「删除 `blocks_` 的安全前提」清单。
