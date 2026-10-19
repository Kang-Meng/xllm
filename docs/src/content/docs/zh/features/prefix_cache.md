---
title: "Prefix Cache 优化"
sidebar:
  order: 50
---
## 功能介绍
xLLM支持prefix_cache匹配。prefix_cache基于mermer_hash，使用lru淘汰策略，提供更极致的匹配效率，同时提高prefix_cache命中率。
同时对prefix_cache进行了优化，支持continuous_scheduler、chunked_scheduler和zero_evict_scheduler，在prefill之后即更新
prefix_cache，提高匹配时效性，同时对于chunked_scheduler，支持多阶段chunked_prefill匹配，减少计算量并尽可能减少kv_cache占用。

## 使用方式
prefix_cache已在xLLM实现，并向外暴露gflag参数，控制功能的开关。

- 开启zero_evict策略，并设置max_decode_token_per_sequence。
```
--enable_prefix_cache=true
```

## Cache 感知 DP 路由

当使用数据并行（`dp_size > 1`）时，请求可被路由到持有最长 prefix cache 命中的 DP rank，提高跨 rank 的 KV cache 复用率。

- 开启 cache 感知 DP 路由：
```
--enable_prefix_cache=true --enable_prefix_cache_aware_dp_routing=true
```

- `prefix_cache_aware_dp_match_threshold`（默认 `0.5`）：选择 cache 亲和 rank 所需的最低 prefix block 命中比例，低于该阈值时回退到空闲 block 均衡。
- `prefix_cache_aware_dp_imbalance_threshold`（默认 `0.1`）：跨 rank KV 利用率差异上限（`(max_used-min_used)/total_blocks`），超过此值时路由到负载最低的 rank。

## 性能效果
开启prefix_cache之后，在Qwen3-8B模型上，限制TPOT50ms，E2E时延 **下降10%**。

## LINEAR 状态缓存升级说明

LINEAR 常规检查点和 prompt 尾检查点使用独立于 KV 的 hash 链，下标粒度是 KV block。
从使用 chunk 粒度 hash 链的版本升级时，Mooncake Store 中旧的 LINEAR 条目可能无法命中，
需要重新计算并填充缓存；旧条目按 Store 的淘汰策略回收。此变化不改变 KV 条目的 key。
进程内的 Host prefix cache 随服务重启重新建立。

启用 LINEAR prefix cache 后，chunked prefill 会在常规检查点和可复用的尾边界停下，
以保存对应 token 边界的状态。非对齐 prompt 可能增加 prefill 步数；其 TTFT 和吞吐收益
取决于前缀复用率，需要对冷缓存和命中缓存分别测量。上面的 Qwen3-8B 数据不代表此改动的性能结果。

:::note
PD分离下的prefix cache支持范围与调度器角色相关，参见[PD分离](/zh/features/disagg_pd/)中的支持配置说明。
:::
