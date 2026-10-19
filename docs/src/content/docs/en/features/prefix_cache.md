---
title: "Prefix Cache Optimization"
sidebar:
  order: 50
---
## Feature Introduction
xLLM supports prefix cache matching. The prefix cache is based on `murmur_hash` and uses an LRU eviction policy, delivering superior matching efficiency and increased prefix cache hit rates.
Additionally, the prefix cache has been optimized to support the `continuous_scheduler`, `chunked_scheduler`, and `zero_evict_scheduler`. The cache is updated immediately after prefill operations, enhancing matching timeliness. For the `chunked_scheduler`, multi-stage chunked prefill matching is supported, reducing computational overhead and minimizing KV cache usage as much as possible.

## Usage
The prefix cache is implemented in xLLM and exposed through gflags parameters to control its functionality.

- Enable prefix cache with specific policy and settings:
```
--enable_prefix_cache=true
```

## Cache-Aware DP Routing

When running with data parallelism (`dp_size > 1`), requests can be routed to the DP rank that holds the longest prefix-cache hit, improving KV cache reuse across ranks.

- Enable cache-aware DP routing:
```
--enable_prefix_cache=true --enable_prefix_cache_aware_dp_routing=true
```

- `prefix_cache_aware_dp_match_threshold` (default `0.5`): minimum prefix block hit ratio to prefer the cache-affinity rank. Below this threshold, routing falls back to free-block balancing.
- `prefix_cache_aware_dp_imbalance_threshold` (default `0.1`): maximum cross-rank KV utilization gap (`(max_used-min_used)/total_blocks`). Exceeding this disables affinity routing and selects the least-loaded rank.

## Performance Impact
After enabling prefix cache, on the Qwen3-8B model with a TPOT constraint of 50ms, the E2E latency **decreased by 10%**.

## Upgrading LINEAR State Caches

Regular LINEAR checkpoints and prompt-tail checkpoints use a hash chain that is
separate from the KV chain and indexed at KV-block granularity. When upgrading
from a version that hashes LINEAR states at chunk granularity, existing LINEAR
entries in Mooncake Store may miss and need to be recomputed. Old entries are
reclaimed by the Store eviction policy. KV entry keys are unchanged. The
in-process Host prefix cache is rebuilt on service restart.

With LINEAR prefix cache enabled, chunked prefill stops at regular checkpoints
and reusable tail boundaries to save the state at the corresponding token count.
Unaligned prompts may require more prefill steps. TTFT and throughput benefits
depend on prefix reuse and should be measured separately for cold and warm caches.
The Qwen3-8B result above is not a benchmark of this change.

:::note
For disaggregated PD, prefix cache is supported in specific scheduler roles. See [Disaggregated PD](/en/features/disagg_pd/) for the supported configurations.
:::
