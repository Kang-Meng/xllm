---
title: "PD分离"
sidebar:
  order: 30
---
## 背景
LLM在线推理服务通常需要满足TTFT和TPOT两项性能指标，而传统的Contiguous Batching调度策略将Prefill和Decode请求混合在一起调度，导致P和D会互相抢占计算资源，影响性能指标无法最大程度的利用计算资源。为解决上述矛盾，将Prefill和Decode两阶段拆分到独立的计算资源并行执行，从而同时降低TTFT和TPOT并提升吞吐量。
## 功能介绍
xLLM PD分离功能主要通过以下三个模块实现：

- **etcd**: 存储实例信息等元数据
- **xLLM Service**: 调度请求和管理所有计算实例
- **xLLM**: 请求计算实例

整体架构图如下：
![xLLM PD分离架构图](../../assets/pd_architecture.jpg)
## 功能使用示例
### 使用准备
#### 安装相关依赖
- **xLLM**: 参见[安装编译](/zh/getting_started/quick_start/)
- **xLLM Service**: 参见[xLLM Service](https://github.com/xLLM-AI/xllm-service)

#### 统一 API 错误兼容性

先升级 xLLM Service，再升级 xLLM。`GenerationStatus.error`（字段 5）传递完整
公开错误：HTTP 状态、type、code、message、可选 param 和 retryable。xLLM 只映射
一次，Service 直接渲染为 HTTP/SSE，不再二次映射。字段 1–4 保留内部状态语义；
没有 `error` 时使用旧版映射。旧版读取端会忽略新字段，因此混合版本期间不能保证
完整错误一致性。回调 RPC 的接收确认不代表业务成功。

### 启动PD分离服务
1. 启动etcd
```
./etcd
```
2. 启动xLLM Service
```bash
ENABLE_DECODE_RESPONSE_TO_SERVICE=true ./xllm_master_serving --etcd_addr="127.0.0.1:12389" --http_server_port 28888 --rpc_server_port 28889 --tokenizer_path=/path/to/tokenizer_config_dir/
```
3. 启动xLLM

4. 以Qwen2-7B为例
    - 启动Prefill实例
        ```bash
        ASCEND_RT_VISIBLE_DEVICES=0 /path/to/xllm --model=Qwen2-7B-Instruct \
               --port=8010 \
               --master_node_addr="127.0.0.1:18888" \
               --enable_prefix_cache=false \
               --enable_chunked_prefill=false \
               --enable_disagg_pd=true \
               --instance_role=PREFILL \
               --etcd_addr=127.0.0.1:12389 \
               --transfer_listen_port=26000 \
               --disagg_pd_port=7777 \
               --node_rank=0 \
               --nnodes=1
        ```
    - 启动Decode实例
        ```bash
        ASCEND_RT_VISIBLE_DEVICES=1 /path/to/xllm --model=Qwen2-7B-Instruct \
               --port=8020 \
               --master_node_addr="127.0.0.1:18898" \
               --enable_prefix_cache=false \
               --enable_chunked_prefill=false \
               --enable_disagg_pd=true \
               --instance_role=DECODE \
               --etcd_addr=127.0.0.1:12389 \
               --transfer_listen_port=26100 \
               --disagg_pd_port=7787 \
               --node_rank=0 \
               --nnodes=1
        ```
    需要注意：
    
    - PD分离需要读取`/etc/hccn.conf`文件，确保将物理机上的该文件映射到了容器中
  
    - `etcd_addr`需与`xllm_service`的`etcd_addr`相同

:::tip[Chunked prefill的Prefill实例支持prefix cache]
使用chunked-prefill PD调度器时，Prefill实例已支持prefix cache。开启后，调度器会先匹配已有prefix cache block，再计算当前chunk budget，避免重复计算已缓存的prompt block。
``` shell
--enable_chunked_prefill=true
--enable_prefix_cache=true
```
:::

## 阶段延迟监控

从每个 xLLM 实例的 `/brpc_metrics` 采集。普通部署的
`xllm_request_*_microseconds` 仅包含完整 local 单输出请求；PD 使用独立的
`xllm_pd_request_*_microseconds`，不能与普通部署的曲线混合比较。

| PD 指标 | 角色 | 测量边界 |
| --- | --- | --- |
| `xllm_pd_request_queue_microseconds` | `prefill` / `decode` | 本角色进入调度等待域至第一次实际选入 batch |
| `xllm_pd_request_prefill_microseconds` | `prefill` | 第一次选入 batch 至首个确认的 core 输出；阶段完成即采样，无需等待请求最终完成 |
| `xllm_pd_request_decode_microseconds` | `decode` | 本机接收迁移 token 至最后一个本地 core 输出，包含本地 KV 准备和调度等待，不含此前的跨机传输时间 |
| `xllm_pd_request_tpot_microseconds` | `decode` | 上述本地 Decode 耗时除以本次接收后新增的确认 token 数；迁移来的 token 不计入分母 |

当前 bvar 导出将角色拼入名称，例如
`xllm_pd_request_queue_microseconds_count_role_prefill`、
`xllm_pd_request_decode_microseconds_latency_role_decode`。它们是
LatencyRecorder 的计数/延迟统计，不是 Prometheus `_bucket` histogram。
这里的 `role` 表示实际计算阶段，而不是进程的 `instance_role` 配置。OOC 返回
到配置为 Prefill 的进程后，恢复执行的 Decode 段仍标为 `decode`；不能仅按进程
配置过滤该系列，也不能把它解读成某台 Decode 机器的累计指标。
P 的 Queue/Prefill 不会因在 D 导入而重复采样。P 阶段采样时已经取消/失败的请求
不计入；已完成并采样的 P 阶段不会因后续 handoff/Decode 失败而撤回。D 仅采成功
完成的本地段。未知边界和不支持的多输出/评分请求不作为完整阶段样本；没有新增
本地 token 时不伪造零 Decode/TPOT。
PD-OOC 返回会重置本地等待和 token 基线，D 指标仅描述最终成功完成所在的本地段，
不是跨多个迁移段的累计耗时。
这些角色本地阶段不能相加当作 PD 端到端延迟：D Queue 是 D Decode 的子区间，
跨机传输也不在上述阶段中。

`server_time_to_first_token_latency_milliseconds` 的普通部署终点是调度器结果就绪，
在 detokenize/HTTP 返回之前；PD 原始 ingress 的终点是成功 handoff ACK。
它可包含多输出请求，不等同于仅完整 local 单输出的
`xllm_request_ttft_microseconds`，也不等同于客户端 SSE TTFT。
指标采集不依赖 INFO 日志。Python SDK 的 `disable_log_stats` 选项可关闭逐请求
统计日志；原生 C++ 二进制不提供这个 CLI 参数，`--minloglevel=1` 仅抑制 INFO 输出。

### 样本覆盖率与限制

`xllm_request_phase_skipped_total` 按固定的 `phase` 和 `reason` 记录每个阶段
被过滤的样本，而非重复 observer 调用次数。普通请求的 phase 为
`queue/prefill/decode/tpot`；PD 为 `pd_prefill_queue/pd_prefill` 和
`pd_decode_queue/pd_decode/pd_tpot`。reason 包括 `cancelled`、`failed`、
`unsupported_output`（含多输出和beam）、`single_token`、`no_local_tokens`、`missing_start`、
`missing_end`、`unknown_boundary`、`interrupted` 和 `invalid_sample`。
内部 graph warmup 不计入。PD 角色分流、导入上游阶段和尚未结束的等待不算跳过；
已采样的 P 阶段或成功终态也不会因后续取消被再次计数。
Fused 单 token 请求保留 Queue/Prefill 样本，Decode/TPOT 以 `single_token`
跳过，不将零长度间隔写入分位数。调度器准入拒绝即使尚未执行，也计入 `failed`。

导出名称例如 `xllm_request_phase_skipped_total_phase_queue_reason_cancelled`。
在同一进程、同一时间窗内，每个阶段的覆盖率为
`该阶段新增 count / (该阶段新增 count + 该 phase 各 reason 的新增 skipped)`；
它针对已结束/已采样阶段，不是所有入口请求（包含在途请求）的完成率。各 phase
不能相加当作被跳过的请求数；缺 Queue 但有有效 Decode 时，仅 Queue 的覆盖率降低。
这些延迟分位数只描述该阶段完整、受支持的样本，不代表取消/失败请求的尾延迟，
更不能替代用户成功率或客户端SLO。TTFT独立在首输出采样，不使用上述四阶段分母。

这些阶段 bvar 指标为进程级聚合，没有 model 维度。多模型共用一个进程时不能据此
比较模型，也不能把全局分位数平均成集群分位数；本实现不增加高基数 model label。
