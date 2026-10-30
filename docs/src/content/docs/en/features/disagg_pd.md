---
title: "Disaggregated PD"
sidebar:
  order: 30
---
## Background
LLM online inference services typically need to meet two performance metrics: TTFT and TPOT. Traditional Contiguous Batching scheduling strategies mix Prefill and Decode requests during scheduling, causing Prefill and Decode phases to compete for computational resources. This prevents maximized utilization of computing resources and impacts performance metrics. To resolve this conflict, the Prefill and Decode phases are split to run on independent computational resources, enabling parallel execution. This simultaneously reduces TTFT and TPOT while improving throughput.

## Introduction
The xLLM PD Separation feature is primarily implemented through the following three modules:  

- **etcd**: Stores metadata such as instance information.  
- **xLLM Service**: Schedules requests and manages all computing instances.  
- **xLLM**: Handles request computation instances.  

The overall architecture is shown below: 
![xLLM PD Separation Architecture](../../assets/pd_architecture.jpg)

## Usage
### Preparation
#### Install Dependencies
- **xLLM**: Refer to [Installation && Compilation](/en/getting_started/quick_start/)
- **xLLM Service**: Refer to [xLLM Service](https://github.com/xLLM-AI/xllm-service)

#### Unified API error compatibility

Upgrade xLLM Service before xLLM. `GenerationStatus.error` (field 5) carries the
complete public error: HTTP status, type, code, message, optional parameter, and
retryability. xLLM maps the business error once; Service renders that object for
HTTP/SSE without remapping it. Fields 1–4 retain their internal-status semantics;
when `error` is absent, Service uses the legacy mapping. Older readers ignore the
new field, so complete error parity is not guaranteed during a mixed-version
rollout. The callback RPC acknowledgement is not the business success status.

#### Obtain Environment Information  
Deploying Disaggregated PD Service requires obtaining the Device IP of the machine to create communication resources. Execute the command `cat /etc/hccn.conf | grep address` on the current AI Server to get the Device IP, for example:
```
address_0=xx.xx.xx.xx
address_1=xx.xx.xx.xx
```
`address_xx` represents the Device IP.

### Start Disaggregated PD Service
1. Start etcd
```bash
./etcd
```
2. Start xLLM Service
```bash
ENABLE_DECODE_RESPONSE_TO_SERVICE=true ./xllm_master_serving --etcd_addr="127.0.0.1:12389" --http_server_port 28888 --rpc_server_port 28889 --tokenizer_path=/path/to/tokenizer_config_dir/
```
3. Start xLLM  
- Taking Qwen2-7B as an example  
    - Start Prefill Instance
        ```bash
        ASCEND_RT_VISIBLE_DEVICES=0 /path/to/xllm --model=Qwen2-7B-Instruct \
               --port=8010 \
               --master_node_addr="127.0.0.1:18888" \
               --enable_prefix_cache=false \
               --enable_chunked_prefill=false \
               --enable_disagg_pd=true \
               --instance_role=PREFILL \
               --etcd_addr="127.0.0.1:12389" \
               --transfer_listen_port=26000 \
               --disagg_pd_port=7777 \
               --node_rank=0 \
               --nnodes=1
        ```
    - Start Decode Instance 
        ```bash 
        ASCEND_RT_VISIBLE_DEVICES=1 /path/to/xllm --model=Qwen2-7B-Instruct \
               --port=8020 \
               --master_node_addr="127.0.0.1:18898" \
               --enable_prefix_cache=false \
               --enable_chunked_prefill=false \
               --enable_disagg_pd=true \
               --instance_role=DECODE \
               --etcd_addr="127.0.0.1:12389" \  
               --transfer_listen_port=26100 \
               --disagg_pd_port=7787 \
               --node_rank=0 \
               --nnodes=1
        ```
    Important notes:
    
    - PD disaggregation requires reading the `/etc/hccn.conf` file. Make sure this file on the physical machine is mapped into the container.

    - `etcd_addr` must match the `etcd_addr` of `xllm_service`

## Notice
With the chunked-prefill PD scheduler, the Prefill instance supports prefix cache. When enabled, the scheduler matches existing prefix-cache blocks before calculating the current chunk budget, so cached prompt blocks are not recomputed:
```shell
--enable_chunked_prefill=true
--enable_prefix_cache=true
```

## Phase latency monitoring

Scrape `/brpc_metrics` on each xLLM instance. `xllm_request_*_microseconds`
contains complete local single-output requests only. PD exports separate
`xllm_pd_request_*_microseconds` series; do not mix the two cohorts.

| PD metric | Role | Boundary |
| --- | --- | --- |
| `xllm_pd_request_queue_microseconds` | `prefill` / `decode` | Admission to this role's scheduling wait domain until its first actual batch selection |
| `xllm_pd_request_prefill_microseconds` | `prefill` | First batch selection to first confirmed core output; sampled at stage completion, before generation terminal |
| `xllm_pd_request_decode_microseconds` | `decode` | Local receipt of migrated tokens to last local core output, including local KV preparation and scheduling waits but excluding preceding inter-host transfer |
| `xllm_pd_request_tpot_microseconds` | `decode` | Local Decode duration divided by newly committed tokens after receipt; imported tokens are excluded from the denominator |

The current bvar exporter flattens the role into the name, for example
`xllm_pd_request_queue_microseconds_count_role_prefill` and
`xllm_pd_request_decode_microseconds_latency_role_decode`. These are
LatencyRecorder counts/latency statistics, not Prometheus `_bucket` histograms.
`role` identifies the executed phase, not the process's configured `instance_role`.
An OOC Decode segment resumed on a Prefill-configured process remains `decode`;
do not filter these series solely by process configuration or interpret them as
the cumulative work of a dedicated Decode host.
Importing P's Queue/Prefill on D does not sample them again. Requests already
cancelled/failed at P sampling are excluded; an already sampled completed P stage
is not retracted by a later handoff/Decode failure. D only samples successfully
completed local segments. Unknown-boundary and unsupported multi-output/scoring
requests are excluded from complete-stage samples. No newly committed local token means no Decode/TPOT
sample, not an artificial zero. PD-OOC returns reset the local wait/token
baseline: D statistics describe the final successfully completed local segment,
not accumulated latency across all migration segments.
Do not sum these role-local stages into PD end-to-end latency: D Queue is a
subinterval of D Decode, and inter-host transfer is not covered by these stages.

`server_time_to_first_token_latency_milliseconds` ends at scheduler result
readiness before detokenization/HTTP delivery in fused deployments, or at the
successful handoff ACK on the original PD ingress. It can include multi-output
requests and is not interchangeable with the complete local single-output
`xllm_request_ttft_microseconds` cohort or client SSE TTFT.
Sampling does not depend on INFO logging. The Python SDK's `disable_log_stats`
option disables per-request statistics logs. The native C++ executable does not
expose that CLI flag; `--minloglevel=1` only suppresses INFO output.

### Sample coverage and limitations

`xllm_request_phase_skipped_total` counts excluded samples per fixed `phase` and
`reason`, not repeated observer calls. Local phases are `queue/prefill/decode/tpot`;
PD phases are `pd_prefill_queue/pd_prefill` and `pd_decode_queue/pd_decode/pd_tpot`.
Reasons are `cancelled`, `failed`, `unsupported_output` (including multi-output
and beam), `single_token`, `no_local_tokens`, `missing_start`, `missing_end`, `unknown_boundary`,
`interrupted`, and `invalid_sample`. Internal graph warmup is excluded. Role
routing, imported upstream phases, and non-terminal waits are not skipped samples.
Later cancellation does not recount an already sampled P stage or successful terminal.
Fused single-token requests retain Queue/Prefill samples but skip Decode/TPOT
with `single_token`; their zero-length interval is not included in quantiles.
Scheduler admission rejections count as `failed` even before execution starts.

The exporter flattens labels, for example
`xllm_request_phase_skipped_total_phase_queue_reason_cancelled`. Within the same
process and time window, phase coverage is `delta(count) / (delta(count) +
sum(delta(skipped) for this phase))`. This cohort contains finalized/sampled
stages, not all ingress requests including in-flight work. Do not sum phases to
count excluded requests; missing Queue need not exclude known Decode. Quantiles
describe complete supported samples, not cancelled/failed request tail latency,
user success rates, or client SLOs. TTFT is sampled separately at first output
and does not use this four-phase denominator.

These phase bvar metrics aggregate at process scope without a model dimension. Multiple
models sharing a process cannot be compared using these series; averaging
process quantiles does not produce a cluster quantile. No high-cardinality model
label is introduced here.
