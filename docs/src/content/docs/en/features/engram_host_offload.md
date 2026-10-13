---
title: "Engram Host Offload"
sidebar:
  order: 83
---

## Overview

Under the DeepSeek-V4.1 W8A8 (Eco-Tech int8) export, the Engram n-gram table holds about **27.7 GiB of HBM per card**. `engram_config.cpu_offload` moves that table into **host memory** (pageable CPU memory) to free the HBM budget, at the cost of one NPU->CPU id sync per step plus a CPU int8 dequant and an H2D upload of the touched rows. That path is served by the fused CPU lookup op (`xllm_ops.engram_int8_lookup_cpu`); only the int8 table can be offloaded.

## Configuration (`--engram-config`)

The config is a **raw JSON string** carried by the C++ gflag `engram_config` and parsed in Python (`xllm/python/models/engram_config.py`). The flag is spelled `--engram-config`:

```bash
xllm --engram-config '{"cpu_offload":true,"dp_shared_memory":false}'
```

xLLM itself is **off by default**: with no `--engram-config`, `EngramConfig()` is `{"cpu_offload":false,"dp_shared_memory":false}` (equivalent to `{}`) and the HBM path is used; enabling offload requires **passing the flag explicitly** (e.g. `--engram-config '{"cpu_offload":true}'`). A deployment launcher that wants it on by default supplies the flag itself; xLLM does not know about that layer.

## Switch semantics

| Key | Default | Meaning |
|---|---|---|
| `cpu_offload` | `false` | Keep the int8 Engram table in host memory (the offload switch). |
| `dp_shared_memory` | `false` | Node-local DP replicas share **one** host table (saves host memory). **Lookup still runs per rank.** |

What `dp_shared_memory` shares is the **read-only TP-shard table data** (the int8 codes and fp32 scales in host memory), not lookup results: each rank still builds its n-gram hash ids from **its own input and history**, runs its own lookup + dequantize + H2D, and does not share the returned rows (the staging output) either. Lookup results are not shared because the hash ids already differ per rank: sharing them would first have to gather the ids in one place and broadcast them back, adding a round of communication and synchronization, so the design has each rank read its local copy of the shared table directly.

Parsing is **strict**: field values must be real booleans -- `"true"` / `0` / `1` are rejected -- and unknown keys, duplicate JSON keys and invalid JSON all raise.

### Legal configurations (pick one)

```jsonc
{}                                              // everything off: HBM path (unlike the old default, which was on)
{"cpu_offload": true}                           // offload to host memory
{"cpu_offload": true, "dp_shared_memory": true} // host offload + node-local DP-shared host table
```

### Illegal combination

```jsonc
{"dp_shared_memory": true}                       // error: shared requires cpu_offload
{"cpu_offload": false, "dp_shared_memory": true} // ditto
```

`dp_shared_memory=true` with `cpu_offload=false` raises `ValueError` at parse time.

## Constraints

- **TP**: only `TP ∈ {1, 2, 4, 8}`, and the **hash-column count must be divisible by TP**. The column count is `(max_ngram_size - 1) * n_heads` (V4.1 defaults `n_heads=8`, `max_ngram_size=4` => **24 columns**, divisible by 1/2/4/8). A non-dividing shard is rejected.
- **PP / DCP**: Engram currently requires `PP=1` and `DCP=1`.
- **Weight format**: safetensors / dummy only (`load_format ∈ {auto, safetensors, dummy}`). **int8-only boundary**: only the W8A8 (int8) table has a CPU decode path; an fp8/bf16 table requesting `cpu_offload` falls back to HBM, and requesting `dp_shared_memory` is a hard error.
- **DP sharing depends on**: a node-local gloo process group (HCCL has no object collectives) and `/dev/shm` capacity (prechecked against the aggregate footprint at startup).
- **PD disaggregation**: TODO -- once V4.1 Engram is wired into a P/D-disaggregated topology it will require `dp_shared_memory=true`; not enforced at runtime yet.

### Node identity and shared-memory prerequisites

Ranks within one DP shared-memory domain should report a consistent node identity. `hostname` is used only to group the prechecks (e.g. the DP-shared `/dev/shm` capacity estimate) and is **not by itself proof of sharing**: distinct physical nodes must be distinguishable, and the same physical machine with separate `/dev/shm` namespaces (e.g. containers with their own tmpfs mounts) cannot share. The real proof is a successful attach **and** the token read back from the segment tail matching the one the leader wrote (`_verify_segment_token`); the `/dev/shm` capacity check is likewise only a precheck. Note that per-rank `hostname` uniqueness is **not** required -- several ranks on one node sharing the same name is normal.

## Related implementation

- Parsing / validation: `xllm/python/models/engram_config.py`, `xllm/python/models/deepseek_v41.py` (`_validate_engram_config`)
- Offload path: `xllm/python/models/deepseek_v41_engram.py`
- DP-shared host table: `xllm/python/models/engram_shared_memory.py`
