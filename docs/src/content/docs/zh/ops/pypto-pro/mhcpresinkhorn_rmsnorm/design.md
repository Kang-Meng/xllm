# mhcpresinkhorn_rmsnorm（v4-K + RmsNorm 融合）设计说明

本实现以一个 PyPTO-Pro MIX Kernel 融合生产链路的 HcPreInvRms → Cast → MatMulV3 → Cast → HcPreSinkhorn → RmsNorm。目标平台为 Ascend 950PR（A5），原设计环境为 CANN 9.2.0、PyPTO-Pro JIT。

## 1. 算子语义

令 `x` 的形状为 `[t,4,4096]`（bf16），`hc_fn` 为 `[24,16384]`（bf16）：

```text
r[t,1]           = rsqrt(mean(float(x)^2, dims=(hc,d)) + eps)
mixes[t,24]      = reshape(x,[t,16384]) @ hc_fn.T     # BF16×BF16→FP32
res[t,24]        = mixes * r
pre[t,4]         = sigmoid(res[:,0:4]*s0 + base[0:4]) + eps
post[t,4]        = sigmoid(res[:,4:8]*s1 + base[4:8]) * 2
comb_flag[t,4,4] = reshape(res[:,8:24]*s2 + base[8:24],[t,4,4])
y_bf16[t,4096]   = bf16(sum_m pre[m]*float(x[t,m,:]))
comb             = Sinkhorn(comb_flag, 稳定行 softmax 后加 eps、列归一化，再做 19 轮行列归一化)
rstd[t,1]        = rsqrt(mean(float(y_bf16)^2) + RMS_EPS)
y_norm[t,4096]   = bf16(float(y_bf16)*rstd*float(gamma))
```

RmsNorm 的平方和针对**已经舍入为 bf16 的中间结果**计算；公开输出 `y` 是归一化后的 `y_norm`，不单独返回原始 `y_bf16`。`comb` 在首轮进行稳定 softmax，后续按行、列归一化，详见[接口说明](api_reference.md)。

## 2. MIX 核与流水

使用 28 个 MIX 逻辑块（28 AIC + 56 AIV），每组 16 个 token 仅做一次 `sync_all(MIX)`：

```text
AIC：预加载权重分片 → 每组加载 x → 分 K 矩阵乘 → 写入部分结果 → MIX 同步
AIV：预加载辅助参数和 gamma → 并行计算 x 平方和 → 等待 MIX 同步
     → 跨 28 个部分结果归约 → 求 r 和门控
     → y 第一遍（8×512，保留 bf16 中间值并累加平方和）
     → rstd → y 第二遍（融合 gamma 的归一化）→ Sinkhorn → 写回
```

Cube 从 GM 直接读取 bf16 的 `x` 和 `hc_fn`，与 Vector 计算输入平方和并行。同步后，Cube 的下一 token 组矩阵乘可与 Vector 的当前组后处理重叠。中间 `y` 留在 UB，省去向 GM 写回再读取以及独立的 RmsNorm Kernel 启动。

## 3. Cube：K 维切分

`K=16384` 被分给 28 个 AIC：核 0–26 的起点为 `core*592`，各处理 592 个元素；核 27 处理末尾 400 个元素，`27*592+400=16384`。两个分片均为 16 的整数倍，分别使用 `[16,592]×[592,32]` 与 `[16,400]×[400,32]` 的 tile，无需补零或尾块。权重分片对同一核的各 token 组保持不变，预加载一次。28 份 FP32 部分乘积写入工作区 `pmix3[28, ceil(t/16)*16, 32]`，由 Vector 侧归约为 24 个有效门控通道。

Mat tile 不支持 `fillpad`；原设计亦指出 Cube 段无法借助所需 GM→L1 barrier 安全地实现补零方案，因此采用上述两种 tile 形状。

## 4. Vector：归约、门控与融合 RmsNorm

### 4.1 归约及门控

`_gates1_vf` 对 28 路矩阵乘部分结果和 32 路输入平方和分片分别求和；用 `scale24`、`pmul24`、`eps24` 模式向量及前缀掩码并行处理三类门控：`pre` 为 sigmoid 后加 `eps`，`post` 为 sigmoid 后乘 2，`comb_flag` 走线性分支。

### 4.2 两遍 y 数据流

| 阶段 | 实现 | 作用 |
| --- | --- | --- |
| 第一遍，8 个 512 维片段 | `_y_pass1_vf` | 展开 4 条 hc 行的 FMA，bf16 中间 `y` 保存在 UB，同时累计 `y` 的平方和 |
| rstd | `_rstd_vf` | 对平方和归约，计算 `rsqrt(mean(y_bf16²)+RMS_EPS)` 并写回 `[t,1]` fp32 |
| 第二遍，8 个 512 维片段 | `_y_norm_vf` | 从 UB 读取中间 `y` 和预加载的 `gamma`，生成并写回 bf16 `y_norm` |

`x` 子块使用双缓冲；加载下一片段可与上一片段的 VF 计算重叠。`gamma` 每个 AIV 预加载一次。

### 4.3 Sinkhorn

`_sinkhorn_vf` 将 4×4 门控矩阵展开为寄存器值。首轮先减去行最大值，做稳定行 softmax（其分母不加 `eps`），在结果上加 `eps`，再除以列和加 `eps`。后续 19 轮依次按行、列归一化，各自的分母均为对应的和加 `eps`；输出 `comb` 和 `post` 写回 GM。

## 5. 存储及平台约束

- UB 地址跨度为 `[0, 0x10100)`，即 65,792 B（64.25 KiB），小于 248 KiB 容量，覆盖输入平方和片段、中间 `y`、`gamma`、双缓冲 `x` 子块及归一化输出、门控归约和 Sinkhorn 临时值。该跨度包含双缓冲槽及 padding，部分小临时 tile 在不同阶段复用重叠存储；不能将其理解为独立 tile 大小之和或统一的最小槽位间距保证。
- L1 使用分别适配 K=592 和 K=400 的源 tile；L0B 最大占用约 63.5 KB（64 KB 上限）。
- ZN/Acc tile 的 N 必须按 16 宽内框对齐，24 个有效门控通道使用 N=32。
- Vec tile 仅在 Vector 段使用；`.next()` 在 Kernel 主体轮换槽位，不在 VF 中调用。
- 跨核同步使用 `sync_all(MIX)`，而非原设计中发现存在计数器共享问题的 INTRA_BLOCK 事件。

`post` 转置在整个 `pl.transpose` 调用期间必须保持源、目的存储不重叠，不能用仍在读取的源矩阵行作为目的缓冲区。

| Tile | 物理形状及类型 | 字节数 | UB 字节范围（半开区间） |
|------|----------------|--------|------------------------|
| `PST_UB` | `[4,8]` fp32 | 128 | `[0x08540, 0x085C0)` |
| `PSTT_UB` | `[1,8]` fp32 | 32 | `[0x08600, 0x08620)` |
| `PRE_UB` | `[1,8]` fp32 | 32 | `[0x08640, 0x08660)` |

源矩阵四行的起始地址分别为 `0x08540`、`0x08560`、`0x08580`、`0x085A0`。下一块 `XSUB_UB` 从 `0x09000` 开始，因此总 UB 地址跨度不变。
两遍 y 计算每段显式使用 128 元素的 bf16 掩码，最后一段从 3968 开始，到 4096 结束（不含）。保留行及片段展开是为了维持八个独立的偶/奇 FMA 累加器及其累加顺序，并不表示 VF 普遍不支持循环。Sinkhorn 对每个 token 始终只启用一个 fp32 lane。

## 6. 演进与历史测量

早期实现从纯 Vector、Cube+Vector、流水版本演进为 K 分片与双缓冲，再融合 RmsNorm。原设计记录的融合版本在 `t=1` / `t=13` 时约为 7.98 / 10.44 µs，属于**历史测量数据，未在当前环境复测**；不能将其当作本仓库的性能保证或当前测试结果。

参数及返回值参见[中文版接口说明](api_reference.md)。

## 7. 可运行示例（Demo）

完整可运行脚本位于本目录下的 `mhcpresinkhorn_rmsnorm_demo.py`，其内容与下方代码一致。脚本构造随机输入、调用算子，并用 PyTorch 参考实现按各输出各自的容差比较四个输出。运行需要可用的 Ascend 950PR 环境（已安装 CANN 与 PyPTO-Pro），并将 `PTO_TILE_LIB_CODE_PATH` 指向本地 `pto-isa` 检出目录；首次调用会触发 JIT 编译。

```python
#!/usr/bin/env python3
# coding: utf-8
"""mhcpresinkhorn_rmsnorm（v4-K + RmsNorm 融合）可运行示例。"""

import os
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import torch
import torch_npu  # noqa: F401

# `pypto-pro` 目录名含 `-` 且不是 Python 包，因此按文件路径加载 kernel 模块。
_KERNEL_PATH = (Path(__file__).resolve().parents[8]
                / "xllm/python/kernels_npu/pypto-pro/mhcpresinkhorn_rmsnorm.py")
_spec = spec_from_file_location("mhcpresinkhorn_rmsnorm", _KERNEL_PATH)
mhcpresinkhorn_rmsnorm = module_from_spec(_spec)
_spec.loader.exec_module(mhcpresinkhorn_rmsnorm)
mhcpresinkhorn_rmsnorm = mhcpresinkhorn_rmsnorm.mhcpresinkhorn_rmsnorm

HC, D, MIX_HC, HC_D = 4, 4096, 24, 16384
EPS, RMS_EPS = 1e-6, 1e-6


def reference(x, hc_fn, hc_scale, hc_base, gamma):
    """PyTorch 参考链路：融合门控（fp32 计算链）+ RmsNorm。"""
    xf = x.float().reshape(x.shape[0], HC_D)
    r = torch.rsqrt((xf * xf).mean(dim=-1, keepdim=True) + EPS)
    res = (xf @ hc_fn.float().T) * r
    pre = torch.sigmoid(res[:, 0:HC] * hc_scale[0] + hc_base[0:HC]) + EPS
    post = torch.sigmoid(res[:, HC:2 * HC] * hc_scale[1]
                         + hc_base[HC:2 * HC]) * 2.0
    comb_flag = (res[:, 2 * HC:].reshape(-1, HC, HC) * hc_scale[2]
                 + hc_base[2 * HC:].reshape(1, HC, HC))
    y = (x.float() * pre.reshape(-1, HC, 1)).sum(dim=1).to(torch.bfloat16)
    t0 = torch.softmax(comb_flag, dim=-1) + EPS
    comb = t0 / (t0.sum(dim=-2, keepdim=True) + EPS)
    for _ in range(19):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + EPS)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + EPS)
    yf = y.float()
    rstd = torch.rsqrt((yf * yf).mean(dim=-1, keepdim=True) + RMS_EPS)
    y_norm = (yf * rstd * gamma.float()).to(torch.bfloat16)
    return y_norm, rstd, post, comb


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    dev = f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', '0'))}"
    torch.npu.set_device(dev)

    # ---- 构造输入（随机数据，仅用于演示）----
    g = torch.Generator(device="cpu").manual_seed(42)
    x = torch.empty((t, HC, D), dtype=torch.bfloat16).uniform_(-1, 1, generator=g)
    hc_fn = torch.empty((MIX_HC, HC_D), dtype=torch.bfloat16).uniform_(
        -1, 1, generator=g)
    hc_scale = torch.empty((3,), dtype=torch.float32).uniform_(-1, 1, generator=g)
    hc_base = torch.empty((MIX_HC,), dtype=torch.float32).uniform_(-1, 1, generator=g)
    gamma = torch.empty((D,), dtype=torch.bfloat16).uniform_(0.5, 1.5, generator=g)
    x, hc_fn = x.to(dev), hc_fn.to(dev)
    hc_scale, hc_base, gamma = (hc_scale.to(dev), hc_base.to(dev), gamma.to(dev))

    # ---- 调用算子（首次调用包含 JIT 编译）----
    y, rstd, post, comb = mhcpresinkhorn_rmsnorm(x, hc_fn, hc_scale, hc_base, gamma)
    torch.npu.synchronize()
    print(f"t={t} | y_norm {tuple(y.shape)}/{y.dtype} | rstd {tuple(rstd.shape)}"
          f" | post {tuple(post.shape)} | comb {tuple(comb.shape)}")

    # ---- 按各输出各自的容差与参考链路比较 ----
    yr, rr, pr, cr = reference(x.cpu(), hc_fn.cpu(), hc_scale.cpu(),
                               hc_base.cpu(), gamma.cpu())
    checks = (("y_norm", y.float().cpu(), yr.float(), 1e-2, 1e-2),
              ("rstd", rstd.cpu(), rr, 1e-4, 5e-3),
              ("post", post.cpu(), pr, 2.5e-5, 5e-3),
              ("comb", comb.cpu(), cr, 2.5e-5, 5e-3))
    check_results = []
    for name, got, ref, atol, rtol in checks:
        ok = torch.allclose(got, ref, atol=atol, rtol=rtol)
        check_results.append(ok)
        diff = (got - ref).abs().max().item()
        print(f"  {name:6s}: {'PASS' if ok else 'FAIL'}  max_abs={diff:.3e}")
    print("[DEMO_" + ("PASS" if all(check_results) else "FAIL") + "]")


if __name__ == "__main__":
    main()
```

配置好昇腾环境后，从 xllm 仓库根目录执行：

```bash
TILE_FWK_DEVICE_ID=0 PTO_TILE_LIB_CODE_PATH=/path/to/pto-isa \
  python3 docs/src/content/docs/zh/ops/pypto-pro/mhcpresinkhorn_rmsnorm/mhcpresinkhorn_rmsnorm_demo.py 3
```

末尾参数为可选的 `t`（默认 `3`）。
