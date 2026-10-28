---
title: "多模态支持"
sidebar:
  order: 60
---
本文档主要介绍xLLM推理引擎中多模态的支持进展，包括支持模型及模态类型，以及离在线接口等。

## 支持模型
- Qwen2.5-VL: 包括7B/32B/72B。
- Qwen3-VL: 包括2B/4B/8B/32B。
- Qwen3-VL-MoE: 包括A3B/A22B。
- MiniCPM-V-2_6: 7B。

## 模态类型
- 图片: 支持单图、多图的输入，以及图片+Prompt组合、纯文本Promot等输入方式。

## 服务级预处理参数

使用 `--mm_args` 传入按模态分组的 JSON 对象，参数交给对应的原生 processor 解释：

```bash
--mm_args='{"image":{"min_pixels":65536,"max_pixels":1048576}}'
```

`image`、`video`、`audio` 的参数相互隔离。当前接入以下图片参数；具体支持范围取决于模型使用的
processor，而不是框架中的模型名称白名单：

| 原生图片 processor | 参数 | 含义 |
| --- | --- | --- |
| Qwen2VLImageProcessor | `min_pixels`、`max_pixels`；或 `size.shortest_edge`、`size.longest_edge` | 图片面积预算，覆盖模型默认值；同一字段同时使用两种命名时，以 min/max_pixels 为准 |
| Glm4VImageProcessor | 同上 | 沿用 GLM 算法的时间维面积预算语义 |
| MiniCPMVImageProcessor | `max_slice_nums`、`scale_resolution` | 最大切片数和切片目标分辨率，同时应用于图片处理和 prompt 占位符 |
| KimiK25ImageProcessor | `in_patch_limit`、`patch_limit_on_one_side` | 输入 patch 预算和单边 patch 限制 |
| Mistral3ImageProcessor | `longest_edge` | 图片最长边限制 |

例如，Qwen 系列只覆盖最大面积时使用 `--mm_args='{"image":{"max_pixels":1048576}}'`，
MiniCPM 使用 `--mm_args='{"image":{"max_slice_nums":4,"scale_resolution":448}}'`。
未指定的字段继承模型配置与 processor 默认值。未设置、空字符串、`{}` 或空的模态对象均保持原配置。
参数在启动时解析并初始化，对服务期间的所有请求一致生效。

框架校验 JSON 结构和模态名称；字段类型、取值和组合由实际 processor 校验。
未被消费的字段在启动时报错，包括尚未接入参数接口的 video/audio processor，避免参数被静默忽略。
像素预算路径要求正整数、合并后的 `min_pixels <= max_pixels`，并至少能容纳一个对齐网格。
采用原生 token-budget resize 的配置暂不接受像素预算覆盖。

图片尺寸按模型的 patch/merge 规则对齐，`max_pixels` 不保证严格的视觉 token 上限。
极窄图片在紧预算下仍需保留非零的对齐尺寸，最终面积可能超过 `max_pixels`。
视觉 token 数由最终 `image_grid_thw` 计算。服务内多张图片分别应用相同参数。

CLI 中的 `--mm_args` 优先于服务配置文件。使用 `--config_json_file` 时，对应字段为 JSON 字符串：

```json
{"mm_args":"{\"image\":{\"max_pixels\":1048576}}"}
```

图片预处理统一使用 xLLM 原生实现，包括 Python 模型执行模式。
原有 `--image_max_tokens_num` 已移除，请改用 `--mm_args` 中的像素参数。
本入口不提供请求级覆盖。仅配置 `image` 时不会改变视频或音频行为。

### 接入新的原生 processor

复用现有 processor 的新模型自动获得相同参数支持。实现新的 processor 时，在构造函数中接收
`const ProcessorConfig&`，通过 `config.value<T>("image.参数名", 模型默认值)` 读取参数并校验算法约束。
装配流程创建 processor 时传入完整的服务覆盖参数，各 processor 自行读取需要的模态，构造完成后检查未消费字段；
无需修改 gflag、公共字段列表或模型分支。
旧的单参数构造函数仍支持空配置，提供非空但未消费的参数会明确报错。

`ProcessorConfig` 只在初始化期间使用，processor 保存解析后的数值。新参数如果也影响 prompt
占位符，应让相应的 prompt processor 读取相同字段，保持预处理与 token 展开一致。
例如 MiniCPM 的图片和 prompt processor 都读取 `image.max_slice_nums` 与 `image.scale_resolution`。
配置对象只提供参数，`create_processor` 负责创建处理器。

:::caution[注意事项]
- 目前多模态后端不支持prefix cache以及chunk prefill，正在支持中。
- 目前，xLLM统一基于JinJa渲染ChatTemplate，部署MiniCPM-V-2_6，模型目录需提供ChatTemplate文件。
- 图片支持Base64输入以及图片Url。
- 目前多模态模型主要支持了图片模态，视频、音频等模态正在推进中。

:::
