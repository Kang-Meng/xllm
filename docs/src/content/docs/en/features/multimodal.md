---
title: "Multimodal Support"
sidebar:
  order: 60
---
This document introduces the current multimodal support in the xLLM inference engine, including supported models, modality types, and offline and online interfaces.

## Supported Models
- Qwen2.5-VL: including 7B/32B/72B.
- Qwen3-VL: including 2B/4B/8B/32B.
- Qwen3-VL-MoE: including A3B/A22B.
- MiniCPM-V-2_6: 7B.

## Modality Types
- Images: supports single-image and multi-image inputs, image + prompt combinations, and text-only prompts.

## Service-Level Preprocessing Parameters

Pass a JSON object grouped by modality with `--mm_args`. Each native processor interprets its own parameters:

```bash
--mm_args='{"image":{"min_pixels":65536,"max_pixels":1048576}}'
```

The `image`, `video`, and `audio` objects are isolated. The following image parameters are currently
supported, depending on the selected processor rather than a framework list of model names:

| Native image processor | Parameters | Meaning |
| --- | --- | --- |
| Qwen2VLImageProcessor | `min_pixels`, `max_pixels`, or nested `size.shortest_edge`, `size.longest_edge` | Image area budgets; min/max_pixels take precedence over the corresponding size aliases |
| Glm4VImageProcessor | Same as above | Preserves the GLM algorithm's temporal area budget semantics |
| MiniCPMVImageProcessor | `max_slice_nums`, `scale_resolution` | Slice count and target resolution, shared with prompt placeholder generation |
| KimiK25ImageProcessor | `in_patch_limit`, `patch_limit_on_one_side` | Input patch budget and per-side patch limit |
| Mistral3ImageProcessor | `longest_edge` | Maximum image edge length |

For example, Qwen models can use `--mm_args='{"image":{"max_pixels":1048576}}'`, while MiniCPM can use
`--mm_args='{"image":{"max_slice_nums":4,"scale_resolution":448}}'`. Omitted fields inherit model settings
and processor defaults. An omitted flag, an empty string, `{}`, or an empty modality object preserves existing
settings. Parameters are resolved at startup and remain fixed for every request in the service.

The framework validates JSON structure and modality names. Each processor validates field types, values,
and combinations. Unconsumed parameters fail at startup, including parameters passed to video/audio
processors that have not adopted the configuration interface. Pixel budgets must be positive integers,
satisfy `min_pixels <= max_pixels` after merging, and fit at least one aligned grid cell.
Native token-budget resize configurations do not accept pixel-budget overrides yet.

Dimensions follow model patch/merge alignment, so `max_pixels` is not a strict visual token limit.
For very narrow images with tight budgets, keeping nonzero aligned dimensions can make the final area exceed `max_pixels`.
The actual token count follows `image_grid_thw`. The same settings apply independently to each image.

The CLI flag takes precedence over the service configuration file. In a file passed to
`--config_json_file`, use a JSON string:

```json
{"mm_args":"{\"image\":{\"max_pixels\":1048576}}"}
```

Image preprocessing uses xLLM's native implementation, including with Python model execution.
The previous `--image_max_tokens_num` flag has been removed; use pixel parameters in `--mm_args`.
This option does not provide request-level overrides. Setting only `image` does not change video or audio behavior.

### Adding a Native Processor

New models that reuse a processor automatically inherit its parameter support. A new processor can accept
`const ProcessorConfig&` in its constructor, read parameters with `config.value<T>("image.name", model_default)`,
and validate its algorithm's constraints. The assembly creates processors with the complete service overrides;
each processor reads the modalities it needs. Unconsumed fields are rejected after construction.
No new gflags, central field lists, or model-name branches
are needed. Existing one-argument constructors continue to work with empty configuration.

`ProcessorConfig` is used during initialization; processors retain resolved values. If a parameter also affects
prompt placeholders, the prompt processor should read the same fields to keep token expansion consistent.
For example, MiniCPM's image and prompt processors both read `image.max_slice_nums` and
`image.scale_resolution`. Configuration objects only provide parameters; `create_processor` handles construction.

:::caution[Notes]
- The multimodal backend does not currently support prefix cache or chunked prefill. Support is in progress.
- xLLM now renders ChatTemplate uniformly based on JinJa. When deploying MiniCPM-V-2_6, the model directory must provide a ChatTemplate file.
- Image inputs support Base64 data and image URLs.
- Multimodal models currently mainly support the image modality. Video, audio, and other modalities are in progress.

:::
