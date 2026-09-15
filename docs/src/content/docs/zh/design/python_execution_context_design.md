---
title: "Python 执行上下文设计"
sidebar:
  order: 5
---
<!--
Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

## 1. 背景

Python 模型接入新的执行特性时，经常需要模型 forward 参数之外的额外输入，例如：

- 根据当前 batch 生成的动态 Tensor；
- ACL Graph capture 和 replay 期间地址必须稳定的 Tensor；
- 多个 layer 可以共享的只读或临时 buffer；
- 每个 layer 必须独占的可写 buffer；
- 每 step 需要原地更新，但不应重新分配的状态。

如果直接在模型层或 runner 中处理这些需求，通常会产生两类问题。

第一类问题是模型感知执行模式。模型开始判断当前是 eager 还是 Graph Mode，并自行选择临时分配
或持久化分配。这样，同一份数学计算会出现多套控制流，模型正确性和执行引擎的地址生命周期
相互耦合。

第二类问题是 runner 感知具体功能。eager runner 和各类 graph runner 逐渐包含不同模型、算子
和功能模块的输入准备逻辑，最终成为大量特例分支的集合。

执行上下文机制的目标，是在二者之间建立清晰边界：

```text
runner 管理执行生命周期
        ↓
provider 准备具体功能需要的地址和内容
        ↓
ForwardContext 传递强类型上下文
        ↓
模型只消费输入并执行计算
```

## 2. 核心设计哲学

### 2.1 模型只感知计算，不感知地址来源

对模型而言，上游传入的是一个可以使用的 Tensor。该 Tensor 是 eager 临时申请的，还是由某个
graph entry 长期持有的固定地址，不应改变模型的计算逻辑。

因此，模型层不应：

- 查询当前是否处于 Graph Mode；
- 自行调用 graph buffer 管理接口；
- 根据 eager、warmup、capture 或 replay 选择不同数学路径；
- 推导某个 Tensor 应该跨 step 还是跨 layer 复用。

模型只负责按照约定读取输入、执行算子并返回输出。

### 2.2 地址的生产者负责生命周期

谁创建 Tensor，谁就应该决定：

- shape、dtype 和 device；
- 地址需要保持多久；
- 内容何时更新；
- 能否跨 layer 或跨 step 复用；
- 是只读共享，还是可写独占；
- 不满足约束时是否应在执行前拒绝或选择其他路径。

执行上下文 provider 是这些决策的唯一责任主体。模型只消费 provider 提供的地址，不反向猜测
地址的生命周期。

### 2.3 runner 只管理时机，不理解功能语义

runner 知道什么时候进入 eager forward，什么时候创建 graph entry，以及什么时候准备 replay。
它天然适合决定调用时机，但不应理解某个模型功能需要哪些 Tensor。

因此 runner 只调用统一生命周期接口：

- eager forward 前调用 `build_eager()`；
- graph entry 创建时调用 `allocate_graph()`；
- graph replay 前调用 `update_graph()`。

具体分配什么、更新什么以及如何复用，都由 provider 决定。

### 2.4 ForwardContext 只负责传递

`ForwardContext` 是一次模型 forward 的运行时载体。它通过
`execution_contexts: dict[type[object], object]` 保存不同功能提供的强类型上下文。

它不负责构造或解释上下文。模型通过类型获取自己需要的对象：

```python
context = get_execution_context(MyFeatureContext)
```

以类型作为 key，可以避免持续为 `ForwardContext` 增加功能专属字段，也能在重复注册或类型错误时
尽早失败。

## 3. 通用抽象

`xllm/python/model_executor/execution_context.py` 定义了统一协议：

```python
class ExecutionContextProvider(Protocol):
    context_type: type[object]

    def build_eager(
        self,
        input_ids: torch.Tensor,
        metadata: AttentionMetadata,
    ) -> object: ...

    def allocate_graph(
        self,
        token_capacity: int,
        device: torch.device,
        metadata: AttentionMetadata,
    ) -> object: ...

    def update_graph(
        self,
        context: object,
        metadata: AttentionMetadata,
        local_token_count: int,
    ) -> None: ...
```

三个接口分别对应三种不同的生命周期语义。

### 3.1 `build_eager()`

为当前 eager forward 构造上下文。Tensor shape 可以根据本次执行输入按需确定，地址不需要跨
forward 保持稳定。

provider 仍然可以在同一次 forward 内复用兼容 layer 的 scratch buffer，因为模型各层通常串行
执行。这里复用的是执行期临时存储，不是让模型感知显存管理策略。

### 3.2 `allocate_graph()`

在 graph entry 创建时执行一次，为一个固定 bucket 分配全部需要稳定地址的 Tensor。capture 和
后续 replay 都使用同一组地址。

该接口只负责建立 graph entry 的长期存储，不应混入具体请求每 step 才能确定的值。

### 3.3 `update_graph()`

在每次 replay 前更新动态内容。更新必须在已有 Tensor 上原地完成，不能替换 capture 时记录的
地址。

不是所有 Tensor 都需要 update。provider 应明确区分：

- 跨 step 不变的内容：分配时初始化一次；
- 每 step 变化、但所有 layer 相同的内容：每 step 更新一次并跨层共享；
- 每 layer 不同的内容：由 provider 管理对应 layer 的独立地址。

## 4. 完整执行流程

### 4.1 初始化阶段

模型中的功能模块通过静态 spec 声明执行需求。spec 只描述约束，不分配运行时 Tensor，例如：

```python
@dataclass(frozen=True)
class FeatureLayerSpec:
    layer_id: int
    hidden_size: int
    dtype: torch.dtype
    device: torch.device
```

`ModelExecutor` 扫描执行模型，根据这些 spec 创建对应 provider，再将 provider 绑定到 eager 和
graph runner。未启用该功能的模型不会创建 provider，也不会影响原有执行路径。

### 4.2 eager 阶段

```text
请求输入和 metadata
  -> runner 调用 provider.build_eager()
  -> provider 完成准入、分配和校验
  -> runner 调用 attention_backend.prepare()
  -> context 写入 ForwardContext
  -> 模型读取 context 并执行 forward
```

上下文构造应在可能进入 collective 的 `attention_backend.prepare()` 之前完成。这样，不支持的输入
可以在任何 rank 进入 collective 前统一失败，避免部分 rank 已经进入通信而其他 rank 提前退出。

### 4.3 Graph Mode 阶段

创建 graph entry 时：

```text
确定 graph bucket
  -> runner 调用 provider.allocate_graph()
  -> provider 分配稳定地址
  -> warmup / capture 使用该上下文
```

每次 replay 前：

```text
本 step metadata
  -> runner 调用 provider.update_graph()
  -> provider 原地更新动态内容
  -> graph 使用原地址 replay
```

模型在 warmup、capture 和 replay 中看到的是同一种上下文类型和同一套计算接口。模型不需要知道
当前 forward 属于哪个阶段。

## 5. Buffer 分类与复用原则

是否可以复用，不能只看 Tensor shape 相同，还需要同时考虑读写关系和生命周期。

| Buffer 类型 | eager | Graph Mode |
| --- | --- | --- |
| forward 内只读、各层内容相同 | 可以跨层共享 | 可以跨层共享 |
| forward 内可写、各层串行使用 | 可以跨层共享 | 需要根据 capture 的写依赖判断 |
| replay 前更新、各层内容相同 | 每次构造或写入一次 | 固定地址，每 step 原地更新一次 |
| layer 独占的可写输出 | 可在生命周期不重叠时复用 | 通常每层保持独立地址 |
| graph bucket 固定输入 | 不需要持久化 | 随 graph entry 持久化 |

provider 应按兼容签名组织共享，例如：

```text
(shape 语义, dtype, device, 读写属性, 生命周期)
```

只有签名和生命周期都兼容的 layer 才能共享 Tensor。是否复用可写 buffer 必须由 provider 明确
决定，不能由模型通过当前执行模式临时判断。

## 6. 模块职责

| 模块 | 应负责 | 不应负责 |
| --- | --- | --- |
| 模型或 layer | 声明静态需求、读取 context、执行数学计算 | 判断 eager/graph、管理持久化地址 |
| feature provider | 分配、复用、更新、校验和执行前准入 | 执行模型数学逻辑 |
| 通用 runner | 在正确生命周期调用所有 provider | 识别具体模型或功能的 Tensor 语义 |
| `ForwardContext` | 承载当前 forward 的强类型上下文 | 创建 context 或决定复用策略 |
| 调度与 batch 构建 | 提供当前 step 的执行视图 | 关心模型内部 buffer 的地址组织 |

这一边界可以概括为：

```text
调度层决定“这一步执行什么”
runner 决定“什么时候准备和执行”
provider 决定“使用哪些地址以及如何维护”
模型决定“拿这些输入做什么计算”
```

## 7. 接入新功能时的约束

新的执行功能若需要额外 Tensor，应优先通过 execution context 接入，并遵守以下规则：

1. 定义强类型 context，字段只包含模型真正需要消费的输入。
2. 定义静态 spec，使 provider 可以在执行前发现所有 layer 的要求。
3. 由 provider 统一校验不同 layer 的 topology、shape、dtype 和 device 是否兼容。
4. eager 使用当前执行所需的最小合理容量，不为了复用 graph 逻辑而无条件申请最大 bucket。
5. Graph Mode 在 entry 创建时分配稳定地址，replay 只原地更新动态内容。
6. provider 明确记录哪些 buffer 跨层共享、哪些每层独占，不能依赖隐含调用顺序。
7. 不在模型中调用 `in_acl_graph()` 或 `get_execution_buffer()` 决定地址来源。
8. 不向 `AttentionMetadata` 增加与 attention 无关的模型专属字段。
9. 所有不支持条件应在模型执行和 collective 开始前检查。
10. provider 不存在时，runner 的原有路径保持不变。

## 8. 为什么不直接使用钩子函数

执行上下文机制在调用形式上类似钩子，但它不是任意 callback 系统。provider 的生命周期接口是
固定且受约束的，返回值也必须以强类型 context 进入 `ForwardContext`。

这种限制有三点价值：

- runner 可以统一编排所有功能，而不允许 provider 任意改变主执行流程；
- 模型依赖的是明确的数据契约，而不是隐藏副作用；
- Graph Mode 的 allocate-once、update-in-place 语义可以被集中测试。

## 9. 长期维护价值

执行上下文机制建立了一条可复用的扩展路径：功能通过 provider 接入执行器，而不是继续向模型、
runner 或公共 metadata 添加特例。

它带来的长期收益包括：

- eager 和 Graph Mode 复用同一份模型计算代码，更容易保证数值一致；
- buffer 所有权、更新频率和复用策略有唯一落点，便于 review 和单元测试；
- runner 保持执行编排职责，不会逐渐演变为模型分支集合；
- 新模型和新功能可以独立提供 context，不需要修改其他功能；
- 新 graph backend 只需遵守统一生命周期，而不需要重新理解模型细节；
- 性能优化可以发生在 provider 内部，不破坏模型接口和数学语义。

最终目标不是让系统所有层都不知道 Graph Mode，而是让 Graph Mode 只被应该感知它的执行层
感知：runner 管理时机，provider 管理地址，模型保持纯计算。
