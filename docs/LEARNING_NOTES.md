# SLOServe 学习笔记

这份文件按执行顺序记录项目学习过程。每完成一步，都追加以下内容：要解决的问题、为什么需要、
关键命令或代码、成功标志、失败诊断、实际结果和下一步。只记录已经观察到的事实，不把 smoke
check 写成性能实验，也不虚构性能提升。

## 2026-08-26 — Week 1 第 1 步：Ubuntu 环境核验

### 要解决的问题

确认本机是否具备建立单 GPU vLLM 开发环境的基础条件，并把正式仓库从 NTFS 迁移到 ext4。

### 为什么需要

vLLM 依赖 Linux、NVIDIA 驱动、CUDA 可见性和可用 GPU。Python 虚拟环境放在 NTFS 上还可能
遇到权限、符号链接、大小写和大量小文件性能问题，所以正式环境必须先落到原生 Linux 文件系统。

### 关键命令与概念

- `uname`、`findmnt`、`df`：分别检查内核、挂载文件系统和磁盘空间。
- `nvidia-smi`：检查驱动、GPU、显存和驱动支持的 CUDA API 上限。
- `nvcc --version`：检查本机 CUDA toolkit 编译器版本。它和 `nvidia-smi` 显示的 CUDA 值不是
  同一个概念。
- `uv python find 3.11.14`：定位项目要求的受管 Python。
- `rsync -a`：把项目文件和 Git 元数据复制到 ext4；`rsync -acn --delete` 用校验和做只读验收。
- `uv lock --check`、Ruff 和 pytest：确认开发锁文件、代码质量与现有行为。

### 成功标志

- Ubuntu 能识别一张 RTX 4080 Laptop GPU，计算能力 8.9，显存 12,282 MiB。
- 正式工作区位于 `/home/jiale/Desktop/SLOServe`，文件系统为 ext4。
- 项目使用 CPython 3.11.14，锁文件、Ruff、格式检查、CLI 和 8 个当时已有的测试通过。

### 失败与诊断

- 初次禁止自动下载时，`uv python find 3.11.14` 找不到解释器，说明缺的是项目 Python，不是系统
  Python 损坏。随后 `uv lock --check` 按 `.python-version` 下载了受管 CPython 3.11.14。
- 原仓库位于 `ntfs3`，硬件条件虽然满足，但不适合作为正式虚拟环境位置；复制到 ext4 后再创建
  `.venv`。

### 实际结论与下一步

硬件、驱动、CUDA 可见性、Python 和 ext4 工作区满足继续安装 vLLM 的前置条件。此时没有安装
PyTorch/vLLM，也没有运行 GPU 服务或性能实验。下一步是建立独立服务环境并固定版本。

## 2026-08-26 — Week 1 第 2 步：固定软件与模型版本

### 要解决的问题

建立与 CPU 开发环境隔离的 vLLM 服务环境，并把 PyTorch、CUDA 用户态、模型和 tokenizer 固定到
可重建版本。

### 为什么需要

vLLM wheel 与具体 PyTorch/CUDA 二进制组合相关。使用浮动版本或模型 `main` 会让后续实验无法保证
使用相同软件和权重，实验结果也就难以复现。

### 关键命令与概念

- `uv venv --python 3.11.14 --seed .venv-vllm`：创建独立服务环境。
- `uv pip install ... vllm==0.27.1 torch==2.13.0 --torch-backend=cu130`：固定 vLLM、PyTorch 和
  CUDA 13.0 wheel 后端。
- `uv pip check`：检查已安装依赖是否存在冲突或缺失。
- `torch.cuda.is_available()` 和小矩阵乘法：验证 PyTorch 能实际调用 GPU。这是安装 smoke check，
  不是 benchmark。
- `hf download Qwen/Qwen3-0.6B --revision <commit>`：按不可变 commit 下载模型，而不是使用
  浮动的 `main`。
- `sha256sum model.safetensors`：验证权重文件内容与仓库元数据一致。
- `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`：强制从本地快照加载配置和 tokenizer，证明缓存完整。

### 成功标志

- `.venv-vllm` 使用 CPython 3.11.14、vLLM 0.27.1 和 PyTorch 2.13.0+cu130。
- PyTorch 识别 RTX 4080，CUDA runtime 报告 13.0，小矩阵乘法结果正确。
- 模型与 tokenizer 都固定到 commit
  `c1899de289a04d12100db370d81485cdf75e47ca`。
- `model.safetensors` 为 1,503,300,328 bytes，SHA256 为
  `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`。
- 离线加载识别 `Qwen3ForCausalLM`，tokenizer 可编码文本，safetensors 中可读取 311 个张量条目。

### 失败与诊断

- Hugging Face Python API 初次失败，错误为 `Unknown scheme for proxy URL`。原因是环境中的
  `ALL_PROXY=socks://...` 不被当前 `httpx` 接受；只对 Hugging Face 命令取消 `ALL_PROXY` 和
  `all_proxy` 后，保留 HTTP/HTTPS 代理即可访问。
- Xet 下载期间发生过 TLS 提前关闭。日志随后持续出现成功的 HTTP 206 分段响应，缓存也持续增长，
  因此判断为可恢复网络抖动，不是文件损坏；最终命令退出码为 0，权重哈希匹配。

### 实际结论与下一步

软件和模型版本已固定并通过安装、依赖、CUDA、文件哈希和离线加载检查。尚未启动真实 vLLM 模型
服务，也没有性能结果。下一步是 Week 1 第 3 步：用保守的单 GPU 参数启动 OpenAI-compatible
server，检查 `/v1/models`，再完成一个非流式和一个流式请求。

## 2026-08-27 — Week 1 第 3 步：跑通真实 vLLM 流式服务

### 要解决的问题

用一条版本化命令，以保守的单 GPU 参数启动真实的 vLLM OpenAI-compatible server，并功能性验证
`/v1/models`、一次非流式请求和一次流式请求。只做 smoke，不测性能。

### 为什么需要

后续的准入、排队和路由层（MVP）都建立在一个真实、可复现的 vLLM 后端之上。如果启动参数靠手敲、
或服务地址与客户端配置不一致，实验就无法复现。因此启动命令必须由版本化配置推导，而不是散落在
命令行历史里。

### 关键命令与概念

- 新增类型化 `ServerConfig` 与 `configs/base.yaml` 的 `server` 段，把 host/port/显存比例/最大
  上下文/最大序列数/dtype/enforce-eager 固化为配置。校验器强制 `server` 地址与
  `backend.base_url` 一致，且 `server.max_num_seqs >= router.max_in_flight`，让外部路由器（而非
  引擎）成为并发瓶颈。
- `sloserve serve-command --config configs/base.yaml`：从配置生成确定性的 `vllm serve` 命令。
  `scripts/serve.sh` 用它启动服务，因此没有任何引擎参数是手敲的。
- `--enforce-eager` 跳过 CUDA graph 捕获，`--gpu-memory-utilization 0.5` 限制显存，均为保守起步。
- smoke 客户端 `scripts/vllm_smoke.py` 只用标准库 `urllib`，分别打 `/v1/models`、非流式和流式
  `chat/completions`，把原始响应写入 `results/raw/week1-step3-smoke/`。流式用 SSE，逐行读取，
  确认首个 chunk 与 `data: [DONE]` 结束事件。

### 成功标志

- `scripts/serve.sh configs/base.yaml` 启动后 `~19s` 内 `/v1/models` 返回 `Qwen/Qwen3-0.6B`。
- 引擎报告：可用 KV cache 4.47 GiB，KV cache 41,808 tokens，单请求 4,096 tokens 的最大并发
  10.21x；加载后 `nvidia-smi` 显示 6,136 MiB 占用、空闲 0% 利用率。
- 非流式响应非空（`usage`: prompt 15 / completion 16 / total 31 tokens）；流式可见首个 role
  delta chunk、内容 chunk 与 `[DONE]`。
- 停止服务后无 `.venv-vllm` 的 vLLM 进程，端点拒绝连接，GPU 回到 14 MiB、0%。

### 失败与诊断

- 第一次启动 EngineCore 崩溃，根因不是显存（KV cache 充足），而是
  `TypeError: type 'array.array' is not subscriptable`。`flashinfer==0.6.16.post3`（vLLM 硬依赖）
  的 `comm/fd_exchange.py` 在模块级注解里写了 `array.array[int]`，而 `array.array` 到 CPython
  3.12 才可下标；该文件没有 `from __future__ import annotations`，注解被立即求值。vLLM 在
  warm-up 阶段无条件导入该模块。修复：`scripts/patch_flashinfer_py311.py` 幂等地在该文件插入
  future 导入（保持精确的固定依赖集，`uv pip check` 不受影响），而不是卸载依赖或改动行为。
- 修好导入后再次崩溃，根因 `FileNotFoundError: 'ninja'`。vLLM 采样器走 flashinfer 的 top-k/top-p
  路径，会在首次使用时用 `ninja`+`nvcc` 即时编译 CUDA kernel，而离线固定环境没有这条工具链。
  修复：`scripts/serve.sh` 导出 `VLLM_USE_FLASHINFER_SAMPLER=0`，改用 vLLM 原生 PyTorch 采样器，
  无需运行时编译；日志确认 "FlashInfer top-p/top-k sampling disabled"。
- 操作教训：`pkill -f "vllm serve"` 会匹配到执行该命令的 shell 自身命令行（其中含该字符串），
  导致自杀式退出（exit 144）。改用 `pkill -f '[.]venv-vllm/bin/vllm'`，方括号让匹配串本身不出现
  在命令行里，避免自匹配。

### 实际结论与下一步

真实 vLLM 单 GPU 服务已可用一条版本化命令启动，`/v1/models`、非流式与流式请求均功能正常，原始
证据已落盘。两个 Python 3.11 环境阻塞已用可复现方式修复并记入 `docs/ENVIRONMENT.md`。本步只是
管线功能验证，未产生任何吞吐、TTFT、TPOT、延迟、SLO 或持续利用率数据。下一步是 Week 1 第 4 步：
实现可测试的异步负载生成器（固定速率到达、固定种子、假后端可控测试）。

## 2026-08-27 — Week 1 第 4 步：异步负载生成器最小切片

### 要解决的问题

在不连接真实网络、不提前定义指标事件 schema 的前提下，完成一个可测试的异步负载生成最小切片：
确定性生成请求、固定速率到达、经 FCFS 公共策略接口异步分发，并覆盖并发、错误、超时和取消路径。

### 为什么需要

真实 vLLM smoke 已跑通，但若请求内容、到达时间和异步故障行为不可复现，后续 FCFS 基线和策略对比
就无法区分调度差异与负载随机性。先用纯内存假后端验证边界，还能在引入 HTTP 流解析和指标采集前
隔离 asyncio 并发、超时与取消清理问题。

### 关键命令与概念

- `FixedRateArrivalSchedule.arrival_times()` 只负责时间序列，按 `i / request_rate_rps` 生成；
  `generate_requests()` 只消费到达计划并生成 `RequestEnvelope`，两者通过 `ArrivalSchedule` 协议解耦。
- `random.Random(workload.random_seed)` 使用局部 RNG，按 `interactive_fraction` 选择类别，再从对应
  `RequestProfileConfig` 的闭区间采样输入和最大输出 token；deadline 为
  `arrival_time_s + end_to_end_slo_ms / 1000`。
- 预热采用“额外前置”语义：总生成数为 `warmup_requests + total_requests`。预热占用从 0 开始的
  sequence ID 并消耗同一 RNG 流，正式请求从 `sequence_id == warmup_requests` 开始；request ID 由
  sequence ID 确定性构造。
- `AsyncRequestBackend` 是未来 HTTP/流解析实现的异步边界；当前 `FakeBackend` 按请求配置成功、抛错、
  延迟或主动取消，且只在内存中记录在途数与启动顺序。
- `AsyncDispatcher` 通过 `SchedulingPolicy.order()` 使用 `FcfsPolicy`，按相对到达时间等待，以
  `router.max_in_flight` 构造 semaphore，并用 `backend.request_timeout_s` 包裹单请求调用。每个请求只
  返回基本标识、success/error/timeout/cancelled 状态和可选错误文本，不包含指标时间戳。
- 聚焦测试命令：
  `UV_CACHE_DIR=/tmp/sloserve-uv-cache uv run pytest tests/test_workload_arrivals.py tests/test_workload_generator.py tests/test_workload_dispatcher.py -q`。

### 成功标志

- 同一个 `WorkloadConfig` 重复生成的 `RequestEnvelope` 元组逐字段相等；类别、token 范围、到达时间、
  deadline、request ID 和 sequence ID 都可复现。
- 11 个 workload 聚焦测试通过；并发峰值不超过配置上限，success/error/timeout/cancelled 均有断言。
- 取消整个分发任务会继续向调用者传播 `CancelledError`；在途后端任务被取消并等待清理，随后同一
  dispatcher 仍能成功分发请求，未观察到容量泄漏。
- `uv lock --check`、Ruff lint、Ruff format 和全量 26 个 pytest 测试通过。

### 失败与诊断

- 首次执行聚焦测试时，`uv` 尝试在只读的 `/home/jiale/.cache/uv` 创建临时锁文件，命令以退出码 2
  失败，尚未进入 pytest。把本轮缓存显式设为可写的
  `UV_CACHE_DIR=/tmp/sloserve-uv-cache` 后，测试正常运行。
- 首次局部 Ruff 检查发现 `workload/__init__.py` 原有包 docstring 与新增 docstring 相邻，导致新增
  imports 被判为 E402；合并为单个模块 docstring 后 lint 通过。随后 format check 指出两个可机械
  折叠的函数/生成式换行，用 Ruff 格式化后通过。

### 实际结论与下一步

固定到达的确定性生成与纯内存异步分发最小切片已经可执行和可测试，没有修改现有配置 schema，
也没有接入真实 HTTP、指标事件或性能实验。Poisson 与 burst 当前会清晰抛出
`NotImplementedError`；下一项负载生成器扩展是分别实现带固定种子的 Poisson 到达和版本化 burst
参数。在 Week 1 主线中，下一步是第 5 步：先定义原始指标事件与结果格式，再把真实 HTTP 流解析接到
现有 `AsyncRequestBackend` 边界。

## 2026-08-27 — Week 1 第 5 步：原始指标事件与离线计算最小切片

### 要解决的问题

在不连接真实 HTTP/vLLM、不修改 dispatcher/backend 的条件下，建立可重放的请求级事实记录和纯离线
指标计算，使分位数、SLO、终态计数、最长等待、吞吐与公平性都能从保存的原始文件重算。

### 为什么需要

如果只保存聚合结果，公式错误、分位数口径变化或失败请求被遗漏后无法审计。先固定请求事件 schema、
JSONL 事实源和公平性公式，能让第 6、7 步接线时只负责产生真实事件，而不会边跑实验边改变统计定义。

### 概念背景（LLM 服务度量原理，复习用）

为什么度量是本项目的核心：SLOServe 是 vLLM 前面的准入/排队/路由层，它唯一能控制的是“哪个请求何时
进入 vLLM”。它凭什么“更好”，只能用数字证明——同样负载下让更多请求满足 SLO。所以度量不是附属品，
它就是评判标准本身，也是 AGENTS “数字必须能从原始数据重算、不得编造” 的由来。

一个请求的一生（五个时间戳）：
`arrival`（请求产生）→ `enqueue`（进入准入队列）→ `dispatch`（被放进 vLLM）→ `first_token`
（收到第一个 token）→ `completion`（收到最后一个 token；失败则为终态时刻）。核心洞察是能把延迟拆开：

```
端到端延迟 = 排队等待(dispatch − arrival) + 服务时间(completion − dispatch)
```

排队等待是路由器造成的，服务时间是 vLLM 造成的。好的调度会主动让低优先级请求多排队以保护高优先级
请求的 SLO——必须能分别测出两半，才能讲清路由器做了什么。

四个核心指标：
- TTFT（首 token 时间）= `first_token − arrival`：交互式请求最敏感（config 里 `interactive.ttft_slo_ms`）。
- TPOT（每输出 token 时间）= `(completion − first_token) / (output_tokens − 1)`：出字流畅度；分母减 1
  是因为第一个 token 已计入 TTFT。仅 success 且 `output_tokens ≥ 2` 才有定义。
- 端到端延迟 = `completion − arrival`：受 `end_to_end_slo_ms` 约束。
- 吞吐量（请求/秒或 output token/秒）：与延迟通常此消彼长——塞得越满吞吐越高、但每个请求排队越久。

为什么用 P50/P95/P99 而不是平均值：LLM 延迟分布重尾，少数请求会被长请求阻塞而慢很多，平均值会
掩盖尾部。SLO 通常写成分位数或“99% 请求满足 deadline”。SLO 达标 = success 且 TTFT、端到端均不超过
该类阈值；失败/超时/取消一律算未达标。

为什么 JSONL 是事实源、CSV 只是派生：JSONL 逐请求保存原始字段（五时间戳、token、状态、策略、
config 哈希、重复编号、环境版本），所有聚合都从它重算，才能被审计和换角度重分析；绝不能只存聚合。

方法论三件套：预热（头几个请求受 CUDA 图/缓存/JIT 冷启动影响，不计入统计——即第 3 步的
ninja/flashinfer 首次开销）、固定随机种子（同 seed 同请求序列，保证策略差异来自调度而非随机负载）、
≥3 次重复（测量有噪声）；失败请求也要记进 JSONL，不能只留“幸存者”。

排队直觉——Little's Law：`L = λ × W`（在系统请求数 = 到达率 × 平均停留时间）。`max_in_flight` 限制 L；
若到达率超过消化能力，W（延迟）会无限增长、队列爆炸——这正是第 6 步要做有界队列（宁可拒绝也不让
延迟无限恶化）的原因。

### 关键命令与概念

- `RequestRecord` 是不可变 dataclass，直接复用 `RequestClass`、`DispatchStatus` 和
  `SchedulerPolicyName`；`from_envelope()` 从公共 `RequestEnvelope` 加终态时间戳构造，不依赖实时
  dispatcher 改动。
- 五个时间戳和所有公式先写入 `docs/METRICS.md`。分位数采用 nearest-rank；成功且至少 2 个输出
  token 才进入 TPOT；error、timeout、cancelled 一律算 SLO 未达标。
- `experiment_config_hash()` 对 `ExperimentConfig.model_dump(mode="json")` 做字段排序、紧凑 JSON
  序列化和 SHA-256；环境版本只接受调用方给出的字符串，本步测试使用显式占位值。
- JSONL 与 CSV 共用同一字段映射；`write_request_records()` 只从 `MetricsConfig` 取得输出目录与
  格式开关。CSV 是派生格式，JSONL 是事实源。
- 聚焦测试命令：
  `UV_CACHE_DIR=/tmp/sloserve-uv-cache uv run pytest tests/test_metrics_records.py tests/test_analysis_metrics.py -q`。
- 完整检查依次为 `uv lock --check`、`uv run ruff check .`、`uv run ruff format --check .` 和
  `uv run pytest`，本机均显式使用可写的 `/tmp/sloserve-uv-cache`。

### 成功标志

- JSONL 和 CSV 读回的完整类型化记录与写入前逐字段相等，统一入口正确执行配置中的目录与开关。
- 6 条手算记录从 JSONL 读回后，nearest-rank 端到端 P95 为 5 秒，总体 SLO 达标率为 `2/6`，
  最长等待 1.5 秒，测试用 output-token 吞吐 1.125 token/s，分类别达标率为 0.25/0.5、差距 0.25、
  Jain 指数 0.9；这些只是算术 fixture，不是服务实测。
- 空输入、全失败、单 token 无 TPOT、SLO 等于阈值和零墙钟窗口都有明确断言。9 个聚焦测试、
  锁文件、Ruff lint/format 和全量 35 个测试全部通过。

### 失败与诊断

- 聚焦测试首次有 1 个失败：Python 浮点差值 `2.2 - 1.0` 是 `1.2000000000000002`，而断言使用了
  精确的 `1.2`。指标公式无需人为舍入；把手算断言改为 `pytest.approx` 后 9 个测试通过。
- 首次 Ruff lint 报告两行 104 字符，换行后通过。首次 format check 发现分析模块未采用 Ruff 的
  标准折叠形式；只格式化该新增模块并重新执行 lint/format 后通过。

### 实际结论

Week 1 第 5 步的离线事实源和指标计算切片已经可执行。现有 `MetricsConfig` 已包含本步所需的输出目录、
JSONL/CSV 开关，不需要扩展 schema；`gpu_sample_interval_s` 在本步未使用。没有生成或声称任何真实
延迟、吞吐、公平性或 GPU 结果。

### 下一步

进入 Week 1 第 6 步的最小集成：在外部 FCFS 准入/有界队列边界产生 arrival、enqueue、dispatch
和 terminal 生命周期事件，继续用假后端验证顺序、超时、取消和容量释放。真实 HTTP 流的
first-token 解析、完整 `RequestRecord` 接线、运行环境元数据填充和 GPU 采样仍留到第 7 步端到端
smoke benchmark。

## 2026-08-28 — Week 1 第 6 步：外部 FCFS 有界准入队列最小切片

### 要解决的问题

在不连接真实 HTTP/vLLM 的条件下，实现位于 vLLM OpenAI-compatible server 前面的外部有界准入
队列：严格 FCFS 选择、限制 in-flight 数、等待队列满时拒绝，并让排队和后端执行共享一个总超时。

### 为什么需要

原有 `AsyncDispatcher` 可以验证固定到达和并发，但会在取得并发槽之前等待，不能表达独立的有界等待
队列或立即拒绝。若外部到达率长期超过后端消化能力，无界排队会让内存和等待时间持续增长；超时或
取消若没有完整清理，还会永久缩小可用并发容量。

### 关键命令与概念

- `AdmissionQueue` 直接读取 `RouterConfig.queue_capacity`、`RouterConfig.max_in_flight` 和
  `BackendConfig.request_timeout_s`，没有扩展配置 schema。
- 提交、队列排序、槽位预留与终态迁移由同一把 `asyncio` 状态锁串行化。空闲槽存在时立即从等待项中
  调用 `SchedulingPolicy.order()`；本步默认实现为 `FcfsPolicy`，key 是
  `(arrival_time_s, sequence_id)`。
- in-flight 计数是在锁内对活动条目状态的等价 semaphore。后端 runner 的 `finally` 完成终态、释放
  活动条目并继续泵送队列，success/error/timeout/cancelled 都走同一释放路径；rejected 从未占用资源。
- 每个接收项有一个总超时监督任务，deadline 为 `enqueue_time + request_timeout_s`。排队超时直接从
  等待列表移除；in-flight 超时设置强制终态并取消 backend task，由 runner 等待取消完成后释放槽位。
- `submit()` 用 `asyncio.shield` 保护内部结果 future。调用方取消时，排队项被移除或 in-flight backend
  被取消并等待，然后继续向调用方传播 `CancelledError`；`aclose()` 则把受影响项解析为
  `CANCELLED` 并等待全部内部 task。
- 聚焦测试命令：
  `UV_CACHE_DIR=/tmp/sloserve-uv-cache uv run pytest tests/test_admission_queue.py -q`。完整检查依次为
  `uv lock --check`、`uv run ruff check .`、`uv run ruff format --check .`、`uv run pytest`，本机均
  使用可写的 `/tmp/sloserve-uv-cache`。

### 成功标志

- 9 个纯内存测试确认相同 arrival 时按 sequence ID 破同序、最大并发不超过 2，并可同时接收
  `max_in_flight + queue_capacity` 个请求；再多一个等待项立即得到 `REJECTED`，且没有进入后端。
- 注入的手动 monotonic clock/sleep 可分别唤醒指定请求的超时，不依赖真实时间，确定性验证排队超时的
  `dispatch_time_s is None` 和 in-flight 超时的后端取消。
- success、error、backend 自取消、排队/in-flight 的调用方取消、timeout、rejected 混合路径后，后续
  请求仍为 success；关闭后没有活动 fake backend 或以 `sloserve-admission-` 命名的悬挂 task。
- 锁文件解析 14 个 package，Ruff lint 通过，43 个文件格式检查通过，全量 44 个 pytest 测试通过。

### 失败与诊断

- 初版采用固定 worker 消费等待列表。复核边界时发现，事件循环在 worker 被唤醒前可能连续执行多个
  submit，使本应进入空闲 in-flight 槽的请求短暂占用等待容量。改为在提交所持的同一状态锁内立即预留
  空闲槽并创建 backend runner 后，系统可精确容纳 `max_in_flight + queue_capacity`，再多才拒绝。
- 初次局部 `ruff format --check` 报告 `admission.py` 两处可折叠格式；仅对新增文件运行 Ruff formatter
  后，聚焦测试与全量格式检查通过。

### 实际结论

Week 1 第 6 步的纯内存外部准入边界已经可执行。`DispatchStatus` 只追加了
`REJECTED = "rejected"`，没有改 `AsyncDispatcher` 逻辑或 analysis 计数器。结果对象包含请求标识、
终态和 enqueue/dispatch/completion 时间，可作为后续 `RequestRecord` 接线输入，但本步没有接线。
本步只决定请求何时送到外部 backend，不是也没有声称修改 vLLM 内部 token scheduler；没有运行真实
服务、GPU 或性能实验。

### 下一步

进入 Week 1 第 7 步最小端到端 smoke：实现真实 OpenAI-compatible HTTP 流后端，在 admission 边界接
`RequestRecord` 与 first-token 事件，补充 `REJECTED` 的分析口径和运行环境元数据，再以低请求量、
固定种子、预热和至少 3 次重复验证完整管线，同时采集 GPU 指标。该 smoke 仍只验证管线，不用于策略
性能比较。

## 2026-08-28 — 清尾：公平性只对出现过的类别计算

### 要解决的问题

第 5 步的公平性把 `interactive` 与 `batch` 两类都固定计入，空类别（该轮没有请求）被当成 `0.0`
达标率参与 Jain 指数与差距计算。

### 为什么需要

“该类没有请求”和“该类 0% 达标”是两回事。前者是**无观测**，后者是**实测的坏结果**。把无观测
按 0% 计，会让纯单类负载（例如全 interactive）虚报成“极不公平”（batch=0%、gap 拉满、Jain 被拉低），
误导后续 FCFS vs SLO-aware 的对比。

### 关键改动与概念

- `calculate_metrics` 先取 `present_classes = {r.request_class for r in records}`，公平性与分类别
  报告只覆盖出现过的类别，按 `RequestClass` 顺序保证确定性。
- Jain 的 `n` 改为**出现过的类别数**（0/1/2）；`slo_attainment_gap` 在无类别时约定为 `0.0`。
- 语义边界写进 `docs/METRICS.md`：空类别不进公平性，也不出现在分类别报告，须结合 `request_count`
  按“无观测”解读。

### 成功标志

- 新增 `test_fairness_ignores_absent_request_classes`：全 interactive 数据只报告 interactive，
  达标率 0.5、gap 0.0、Jain 1.0；`slo_by_class` 不含 batch 键。
- 既有手算样例（两类都在）、全失败、空输入用例行为不变。6 个 analysis 聚焦测试、全量 46 个测试、
  锁文件与 Ruff lint/format 均通过。

### 实际结论与下一步

这是一处指标**定义**修正，不涉及真实数据或性能结论。`REJECTED` 接入 analysis 计数器需要先把
`RequestRecord` 的 dispatch/first-token 时间戳改为可选（被拒请求从未 dispatch），该 schema 改动
与真实记录接线一起放到 Week 1 第 7 步。

## 2026-08-28 — Week 1 第 7 步（7a-1）：真实 httpx 流式后端

### 要解决的问题

在不连接真实网络、vLLM 或 GPU 的条件下，实现现有 `AsyncRequestBackend` 协议的真实 HTTP 流式
后端，并验证它可以被外部 `AdmissionQueue` 直接驱动。后端还需暴露首 token、输出 token 数和 HTTP
结果，使后续运行器能按 `request_id` 与 `AdmissionResult` 合并，而不修改现有协议或准入队列。

### 为什么需要

第 6 步只用纯内存 `FakeBackend` 验证了外部准入语义。第 7 步端到端 benchmark 若同时引入 HTTP、
运行器、GPU 采样和结果写入，故障边界过大；先把 OpenAI-compatible SSE 请求与解析隔离成可离线
验证的零件，后续才能分别判断错误来自网络流、准入生命周期还是记录管线。

### 关键命令或代码

- `HttpStreamingBackend` 接收 `BackendConfig`、调用方拥有的 `httpx.AsyncClient` 和可注入单调时钟；
  POST 地址由 `backend.base_url` 组成，model 与 HTTPX 网络阶段 timeout 也来自配置。
- 请求固定使用 `temperature=0`、常量提示词、`stream=true` 与
  `stream_options={"include_usage": true}`；精确 input-token 提示造型留给 7a-2 运行器。
- SSE 只消费 `data: ` 行并在 `[DONE]` 结束。首个非空 content delta 用注入时钟记一次；最终
  `usage.completion_tokens` 优先作为输出 token 数。缺少 usage 时，回退为非空 content delta 个数；
  这是 chunk 计数口径，不是 tokenizer 精确 token 数。
- `telemetry: dict[str, HttpCallTelemetry]` 以 request ID 为键，保存 `first_token_time_s`、
  `output_tokens`、`http_status`、`finish_reason` 和 `error`。非 2xx 响应先读完再抛出。
- HTTPX timeout 限制连接、读、写和连接池等网络阶段；`AdmissionQueue` 仍负责 enqueue 到终态的总
  timeout 并取消 backend task。后端不捕获 `CancelledError`，流式上下文负责关闭响应后继续向上传播。
- 聚焦测试命令：
  `UV_CACHE_DIR=/tmp/sloserve-uv-cache uv run pytest tests/test_http_backend.py -q`。

### 成功标志

- 4 个 `httpx.MockTransport` 离线测试覆盖带 usage 的成功流、无 usage 回退、HTTP 500，以及
  `AdmissionQueue.submit` 成功后按 request ID 读取首 token 遥测；测试没有访问真实网络或 GPU。
- 聚焦测试最终为 `4 passed in 0.09s`；锁文件解析 20 个 package，Ruff lint 通过，45 个文件通过
  格式检查，全量 50 个 pytest 测试通过。

### 失败与诊断

- 首次局部格式检查指出新增的两个文件不符合 Ruff 的机械折叠格式；运行 Ruff formatter 后重测通过。
- 收紧 usage 优先的最终归并后，格式检查再次要求折叠条件表达式；只格式化新增后端文件后，聚焦测试
  再次通过。没有观察到实现或测试逻辑失败。

### 实际结果

真实的 httpx 流式后端零件已实现，并在纯离线环境验证了请求构造、SSE 解析、HTTP 错误和现有准入
协议集成。共享注入时钟让 AdmissionQueue 生命周期时间与后端 first-token 时间处于同一基准。该结果
只证明功能边界，不包含真实 vLLM 兼容性、GPU 状态或任何性能数值。

### 下一步

实现 7a-2 benchmark 运行器：把 `AdmissionResult` 与 HTTP 遥测合并为请求记录，补齐拒绝记录口径、
精确 input-token 提示造型、环境元数据、GPU 采样和完整性报告；随后才在 7b 用真实单 GPU vLLM 做
低请求量、预热、固定种子和至少三次重复的端到端 smoke。

## 2026-08-28 — Week 1 第 7 步（7a-2）：离线 benchmark 运行器与完整性报告

### 要解决的问题

在不连接真实网络、vLLM 或 GPU 的条件下，把确定性负载生成、开环到达、外部 `AdmissionQueue`、
可注入后端遥测、请求事实记录、正式请求指标和完整性报告串成一个可重复运行的 benchmark 管线；同时
让原始 schema 忠实保存 rejected、排队中 timeout/cancelled 这类从未 dispatch 的终态。

### 为什么需要

第 7a-1 步只验证了单次 HTTP 流调用。若 benchmark 运行器把相对 arrival 当作真实单调时间、串行等待
每个 `submit()` 完成，或跨 repetition 后才按重复 request ID 读取 telemetry，就会分别造成时间线
不一致、把开环到达退化为闭环、以及遥测被覆盖。未 dispatch 记录若伪造 dispatch 时间，还会污染等待
指标和终态计数。

### 关键命令或代码

- `RequestRecord.dispatch_time_s` 改为可选；无 dispatch 时 first token 必须为空、status 不得为
  success，completion 不得早于 enqueue；有 dispatch 时继续校验 enqueue → dispatch → first token
  → completion。success 仍要求 dispatch、first token 和正 output token 数。
- JSONL 保留 `null`，CSV 以空串表示可选 dispatch/first token；两种格式都由现有序列化入口读回。
- `calculate_metrics()` 跳过未 dispatch 记录的排队等待，新增 `rejected_count`，并显式检查五种终态
  计数之和等于请求总数。
- `run_benchmark()` 每轮重新调用 `generate_requests()`，把相对时间平移到该轮注入 clock 的起点；预先
  创建所有 arrival task，各自在目标时刻调用即时 `AdmissionQueue.submit()`，因此一个提交等待终态时
  不会阻塞后续到达。
- 每轮全部 `AdmissionResult` 完成后、下一轮开始前，立即按 sequence/request ID 与
  `backend.telemetry` 合并。未 dispatch 记录不读取 telemetry，避免误取上一轮同名 request 的值。
- 全部 warmup 与正式记录共同写 JSONL/CSV；只用 `sequence_id >= warmup_requests` 的正式记录调用
  `calculate_metrics()`。完整性 JSON 为每个要求字段保存 value 或 missing_reason，无 GPU 样本时对
  利用率、显存、功耗写明“未采集（无采样器，留待 7a-3/7b）”。
- 聚焦测试：
  `UV_CACHE_DIR=/tmp/sloserve-uv-cache uv run pytest tests/test_metrics_records.py tests/test_analysis_metrics.py tests/test_benchmark.py -q`。
- 完整检查依次为 `uv lock --check`、`uv run ruff check .`、`uv run ruff format --check .`、
  `uv run pytest`，均使用 `/tmp/sloserve-uv-cache`。

### 成功标志

- schema/analysis/benchmark 的 14 个聚焦测试通过；两轮、每轮 1 个 warmup + 2 个正式请求共 6 条事实
  记录，JSONL/CSV 逐字段读回相等，repetition index 为 `0,0,0,1,1,1`。
- 离线假件用唯一 in-flight 槽和一个 waiting 槽确定性地产生每轮一条正式 rejected 记录；正式指标只
  看到 4 条记录，得到 2 个 success、2 个 rejected，warmup 仍保留在原始文件中。
- 同配置、同 seed、重置后的注入 clock/backend 连续运行两次，内存中的全部记录逐字段相等。
- 完整性报告逐项满足 value/missing_reason 二选一，包含正式请求终态计数一致性检查、总体/分类别 SLO、
  延迟分位数、吞吐、最长等待、公平性，以及三个显式缺失的 GPU 字段，并写明“仅管线验证，不作策略
  性能比较”。
- 锁文件解析 20 个 package，Ruff lint 通过，47 个文件通过格式检查，全量 54 个 pytest 测试通过。

### 失败与诊断

- 第一次局部 Ruff lint 把报告要求的中文全角标点判为 RUF001，同时发现一处超长行。保留规范要求的
  原文并对两条常量和对应断言做精确 `noqa`，拆分超长行后 lint 通过。
- 第一次局部 format check 指出运行器和测试的三处机械折叠差异；只对相关文件运行 Ruff formatter，
  随后聚焦与全量格式检查通过。没有发生功能测试失败，也没有访问网络或 GPU。

### 实际结果

Week 1 第 7 步 7a-2 的纯离线管线已完成。结果只证明负载、到达、外部准入、遥测 join、原始落盘、
正式指标与完整性报告之间的功能接线；测试数值是确定性假件数据，不是服务性能结果。GPU 未采集，
`env_version` 使用调用方注入的占位字符串，HTTP 后端仍使用固定短提示而未按 `input_tokens` 精确造型，
也未连接真实 vLLM 或新增 CLI。

### 下一步

进入 7a-3：增加 GPU 样本采集、真实运行环境/版本元数据采集和 `benchmark` CLI 子命令，同时保持本片
运行器的注入边界与原始事实口径。完成这些离线/接线能力后，7b 才在真实单 GPU vLLM 上执行低请求量
端到端 smoke；在 7b 之前不把 Week 1 第 7 步标记为完成，也不作策略性能比较。

## 2026-08-29 — Week 1 第 7 步（7a-3）：GPU 采样、环境元数据与 benchmark CLI

### 要解决的问题

在不连接真实 GPU、vLLM 或网络的开发环境中，补齐端到端 smoke 所需的最后一组接线能力：与请求并行
运行的单 GPU 采样器、明确缺失值的环境元数据，以及把真实 HTTP 流后端、采样器和 benchmark 运行器
组装起来的 CLI。真实 GPU 端到端运行仍留给 7b 人工执行。

### 为什么需要

7a-2 已能保存请求事实和完整性报告，但 GPU 字段必然缺失、`env_version` 仍由测试占位符注入，也没有
一条用户命令连接已经运行的 vLLM。若 GPU 样本使用另一时间基准，或功耗不支持时被填成 0，完整性报告
会产生不可审计的假证据；若 CLI 工厂在构造阶段就发请求，离线接线也无法独立测试。

### 关键命令或代码

- `NvidiaSmiSampler` 是异步 context manager；进入时创建命名后台 task，每 tick 经可注入 `reader`
  读取 utilization、memory、power，使用调用方共享的 monotonic `clock` 记样本时间，退出时取消并等待
  task。reader、utilization 或 memory 解析失败会跳过该 tick；不会追加 0 或其他无效占位值。
- 默认 GPU reader 使用标准库 `asyncio.create_subprocess_exec` 执行单 GPU MVP 的
  `nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader,nounits`。
  power 为 `[N/A]`、非数字或非有限值时保留为 `None`。
- `run_benchmark()` 只增加采样器外层 context 编排；一个 sampler 覆盖全部 repetition，退出后才读取
  它的不可变 samples 快照。保留 `gpu_samples` 直接注入；两者同时提供时 sampler 的实采样本优先。
- GPU 报告对部分缺失 power 只聚合非 `None` 值；全部缺失时写
  `missing_reason="device did not report power"`，utilization 和 memory 仍正常报告。
- `capture_environment()` 固定字段顺序输出 Python、driver、model revision、config hash、vLLM 和 torch
  版本。driver 默认调用 `nvidia-smi`，包版本默认查安装元数据；reader 抛错或返回空串统一写
  `unavailable`，不猜测版本。
- `build_benchmark_runtime()` 只从 config 构造 `httpx.AsyncClient`、`HttpStreamingBackend` 和
  `NvidiaSmiSampler`，并把同一个 `time.monotonic` 传给后端、sampler 和 runner；构造阶段不进入 client、
  不启动 sampler、不发网络。`sloserve benchmark --config <path>` 只在实际执行时进入 client 并运行。
- 聚焦测试命令为
  `UV_CACHE_DIR=/tmp/sloserve-uv-cache uv run pytest tests/test_gpu.py tests/test_metadata.py tests/test_benchmark.py tests/test_cli.py -q`。

### 成功标志

- 罐装 GPU 行 `45, 6136, 60.50` 与 `12, 6100, [N/A]` 产生使用注入 clock 的两个样本，退出 context 后
  没有 `sloserve-gpu-sampler` 悬挂 task；reader 首 tick 抛错后下一 tick 仍能采样。
- 注入 metadata readers 后字段顺序和值确定，vLLM/torch 不可读均显式为 `unavailable`。
- 假 sampler 跑完整个 benchmark 后，完整性报告有 utilization/memory 值，power 只聚合有效样本；同时
  注入直接 `gpu_samples` 时确认 sampler 优先。
- CLI parser 识别 `benchmark --config`，工厂测试验证 base URL、timeout、采样间隔和共享 clock，
  `MockTransport` handler 从未被调用。
- 4 组聚焦测试共 `12 passed in 0.15s`；首轮全量 pytest 为 `61 passed in 0.32s`。

### 失败与诊断

- 首次局部 Ruff lint 报告 sampler 清理可用 `contextlib.suppress(CancelledError)` 表达，并指出测试 import
  分组不标准；按建议调整后 lint 通过，清理语义不变。
- 首次全量 format check 只指出 benchmark arrival task 的长 `name` 参数换行不符合 formatter；按其
  精确建议换行。这是机械格式问题，不涉及运行逻辑。
- 测试全过程都由 reader、clock、sleep、sampler 与 `httpx.MockTransport` 假件驱动，没有执行默认
  `nvidia-smi` reader、真实 GPU、真实 vLLM 或网络请求。

### 实际结果

7a-3 的离线实现与接线测试完成。它证明 sampler 生命周期、缺失功耗语义、环境字段缺失纪律以及 CLI
构造边界可工作；所有数值均来自测试 fixture，不是实测 GPU 或服务性能结果。本片没有修改 vLLM 内部
scheduler，也没有扩展到多 GPU、KV cache 或 SLO-aware 调度。

### 下一步

进入 7b：人工用 `scripts/serve.sh configs/base.yaml` 启动固定版本的单 GPU vLLM，再真实运行
`sloserve benchmark --config configs/base.yaml`，核对原始 JSONL/CSV、环境字符串和完整性报告中的请求
及 GPU 字段，并保存失败证据。只有 7b 验收后才能把 Week 1 第 7 步标记完成；smoke 仍不作策略性能比较。

## 2026-08-29 — Week 1 第 7 步（7b）：真实单 GPU 端到端 smoke，Week 1 收官

### 要解决的问题

把 7a 的整套离线机器接到真实 vLLM，用一条命令跑出完整、可复现、带 GPU 指标的原始数据管线，验证
第 7 步验收：所有字段无无说明缺失。

### 为什么需要

前面所有零件都在假件下验证过，但"能离线跑"不等于"能对真实 vLLM 跑通并采到完整数据"。7b 是第一次
真上 GPU、第一次让五个时间戳全部来自真实事件、第一次把 GPU 曲线和请求时间线用同一时钟对齐。

### 关键命令与概念

- `scripts/serve.sh configs/base.yaml` 起服务（复用第 3 步的两处 Python 3.11 修复：flashinfer 补丁
  已幂等在位、`VLLM_USE_FLASHINFER_SAMPLER=0`）。
- `uv run sloserve benchmark --config configs/base.yaml` 在 dev `.venv` 里当客户端跑，httpx 打
  `localhost:8000`，并行 nvidia-smi 采样，写 `results/raw/week1-step7-smoke/`。
- 停服用 `pkill -TERM -f '[.]venv-vllm/bin/vllm'`（方括号防自匹配，第 3 步教训）。
- 完整性校验：脚本扫描报告，确认没有 `value=null 且 missing_reason=null` 的字段。

### 成功标志

- 75 条记录（3 rep × [5 预热 + 20 正式]），每 rep 25 条；正式 60 条全 success，五类终态计数之和
  等于总数。
- 完整性报告字段齐全：TTFT P50/P95/P99 ≈ 0.029/0.035/0.039 s；TPOT P50 ≈ 0.0095 s/token；端到端
  P50/P95/P99 ≈ 1.23/3.99/4.42 s；排队等待 P99 ≈ 0.0017 s；token 吞吐 ≈ 153.9 tok/s；SLO 达标
  60/60（两类各 30/30）；公平性 gap 0.0、Jain 1.0；GPU 利用率均值 40.5%/峰 46%、显存峰 6136 MiB、
  功耗均值 69.8W/峰 77.5W。
- 记录时间戳单调且与 GPU 样本共享同一单调时钟；停服后无残留进程、端点拒连、GPU 回到 14 MiB。

### 失败与诊断

- 就绪探测一度显示"~1s 就绪",一度怀疑是残留监听。核对 `/v1/models` 返回体、`ss -ltnp` 的
  pid、GPU 6136 MiB 和 server.log 的 "Application startup complete",确认是本次新启的真实服务——
  "1s" 只因读前置检查时它已加载完，探测启动时正好已就绪。教训：就绪探测要核对进程/显存/日志，
  不能只看端口应答。

### 实际结论与下一步

Week 1 收官：单 GPU 外部 FCFS 准入/排队/路由 MVP 已端到端建成并验证,产出可复现、字段齐全的原始
数据管线。**这是低负载管线验证,不是性能结论**——几乎不排队、SLO 全达标是设计使然。已知缺口:
`env_version` 的 vllm/torch 标 `unavailable`(CLI 在 dev venv 无法 import 服务环境的包;真值 vLLM
0.27.1 / torch 2.13.0+cu130 记于 ENVIRONMENT.md),可作为小 follow-up(让 capture_environment 通过
子进程读 `.venv-vllm`)。下一阶段(Week 2)才进入策略对比:加压、实现 SLO-aware/priority,并在
相同负载下比较 FCFS 与新策略。

## 2026-08-29 — Week 2（W2-1）：Static Priority 与策略可切换

### 要解决的问题

在现有单 GPU 外部准入/排队/路由层实现 Static Priority，并让 benchmark 依据
`config.router.policy` 在 FCFS 与静态优先级之间切换；本片只做纯离线假件验证，不连接 GPU、网络或
真实 vLLM，也不实现 W2-2 的 SLO-aware 与 aging。

### 为什么需要

Week 1 的 `AdmissionQueue` 已经通过公共 `SchedulingPolicy.order()` 选择下一个等待请求，但
`run_benchmark()` 没有注入配置对应的策略，因此 CLI 仍只能走队列内部的 FCFS 默认值。先完成简单、
可解释的静态优先级，可以验证策略接口和配置切换链路，再把动态时间、服务时间估计与公平性问题隔离到
W2-2。

### 关键命令或代码

- `StaticPriorityPolicy.priority_key()` 返回
  `(class_rank, arrival_time_s, sequence_id)`：interactive 的 `class_rank=0`，batch 为 `1`；同类请求
  自然复用 FCFS 的 arrival/sequence 顺序。策略不覆盖基类 `order()`，`now_s` 按 FCFS 风格显式丢弃。
- `build_policy(config)` 按 `SchedulerPolicyName` 分发：FCFS 构造 `FcfsPolicy`，STATIC_PRIORITY 构造
  `StaticPriorityPolicy`，SLO_AWARE 明确抛出
  `NotImplementedError("SLO-aware policy is implemented in W2-2")`。
- `run_benchmark()` 只有一个接线变化：构造 `AdmissionQueue` 时传入
  `policy=build_policy(config)`。`configs/base.yaml` 仍为 FCFS，测试通过 `model_copy` 使用类型化枚举
  切换策略。
- 聚焦命令：
  `UV_CACHE_DIR=/tmp/sloserve-uv-cache uv run pytest tests/test_static_priority_policy.py tests/test_policy_factory.py tests/test_benchmark.py -q`。

### 成功标志

- 混合乱序请求先输出全部 interactive，再输出 batch；两类内部都按 arrival time、sequence ID 排序，
  因此更早到达的 batch 仍排在 interactive 后。
- 工厂分别返回 FCFS/Static Priority 实例，并对 SLO-aware 给出 W2-2 的明确未实现错误。
- benchmark 集成假件把一个 batch 保持在唯一并发槽内，使更早到达的 batch 与更晚到达的 interactive
  同时等待；槽释放后，后端启动顺序为 in-flight batch、interactive、waiting batch。
- 修正测试配置类型后，10 个聚焦测试通过；相关 6 个文件的 Ruff lint 与 format check 通过。
- 最终四项检查依次通过：锁文件解析 20 个 package，Ruff lint 无错误，55 个文件符合 Ruff 格式，
  全量 pytest 收集并通过 67 个测试（`67 passed in 0.33s`）。

### 失败与诊断

- 第一次聚焦运行是 `9 passed, 1 failed`。集成测试用
  `model_copy(update={"policy": "static_priority"})` 写入裸字符串；Pydantic 的 `model_copy` 不重新
  校验，导致工厂收到 `str` 而不是 `SchedulerPolicyName`，并抛出 unsupported policy。把更新值改为
  `SchedulerPolicyName.STATIC_PRIORITY` 后，类型化配置契约恢复，聚焦测试全部通过。该失败没有涉及
  排序算法或异步队列行为。

### 实际结果

W2-1 的静态优先级排序和从 config 到 benchmark/CLI 的策略切换链路已在纯离线假件下打通。FCFS
配置继续构造原有 `FcfsPolicy`，没有修改 admission、配置 schema、CLI、vLLM 内部 scheduler 或
benchmark 的 join/schema/报告/采样逻辑。本片没有产生性能数据，也没有验证真实负载下的收益。

### 下一步

进入 W2-2：实现 SLO-Aware + aging。设计重点是 cost/slack/waiting time 的归一化、
`estimated_service_time` 的可解释估计、aging 的防饥饿边界与可控时钟测试；完成后再在同一负载下做
三策略正确性与性能比较。

## 2026-08-29 — Week 2（W2-2）：SLO-Aware 归一化评分与两级硬 aging

### 要解决的问题

在现有外部准入、排队和路由层的公共 `SchedulingPolicy` 接口后实现 `slo_aware`，并让配置工厂把
`SchedulerPolicyName.SLO_AWARE` 构造成带现有 `SloAwareConfig` 系数的策略。本片只实现请求排序和
工厂接线，不修改 admission、benchmark、HTTP、指标、GPU 或 vLLM 内部 token scheduler。

### 为什么需要

W2-1 已验证策略接口和配置切换，但静态类别优先级不能随请求预算、预计计算量、deadline 紧迫程度和
已等待时间变化。W2-2 需要一个纯函数式、可解释且可确定复现的动态 key，同时必须给持续等待的请求
一个不受评分权重或后续到达模式影响的优先级提升边界。

### 关键命令或代码

- `SloAwarePolicy` 在构造时保存 `SloAwareConfig` 的 input/output token cost、三项 weight 和
  `aging_threshold_s`；`priority_key()` 不读取全局配置，也不覆盖基类 `order()`。
- 对请求先计算 `budget = deadline - arrival`，非正预算用 `_EPSILON = 1e-9` 防止除零；
  `service_est = max_output_tokens * output_token_cost`，
  `cost = input_token_cost * input_tokens + output_token_cost * max_output_tokens`，
  `slack = deadline - now - service_est`，`waiting = now - arrival`。
- cost、slack、waiting 都除以同一请求 budget。最终
  `score = cost_weight * cost_norm + slack_weight * slack_norm - waiting_weight * waiting_norm`，
  分数越小越先 dispatch；每个 scored key 以浮点 `sequence_id` 确定性破同分。
- 两级硬 aging key 为：等待达到阈值时
  `(0.0, arrival_time_s, float(sequence_id))`，否则
  `(1.0, score, float(sequence_id))`。tier 0 是绝对最高优先级，内部按最早 arrival 排序，因此等待上界
  是 `aging_threshold_s` 加队首服务时间，不依赖 weights、cost 或到达模式；这提供了防饥饿的 bounded-
  wait liveness 保证，而不是依赖 soft score 最终“碰巧”足够小。
- 工厂的 SLO-aware 分支改为 `SloAwarePolicy(config.slo_aware)`；FCFS、Static Priority 和最终未知策略
  错误分支保持不变。
- 聚焦命令：
  `UV_CACHE_DIR=/tmp/sloserve-uv-cache uv run pytest tests/test_slo_aware_policy.py tests/test_policy_factory.py -q`。

### 成功标志

- 8 个策略测试分别覆盖 name、手算归一化 score、deadline 紧迫性、cheapness、硬 aging 跨 tier、aged
  bucket 最老优先、sequence ID 确定性和零预算有限 key；工厂测试确认三种配置均构造正确策略。
- 聚焦策略与工厂测试为 `11 passed in 0.07s`。
- 首轮四项全量检查依次通过：`uv lock --check` 解析 20 个 package；Ruff lint 通过；56 个文件符合
  Ruff 格式；pytest 收集并通过 75 个测试（`75 passed in 0.34s`）。

### 失败与诊断

本片聚焦测试、局部 Ruff 和首轮四项全量检查均一次通过，没有失败输出。所有分数和排序断言都来自
手算 fixture；没有启动 GPU、网络或真实 vLLM，也没有新增运行或性能比较结果。

### 实际结果

`slo_aware` 已作为外部请求调度策略接入现有配置工厂。其所有分支只依赖请求字段、构造时系数和显式
`now_s`，保持纯且确定；硬 aging 的 tier 优先级不受评分系数影响。本片没有产生吞吐、延迟或 SLO
改善数据，不能据此作性能结论。

### 下一步

进入 W2-3，按后续任务范围完成下一项 Week 2 工作；在新的明确验收合同之前不增加负载、不执行策略
性能比较，也不扩展到多 GPU、KV cache 感知或 vLLM 内部调度。

## 2026-08-30 — Week 2（W2-3a）：多策略正确性实验(离线半)

### 要解决的问题

Week 2 最后一项"完成小规模正确性实验"的验收是:三种策略可配置切换、测试全通过、无明显请求
饥饿。本片(W2-3a)只做**离线可测**的一半:一个"每策略跑一遍 benchmark 管线 + 从原始记录派生防
饥饿分析"的 runner、一个制造争用的起点配置、CLI 子命令与全套离线测试。真实 GPU 执行与裁决留给
W2-3b。不修改 vLLM 内部调度、不新增负载结论。

### 为什么需要

到 W2-2 为止三种策略都已实现,但从未在"存在排队争用"的场景下被一起检验过——step7 的 smoke
负载太低、几乎不排队,无法证明 SLO-aware 硬 aging 真能兜住等待、也无法暴露 Static Priority 在
高争用下饿死 batch 的行为。需要一个纯函数、可确定复现、以原始 JSONL/CSV 为事实源的分析,把
"无明显饥饿"变成可判读的量化证据。

### 关键命令或代码

- `src/sloserve/experiments/correctness.py`:`run_correctness_experiment(config, make_backend,
  env_version, policies, bound_margin_s, ...)` 对每个策略用 `_with_policy` 复制配置、`make_backend`
  工厂**每策略新建后端**(固定种子会复用 request_id,共享后端会串味),调用现有 `run_benchmark`
  并以 `file_stem=f"correctness-{policy}"` 落盘,再对 `formal_records` 调 `analyze_policy`。
- `analyze_policy` 纯函数派生:`max_queue_wait_s=max(dispatch−arrival)`、`_max_queue_depth`
  用 (enqueue,+1)/(dispatch,−1) 扫描线求同时等待峰值(同时刻先出后进 → 半开区间 [enqueue,dispatch)),
  `observed_max_service_s`、per-class 等待 max/P95(nearest-rank)、`rejected_count` 与
  `unfinished_count`(非终态状态计数)。
- 裁决 `StarvationVerdict`:`max_queue_depth<2 → NO_CONTENTION`;否则 `all_terminal 且
  unfinished_count==0 且 max_queue_wait ≤ aging_threshold_s+bound_margin_s → WITHIN_BOUND`;
  否则 `BOUND_EXCEEDED`。该 bound 正是 W2-2 硬 aging 承诺的"有界等待"的可检验形式。
- `configs/correctness.yaml`:争用起点 `max_in_flight=1`、`request_rate_rps=8.0`、`total_requests=60`、
  `interactive_fraction=0.6`、`aging_threshold_s=3.0`、`queue_capacity=256`、`repetitions=3`;文件
  顶部注明这是**起点**,W2-3b 要在真机上调到确有争用。
- `sloserve correctness --config ...` 子命令 + `build_correctness_runtime`(lazy 工厂,构造不做 I/O)。
- 检查:`UV_CACHE_DIR=/tmp/sloserve-uv-cache uv run pytest tests/test_correctness.py -q`。

### 失败与诊断

- **本片发生一次并发操作事故(已复盘并纠正)**:一次基于过期 `git status` 快照的误判,认为某个
  后台实现流程"没有产出",随即在主工作副本里开始写 `correctness.py`;实际上那个后台流程此时才刚
  启动并在并发改同一批文件,造成竞态。发现后:停掉该后台作业(只杀该 job 进程)、以磁盘最终收敛版
  (更防御的那一版)为准、再接管收尾。**教训**:并发操作时要以"进程/文件系统实时状态"而非一次性
  快照判断从属任务是否在跑;后台作业未真正阻塞时,用进程/状态而非口头回报来判断。
- 收敛后跑测试发现随附的 `test_no_contention...` 与其实现自相矛盾:它取 `records[:1]`
  (record 0 的 `enqueue==dispatch`,零宽等待)却断言 `max_queue_depth==1`;而半开区间语义(已被
  `depth==2` 那条通过测试佐证是自洽的)对零宽区间给 0。修法是让单条样例改用**真的等待过**的记录
  `records[1:2]`,使 `depth==1` 名副其实——保留正确的实现,修正选错样例的测试。

### 实际结果

四项检查全绿:`uv lock --check`、Ruff lint、Ruff format(58 文件)、`uv run pytest` **81 passed**
(较 W2-2 的 75 增 6 条正确性测试)。`config-check configs/correctness.yaml` 通过 schema 校验。
本片**没有**产生任何吞吐/延迟/SLO 数据,也未启动真实 GPU——三条策略的实际裁决(尤其 Static
Priority 是否 BOUND_EXCEEDED、SLO-aware 是否 WITHIN_BOUND)必须由 W2-3b 的真机运行给出。

### 下一步

W2-3b:在原生 Ubuntu 起真实 vLLM,用 `correctness.yaml` 先探一次、按 `max_queue_depth` 调到确有
争用(≥2)并触发硬 aging,再固定种子跑三策略各≥3 次,如实判读 `correctness-report.json`(不篡改
负面结果),GPU 利用率/显存/功耗与请求指标一并记录,更新 PROGRESS 勾掉 Week 2 最后一项。

## 2026-08-31 — Week 2（W2-3b）：真机多策略正确性实验(两个负载点)

### 要解决的问题

在真实单 GPU vLLM 上跑 W2-3a 的 `sloserve correctness`,验收 Week 2 最后一项:三策略可配置切换、
测试通过、无明显请求饥饿。只跑数与记录,不改策略/分析代码,不作性能结论,不碰 vLLM 内部调度。

### 为什么需要两个负载点

单槽服务能力 ≈ 0.6 rps。**固定到达 + max_in_flight=1 是确定性 D/D/1 队列**:rps<0.6 队列空
(NO_CONTENTION),rps>0.6 单调无界增长(饱和)——没有稳定中间态。要"会堆又能排空的中度争用",
数学上必须引入到达随机性(Poisson 未实现)或多并发槽(mif>1)。经 4 次探针确认:**"static 饿死
batch 的对照"是饱和现象(mif=1),"人人有界不饥饿"是非饱和现象(mif=4),二者在当前机制下互斥**。
故正式跑两个点各取其一,拼出完整诚实图景。

### 关键命令或代码

- 两个版本化配置:`configs/correctness-overload.yaml`(mif=1, rps=1.5)、
  `configs/correctness-concurrent.yaml`(mif=4, rps=2.0);其余同 `correctness.yaml`
  (total=60, interactive_fraction=0.6, warmup=5, repetitions=3, aging_threshold_s=3.0)。
- 每点:`scripts/serve.sh <cfg>` 起真实 vLLM(Qwen3-0.6B commit c1899de2…)→ 轮询 `/v1/models`
  就绪 → `uv run sloserve correctness --config <cfg>` → `pkill` 停服务、确认端口空 → 下一点。
  两点不同时起服务。固定种子、3 重复,原始数据落 `results/raw/week2-step3-correctness/{overload,concurrent}/`。

### 成功标志 / 实际结果(真机,3 重复,每策略 formal=180、全 dispatched、全终态、0 拒绝)

**过载点 overload(mif=1, rps=1.5,系统饱和 → 全 bound_exceeded,aging_bound=5s):**

| 策略 | 裁决 | depth | interactive 最大等待 | batch 最大等待 |
|---|---|---|---|---|
| fcfs | bound_exceeded | 41 | 69.8s | 70.0s |
| static_priority | bound_exceeded | 27 | **3.4s** | **73.0s** |
| slo_aware | bound_exceeded | 43 | 74.2s | 74.4s |

- static_priority **如实暴露 batch 饥饿**(interactive 3.4s vs batch 73.0s);fcfs 类别无视(两类≈70s);
  slo_aware 过载下所有请求瞬间超 3s 阈值、全进硬 aging 桶按到达排序,**退化为 FCFS-公平**(两类≈74s,
  无类别被单独针对)。硬 aging 保证"相对不被单独饿死",不保证系统整体过载时的绝对等待。

**并发点 concurrent(mif=4, rps=2.0,非饱和 → 全 within_bound,0 拒绝):**

| 策略 | 裁决 | depth | interactive | batch |
|---|---|---|---|---|
| fcfs | within_bound | 3 | 1.33s | 1.09s |
| static_priority | within_bound | 4 | 1.21s | 2.59s |
| slo_aware | within_bound | 3 | 2.46s | 2.13s |

- 有真实排队(depth 3–4)、三策略全 within_bound、无任何类别饥饿、0 拒绝——干净满足"无明显饥饿"。
  该点 waits<3s 阈值,硬 aging 作为休眠安全网未触发(其正确性由 W2-2/W2-3a 离线单测覆盖)。

**GPU(两点一致量级):** 利用率 mean ~40–44% / peak 43–59%;显存 peak 6136 MiB;功耗 mean ~79–83W /
peak 86–102W。观测值可由 `results/raw/week2-step3-correctness/` 的 JSONL、per-policy completeness、
config_hash 与 metadata 复现。

**边界**:这是单 GPU **正确性验证**,不是策略性能比较;overload 的 bound_exceeded 是"系统过载"而非
调度缺陷,concurrent 的 within_bound 才是防饥饿证据。三策略均仅由 config 切换。

### 失败与诊断

- 在沙箱构建环境起 vLLM 报 `RuntimeError: Failed to infer device type`——**该沙箱看不到 GPU/CUDA**,
  真机 serve 只能在有 GPU 访问的环境做;非 GPU 活(建配置/写文档)仍可在沙箱完成。
- 编排脚本首轮"SERVER DIED"误报:serve.sh 在 exec vllm 前先跑 serve-command+flashinfer 补丁,
  最初几秒 vllm 进程未起,过早的 pgrep 死检查误判。改为纯 `/v1/models` 就绪轮询 + 30s 后才判死,修复。

### 下一步

Week 2 完成(三策略可切换、测试全通过、并发点无明显饥饿、过载点如实记录 static 饥饿)。进入
Week 3:完整评测(实验 A–E)、每组≥3 重复、吞吐-延迟/速率-P99/SLO 达标率图表、README 与技术报告。

## 2026-09-02 — Week 3 W3-0 评测工具(扫描器 + 出图,尚未跑数)

### 问题与为何需要

Week 3 要跨"负载谱"对比策略,而不是跑单点。已有的 `run_benchmark()` 只能对**一个** (policy, rate)
产出一份汇总;缺的是**参数扫描 + 跨点聚合 + 自动出图**。本步只搭工具并用合成后端验证,**不碰 GPU、不跑
真实实验**,因此本节没有任何性能数字——真实 A–E 由后续在单 GPU 上执行后再记录。

### 关键代码

- `src/sloserve/experiments/sweep.py`:`SweepPoint`(label + 受支持的覆盖项:policy / request_rate_rps /
  interactive_fraction / max_in_flight / aging_threshold_s / cost·slack·waiting 权重 /
  disable_length_estimate)与 `run_sweep()`。**每点用全新后端**(与 W2-3a 同理:固定种子复用 request_id,
  共享后端会串遥测),复用 `run_benchmark()`,把每点参数 + MetricsSummary 全部头条指标摊平成一行,写
  `sweep-results.csv` 和 `.json`(`allow_nan=False`)。每点自身的原始 JSONL/CSV 仍由 run_benchmark 保留。
- `src/sloserve/analysis/plot_results.py`:懒加载 matplotlib(Agg 无头后端),**只从聚合 CSV** 重生三张图——
  ①吞吐-延迟(x=token 吞吐, y=端到端 P99)②速率-P99 ③SLO 达标率-速率(总体+分类),每策略一线。满足验收
  "图表可由原始数据自动生成"。无 pandas 依赖。
- CLI 新增 `sloserve sweep --config <sweep.yaml>` 与 `sloserve plot --results <csv> --out <dir>`,
  沿用 benchmark/correctness 的 runtime 工厂(每点新建并关闭 httpx client)。
- `configs/sweeps/`:expA(3策略@近容量点)、expB(速率阶梯 0.5–4 × 3策略)、expC(interactive 比例
  0.2/0.5/0.8 × 3策略)、expE(slo_aware 消融:full / no-slack / no-aging(阈值置极大)/ no-length-estimate)。

### 设计决策

- **实验 D 服务端参数**:给 `ServerConfig` 加了可选的 `max_num_batched_tokens` / `enable_chunked_prefill` /
  `enable_prefix_caching`(默认 None → 不进 serve 命令 → 行为不变),plumb 进 `vllm_serve_args()`。D 因是
  服务端参数需**每点重起 vllm**,由外层 shell 用一系列独立 base 配置驱动,不进 Python 扫描器。
- **no-length-estimate 消融**:给 `SloAwareConfig` 加 `disable_length_estimate`(默认 False,保住 W2-2 语义)。
  开启时 `service_est=0` **且** `cost=0`——即同时移除服务时间估计与整个 token-cost 项(二者主要由输出长度
  估计驱动);策略退化为"纯 deadline slack + waiting + aging"。报告中须如实写明该点移除的是这两项,而非只
  移除 service_est。

### 成功信号 / 边界

- 四项检查全绿:`uv lock --check` 通过(matplotlib 仅加入 dev 组,运行时依赖保持精简)、ruff check/format 通过、
  **pytest 90 passed**(新增 sweep/plot/config/policy 共 9 个测试,含单策略与缺失分类的边界)。
- 边界:**本步只交付工具,未产出任何实验结果**;A–E 的真实数字待单 GPU 执行后填入,期间不得推测性能。

### 下一步

在单 GPU 上起一次常驻 vllm 跑 A/B/C/E(全路由侧),D 每点重起 vllm;再用 `plot` 出图,落 README 与技术报告。

## 2026-09-02 — Week 3 SLO-aware 评分量纲修正与重跑配置

### 问题与为何需要

第一次 Week 3 扫描后复查评分公式，发现 `service_est = max_output_tokens * output_token_cost`
仍是抽象 cost 单位，却从以秒为单位的 `deadline_time_s - now_s` 中直接相减。这个量纲错误会让
service estimate 主导 slack，并使按 budget 归一化的 cost 项对短 deadline 请求产生非预期惩罚。
因此第一次 Week 3 运行中的全部 `slo_aware` 数据已经失效，必须重跑；`fcfs` 和
`static_priority` 不使用该评分公式，其已有行不受影响。原始实验目录未删除或改写。

### 修正设计与关键代码

- `SloAwareConfig` 将 `input_token_cost` / `output_token_cost` 重命名为
  `input_token_seconds` / `output_token_seconds`，明确它们是每个 token 的服务时间秒数估计；base
  配置使用粗略估计，并注明评分经过归一化，不把具体估计值当成实测性能结论。
- `SloAwarePolicy.priority_key()` 现在先计算
  `service_time = input_tokens * input_token_seconds + max_output_tokens * output_token_seconds`。
  `service_time`、`slack = deadline - now - service_time`、waiting 和 budget 均以秒为单位。
- `cost_weight` 现在权衡 `service_time / budget`，成为归一化 SJF 项；`slack_weight` 权衡真实剩余
  时间 slack，`waiting_weight` 仍降低已等待请求的分数。epsilon budget guard 与两级硬 aging key
  保持不变。
- `disable_length_estimate=true` 时 `service_time=0`，同时移除 SJF 长度项，并使 slack 退化为
  `deadline - now`，即 EDF + waiting + hard aging。
- 新增 `expA-sat.yaml`、`expE-sat.yaml` 两个 3.0 rps 饱和点配置，以及只重跑修正后
  `slo_aware` 行的 `expB-slo.yaml`、`expC-slo.yaml`。所有配置保持固定 seed、warmup、60 个正式
  请求和至少 3 次重复。

### 成功信号、失败诊断与实际结果

纯 CPU 聚焦测试覆盖手算后的量纲一致分数、SJF 长短任务顺序、相同等待下 interactive 优先于大
batch、禁用长度估计和全部新 sweep YAML 的加载。开发过程中没有启动 vLLM、访问 GPU 或执行真实
请求；本步没有新的吞吐、延迟或 SLO 结果，也不作性能改善宣称。

### 下一步

由具备真实单 GPU 访问的实验编排器运行 A/E 饱和点，并用 B/C refresh 配置替换旧聚合 CSV 中失效的
`slo_aware` 行；保留不受影响的 `fcfs` / `static_priority` 行和全部失败、负面及原始请求级数据，
然后从更新后的 CSV 重新生成图表。

## 2026-09-02 — Week 3 完成：完整评测、出图与技术报告

### 要解决的问题

Week 3 需要在真实单 GPU 上完成负载强度、混合比例、策略消融、aging、引擎参数和多 seed 饱和稳健性
评测，并把请求级事实、聚合 CSV、图表和结论连成可复核证据链。结果必须同时报告吞吐、延迟、SLO、
公平性、排队和 GPU 状态，不能把某一个有利指标写成策略全面胜出。

### 方法与环境

默认工作负载为固定到达、每次 60 个正式请求、3 次重复、5 个 warmup、seed 20250825、
`interactive_fraction=0.5`、`max_in_flight=4` 和 `queue_capacity=256`。interactive 输入/输出为
64–256 / 32–128 tokens，batch 为 512–2048 / 128–512 tokens。环境为单张 NVIDIA GeForce RTX
4080 Laptop（12282 MiB），driver 580.178.04、CUDA 13.0、torch 2.13.0+cu130、vLLM 0.27.1、
Python 3.11.14；模型为 Qwen/Qwen3-0.6B revision
`c1899de289a04d12100db370d81485cdf75e47ca`。运行期间 GPU util 约 55%（peak 61%）、显存 peak
6136 MiB、功耗 mean 约 77 W。

`sloserve sweep` 保留请求级 JSONL/CSV、completeness report、config hash、环境元数据和聚合
`sweep-results.csv/json`；`sloserve plot` 与 `scripts/plot_week3_extra.py` 只读保存的 CSV 自动重生
六张图。所有原始事实保留在 `results/raw/`。

### 关键结果与诚实边界

- 速率扫描显示 token throughput 在 rps=3.0 附近约 430 tok/s 进入平台，排队起点在 rps 1.5 与
  2.0 之间。rps≤2 时三策略 interactive SLO 均约 1.00。rps=4.0 时 static_priority interactive
  为 0.85，但 batch 降到 0.72；slo_aware interactive 为 0.32、batch 为 0.99。该图的
  fcfs/static 行来自初始 session，slo_aware 行来自量纲修正后的 session。
- rps=3.0、六个独立 seed 的 headline 结果为：fcfs interactive 0.57±0.31、overall
  0.80±0.14、TTFT P99 4.27±1.66 s、e2e P99 6.86±1.60 s、throughput
  438±16 tok/s、Jain 0.87±0.15；static_priority 为 0.98±0.01、0.99±0.00、
  5.28±2.02 s、7.94±2.16 s、433±14 tok/s、1.00±0.00；slo_aware 为
  0.92±0.07、0.96±0.03、4.03±1.15 s、6.55±1.18 s、446±19 tok/s、
  1.00±0.01。三策略 batch SLO 均为 1.00。
- mix 扫描只在 interactive 为少数的 fraction 0.2 明显分化：interactive SLO 为 fcfs 0.48
  （Jain 0.89）、static 1.00、slo_aware 0.97；fraction 0.5 为 1.00 / 0.97 / 1.00，
  fraction 0.8 三者均为 1.00。
- rps=3.0 消融的 interactive SLO 为 full 0.60、no-slack 0.47、no-length-estimate 0.46、
  no-aging 0.92。去掉 slack 或 shortest-job 项都会降到约 0.46；class-blind hard tier 是饱和下
  限制 interactive SLO 的部分。
- aging 3 s / 10 s / 30 s / ∞ 的 interactive SLO 与最长排队分别为 0.85 / 3.86 s、
  0.97 / 7.07 s、0.96 / 6.44 s、0.88 / 8.25 s。30 s 时最长等待 6.44 s，hard tier 未触发，
  因而 30 s 与 ∞ 的 dispatch order 应相同；0.96 与 0.88 的约 0.08 差异是 vLLM execution-timing
  noise，用于校准饱和点噪声底。
- 单 seed 引擎扫描中，`max_num_seqs=8/16/32` 的 fcfs / slo_aware interactive SLO 分别为
  0.22 / 0.50、0.19 / 0.55、0.36 / 0.55；`max_num_seqs=16` 开 chunked prefill 为
  0.12 / 0.77。每种引擎配置下 slo_aware 都领先 fcfs，但细粒度引擎趋势仍在饱和噪声内。

饱和拐点的 run-to-run variance 很高。同一 slo_aware、aging 3 s、rps=3 配置在一个 session 为
约 0.51/0.55/0.60，在 aging 扫描为 0.85，在多 seed 稳健性实验为 0.92±0.07。固定 seed 只固定
arrival 与 token count，不固定 vLLM continuous batching 的执行时序。因此方向性结论稳健，绝对
interactive-SLO 数值有 session 敏感性，任何单次饱和值都不能当成定论。

### 实际结论

低于容量且 mix 平衡时三策略等价；有排队争用时才分化。static_priority 的 interactive SLO 最强且
最稳定，但 tail latency 最差，并在更深过载时开始饿 batch。fcfs 的 interactive SLO 最差、波动最大且
公平性最低。slo_aware **不是** interactive-SLO 胜者，而是 latency、throughput、fairness 与
no-starvation 的平衡选择；其与 static 的剩余 interactive 差距来自 class-blind hard aging。

### 下一步

增加 seed 与独立 session 以收紧饱和点区间；实现 Poisson 到达；评估更大模型、multi-GPU 与
KV-cache-aware routing；用 learned predictor 替代粗略 token-seconds proxy；把二元 hard tier 改为
逐级提升、tier 内仍保留 SLO 顺序的 multi-level aging。

## 2026-09-02 — Week 4：Poisson 到达与多级 aging 离线实现

### 要解决的问题与为什么需要

固定速率到达形成规则的 D/D/1 型输入，在低负载与持续过载之间很难稳定制造短时聚集和中等排队。
同时，原有二元 aging 在阈值处把所有老请求直接放入 class-blind 最高层，虽然保证 liveness，但会突然
丢失层内的 SLO 顺序。Week 4 的离线实现增加可复现 Poisson 到达，并把阈值以下的 aging 改为多级渐进
提升，用于后续合并评估到达波动与 aging 粒度。

### 关键代码与设计

- `PoissonArrivalSchedule(request_rate_rps, seed)` 使用独立的 `random.Random(seed)`；首个相对到达时刻
  固定为 0.0，后续间隔由 `expovariate(request_rate_rps)` 生成并累加。相同 seed 得到相同时间序列，
  不同 seed 得到不同序列。该 RNG 只属于 arrival schedule，没有消费或改变 `generator.py` 的请求类别
  与 token 内容 RNG 流。
- `arrival_schedule_from_config` 现在按 `ArrivalProcess.POISSON` 构造上述 schedule，FIXED 行为保持
  不变。指数分布间隔允许短时聚集，因此能覆盖固定 D/D/1 输入之外的 bursty 和 moderate-queue
  regime。
- `SloAwareConfig.aging_levels` 默认 1。令 `K=aging_levels`、每级宽度
  `step=aging_threshold_s/K`；阈值以下的请求按等待时间进入渐进 tier，同 tier 内继续用原 SLO score
  排序。等待达到 `aging_threshold_s` 后仍进入 primary key 为 0.0 的 ceiling，严格按 arrival 和
  sequence oldest-first。
- ceiling 没有变化，所以 strict oldest-first bounded-wait guarantee 与 correctness 分析使用的
  `aging_threshold_s + bound_margin_s` 上界保持有效。`K=1` 精确复现此前二元策略：阈值以下 key 的
  primary 为 1.0，达到阈值后为 0.0。
- sweep 聚合新增 `aging_levels` 参数列；`expH-poisson.yaml` 对比 Poisson 下三种策略，
  `expI-multilevel.yaml` 在同一 3.0 s ceiling 下扫描 1、2、3、5 个 graduated tiers。

### 成功信号、失败诊断与实际结果

纯 CPU 测试覆盖 Poisson 的同 seed 重放、不同 seed 分化、长度与单调性、空/负计数、非法速率和大样本
平均间隔；aging 测试覆盖 `K=1` 精确 key parity、中间 tier、同 tier 按 score 而非 arrival 排序，以及
ceiling oldest-first。sweep 测试确认覆盖项进入有效配置和聚合行，两份 Week 4 YAML 可通过现有 loader。
实现期间没有启动 vLLM、访问 GPU 或发起网络请求。本步没有执行真实实验，也没有吞吐、延迟、SLO 或
公平性结果可报告。

### 下一步

由具备单 GPU 访问的实验编排器分别运行实验 H 与 I，保留请求级原始数据、GPU 样本、配置和失败运行，
再同时比较 tail latency、throughput、分类 SLO、最长排队与 fairness；在真实数据产生前不作性能结论。

## 2026-09-02 — Week 4：Poisson × 多级 aging 联合评测（真实 GPU）

### 要解决的问题与为什么需要

Week 4 实现落地后，需要在真实 vLLM 上回答两件事：策略排序在 bursty Poisson 到达下是否仍成立；
以及“多级 aging 能否保住重载下的 SLO 区分度”这一假设是否被数据支持。后者是 Week 4 这条线是否成立
的关键。前一轮单种子结果看似显示 K 越大越差，但饱和拐点方差极高（见 expG），单种子不足为凭。

### 方法与命令

- expH：Poisson（rps=3.0，单种子 20250825）下对比 fcfs / static_priority / slo_aware。
- expI 多种子：`configs/sweeps/expI-multiseed.yaml`，对 `aging_levels` K∈{1,2,3,5} 各用 expG 的
  同 6 个种子（20250825, 11, 202, 3407, 42, 65537）重放，共 24 点，Poisson rps=3.0。
  每个 K 在 6 个种子上算 SLO_int / 最长排队 / batch SLO / jain 的 mean±std（`pstdev`）。
- `uv run sloserve sweep --config configs/sweeps/expI-multiseed.yaml`（37 min，24 行）；
  离线聚合脚本按 label 前缀解析 K。图由 `scripts/plot_week3_extra.py` 的 `multilevel_aging()` 生成。

### 成功信号、失败诊断与实际结果

- expH：单种子下 static 0.86 / slo_aware 0.28 / fcfs 0.14，排序与固定到达一致，无拒绝、batch SLO 满。
  仅作方向性确认，绝对值受同样的饱和方差影响。
- expI 多种子：SLO_int 均值 K=1 0.48±0.20 最高，K=2 0.28±0.13、K=3 0.36±0.23、K=5 0.37±0.23，
  没有任何 K 超过 K=1。**假设不成立**。各 K 标准差 0.13–0.23，盖过均值差距，因此结论不是“K=1 碾压”，
  而是“多级 aging 无可辨识的交互 SLO 收益”——单种子看到的 0.41→0.12 主要是噪声，多种子把它坐实。
- 有连贯的取舍方向：K 增大时最长排队 9.0s→7.6s、batch SLO 0.956→0.991 上升，交互 SLO 与 jain 下降。
  机制：分级让“等待时间”更早、更频繁地粗粒度压过 SLO 打分，把服务从交互挪向 bounded-wait 与批处理。
  多级 aging 是“交互紧迫度 ↔ 饥饿上界”的再分配旋钮，不是恢复 SLO 区分度的手段；真正的旋钮仍是
  ceiling 阈值本身（expF）。

### 下一步

Week 4 这条线以“提出→实现→多种子严谨证伪+机制解释”收尾并写入 REPORT/README。若继续，可换用
真正影响 SLO 区分度的 ceiling 阈值自适应，或转向 learned 长度预测、多卡路由。

## 2026-09-03 — Week 5：自适应 ceiling 阈值（真实 GPU，负面 + 正面副产物）

### 要解决的问题与为什么需要

Week 3–4 反复指向:aging ceiling 的**位置(阈值)**才是掌控重载 SLO 区分度的主旋钮。固定阈值的两难:
太低(3s)→ 饱和队列整体跌进 class-blind ceiling → 退化成 FCFS(expF);太高 → 轻载时对离群
请求收得慢。于是提出「自适应 ceiling」:阈值随实时拥塞浮动 `clamp(margin×median_queue_wait,
floor, cap)`,cap 保留有界最坏等待的 liveness 保证。假设:自适应能在各负载区间都贴近"最佳固定值"。

### 方法与命令

- 新增 `SloAwareConfig.adaptive_ceiling/ceiling_margin/ceiling_floor_s/ceiling_cap_s`(+ floor≤cap
  校验);`SloAwarePolicy._effective_threshold` 取当前队列等待中位数、`order()` 在自适应开启时按浮动
  阈值重排;`adaptive_ceiling=False` 精确复现旧固定行为(parity 测试)。
- `configs/sweeps/expJ-adaptive.yaml`:fix{3,10,30} vs adp-cap{10,30},各 6 种子(同 expG/expI),
  Poisson rps=3.0,aging_levels=1(隔离阈值效应)。共 30 点。
  `uv run sloserve sweep --config configs/sweeps/expJ-adaptive.yaml`(42min)。
- 5 新单测:parity、_effective_threshold 的 floor/cap/scaling 钳制、以及"低固定阈值毁掉的 SLO 顺序
  被自适应保住"的排序对照。全套 106 pytest 通过,ruff/format 干净。

### 成功信号、失败诊断与实际结果

- **自适应决定性失败**(两轴皆输):interactive SLO adp-cap10 0.39±0.20 / adp-cap30 0.40±0.27,
  vs fix3 0.83±0.25 / fix10 0.91±0.11 / fix30 0.91±0.08;且自适应最长排队 7.9–8.4s 反而**最长**
  (固定 5.2–6.1s)。机制(标为对聚合结果的合理解读、非逐次 trace):median-in-queue-wait 是去稳定
  信号——Poisson 突发注入大量 zero-wait 新请求,恰在突发时把中位数拽低 → 阈值降低 → 中龄请求在最坏
  时刻涌入 class-blind ceiling;队列整体变老时阈值又升高。阈值与需求反向、排序 thrash → SLO 与尾延迟
  一起变差。
- **正面副产物(更有用)**:Week-3 饱和方差的元凶是 3s 阈值**设低了**,不是不可约的执行时序噪声。
  抬到 10–30s:interactive SLO 0.83→0.91,run-to-run std **0.25→0.08–0.10**。即大部分"高方差"是配置
  踩在刀刃上(多数请求跨阈成 FCFS、对 dispatch 时序敏感),而非系统固有属性。
- base.yaml 默认 `aging_threshold_s` 本就是 30(好区间),3s 只是各实验 base_overrides 的选择 → **不
  改 base**。

### 下一步

自适应 ceiling 这条按「提出→实现→多种子严谨证伪 + 机制解释」收尾。若继续调度层:换更稳的拥塞信号
(如 EWMA 平滑、按 class 分别定阈)或直接采纳 10–30s 常量;或转向 learned 长度预测、多卡路由。

## 2026-09-03 — Week 6 batch 1：真实重尾长度、长度预测器与估计来源切换

### 要解决的问题与为什么需要

原合成负载把 `max_output_tokens` 同时作为后端生成上限和调度器长度信息，实际输出又通常接近该上限，
因此 SLO-aware 策略近似提前知道真实长度，不能检验长度信息不完美时的排序效果。本批次把后端目标长度
与调度可见信息解耦：真实目标仍由 `max_output_tokens` 发送给后端，调度器只按配置读取真实目标、统一
advertised cap，或由 request class 与粗粒度 prompt kind 预测的长度。所有新增开关默认
`uniform_cap` / `true`，保留旧工作负载与调度计算路径。

### 关键代码、方法与命令

- `RealisticLengthConfig` 定义 short/medium/long 三成分 log-normal 混合，默认权重
  0.50/0.35/0.15、各成分 log-mu 6.2/7.0/7.8、共同 sigma 0.7，并把目标钳制到 8–2048 tokens；
  每个请求同时记录 prompt kind 和 2048-token advertised cap。真实分支只在旧 class/input RNG draw
  之后额外抽样，`uniform_cap` 分支不消耗新随机数。
- 长度预测器用 `(request_class, prompt_kind)` 桶内目标中位数和全局中位数 fallback，不引入新运行时
  依赖。训练命令为 `uv run python scripts/train_length_predictor.py --config
  configs/sweeps/expK-A-length-source.yaml --train-seed 999 --eval-seed 424242 --n 4000 --out
  results/artifacts/length-predictor.json`。artifact 只依赖 train-seed 999（与全部六个 expK-A 评测
  seed 不相交，零泄漏）；held-out MAE 改用与评测集不相交的 seed 424242 报数，避免最初误用 expK-A 的
  s1 seed 20250825。已核实：换 held-out seed 不改变 artifact（md5 相同），且两种 seed 的 MAE 几乎一致，
  说明桶中位数预测器是总体统计、对具体 seed 不敏感。
- SLO-aware 的 `length_source` 支持 true/advertised/learned；learned artifact 在策略构造时加载一次。
  `disable_length_estimate=true` 仍直接令 service time 为零。expK-A 在 Poisson rps=3.0 饱和点对三种来源
  各重放六个 seed，共 18 点；本批次只用 loader 验证所有点，没有执行 sweep。

### 成功信号、失败诊断与实际结果

固定 seed 12345 的八请求 parity fixture 保持旧 class/input/output 序列，新增 envelope 字段均为 `None`；
10,000 请求测试确认三类比例接近配置、每类内部长度有方差，且总体最大值超过中位数两倍。解耦测试用
相同调度可见特征但不同真实目标，确认 advertised/learned 估计不变而 true 估计变化。配置、artifact
round-trip、未知桶 fallback、已知 SLO key 回归和 sweep 行字段均有 CPU 测试覆盖。

训练 seed 999、held-out seed 424242、各 4,000 请求时，learned MAE 为 381.33 tokens、Pearson
corr 为 0.63；统一 2048-token advertised-cap baseline 的 MAE 为 1043.27 tokens、corr 为
0.000000（learned 约为 naive 的 1/2.7 误差）。artifact 保存了六个桶中位数和全局中位数，可由上述命令重建。配置首次验证失败是 YAML
把未加引号的 `true` 解析为布尔值；把六个 length-source 值写成字符串后，18 个点全部加载成功。
最终 lockfile、Ruff lint、Ruff format 和 118 个 pytest 均通过，全程未启动服务、运行 GPU 工作或执行
`sloserve sweep`。

### 下一步

GPU expK-A 留给 orchestrator 执行；运行时必须保留 18 点的请求级 JSONL/CSV、GPU samples、配置与
环境元数据，并联合报告吞吐、TTFT、TPOT、P50/P95/P99、分类 SLO、失败/超时、排队公平性和 GPU
利用率/显存，再判断 learned length source 相对 true 与 advertised 的实际调度取舍。

## 2026-09-03 — Week 6 expK-A：真机三路长度估计对照(oracle 有用、粗预测器反伤)

### 要解决的问题
realistic 重尾负载(log-normal 混合)下,比较喂给 slo_aware 服务时间项的三种输出长度来源:
true(oracle,不现实)/ advertised(naive,恒 2048)/ learned(桶中位数预测器)。判据:learned 是否把
naive→oracle 的交互 SLO 差距赚回。先用 rps-ladder smoke(realistic)定饱和点=rps 1.2(qwait 成形、无超时)。

### 方法与命令
- smoke:`configs/sweeps/probeK-realistic-rps.yaml`(rps 0.4/0.6/0.9/1.5),定 rps=1.2。
- `configs/sweeps/expK-A-length-source.yaml`:{true,advertised,learned}×6 seed=18 点,Poisson rps=1.2,
  realistic 负载,`uv run sloserve sweep`(~52min)。`length_source` 需写成带引号字符串(YAML 否则当 bool)。

### ⚠️ 首轮无效 + 修复(重要)
首轮结果被一个 bug 污染:`benchmark._place_on_clock` 重建 envelope 时**丢了 `advertised_cap_tokens`
和 `prompt_kind`**(重建后的 envelope 才进队列),导致 advertised 静默回退成真实 T(=oracle)、learned 因
prompt_kind=None 退化成全局常数。parity 测试没抓到——uniform_cap 下这俩字段本就是 None。已修
`_place_on_clock` 保留全部字段 + 加回归测试 `test_place_on_clock_preserves_all_scheduling_fields`,
用修正代码重跑。下方为**修正后**结果。

### 成功信号、失败诊断与实际结果(修正后)
- SLO_int(mean±std):**oracle 0.74±0.17 > naive 0.65±0.18 > learned 0.51±0.17**;中位延迟同序(e2e_p50
  oracle 3.82 / naive 4.55 / learned 5.56;qwait_p50 1.00/1.53/2.31)。oracle 在 5/6 seed 最优、learned 在
  5/6 seed 最差,方向稳。0 超时 0 拒。
- **两个并列事实**:①**准确长度确实改进 SLO-aware 调度**(oracle > naive)——SJF 用真实长度正确地把真正短的
  interactive 抢前;②**我们的粗预测器不但没赚回,反而低于 naive**。
- **机制(核心洞察)**:cost 项是 SJF,小 service_time → 抬优先。oracle 正确抢前短请求 → SLO/中位延迟双升;
  naive 人人恒定大估计 → SJF 整个失效、回退到 slack/waiting;learned 桶中位数虽 MAE 更低(381 vs 1043)却
  **最差**,因为**预测精度(MAE)不是调度的正确目标**——它的误差是**结构化**的:同一 `(class,kind)` 桶给同一
  中位数,而 kind ⊥ class 且桶内重尾,于是一个被标 `long` 的短 interactive 被判大 service_time、被 SJF 压后。
  naive 的误差巨大但**均匀**(只是关掉 SJF),learned 的误差虽小但**结构化 → 错排**,比无长度信号更糟。
  **结论:长度感知只在预测器好到能保序时才有用,光降 MAE 没用;粗桶预测器不如不预测。**

### 下一步
batch-2:长尾截断(把长度知识用于截 vLLM max_tokens 出口压重尾 E[S²],而非 SJF 重排)+ expK-B 对照 +
M/G/1 理论叠加。预测器方向:更细的 per-request 回归 / LLM 长度预测,替代粗桶中位数。

## 2026-09-03 — Week 6 batch 2a：长度感知 max-token 截断与效用损失度量

### 要解决的问题与为什么需要

expK-A 证明 oracle 长度有调度价值，但粗桶预测器的结构化误差会主动错排请求。长度知识的第二种用途
不是排序，而是在送入 vLLM 前识别长尾并降低 `max_tokens`，以压低重尾服务时间的二阶矩。这个动作会
减少输出效用，因此延迟收益必须和截断代价一起报告，不能把少生成 token 伪装成无代价的 SLO 改善。

### 关键代码、方法与命令

- 新增默认关闭的 `admission` 配置：`clip_enabled`、`clip_max_tokens`、
  `clip_source={advertised,learned}`、`clip_estimator_path`。learned 截断启用时必须有独立 artifact。
- `OutputClipper` 只读取调度可见的 advertised cap 或预测器特征。只有估计值和真实目标都严格大于
  `clip_max_tokens` 时，才设置独立的 `backend_max_output_tokens`；原始 `max_output_tokens` 继续保存
  用户目标，避免信息和效用损失被覆盖。
- HTTP backend 仅在发请求时读取 effective cap。请求级事实新增 original target、backend cap 和
  `finish_reason`，同时保持旧版 JSONL/CSV 可读。
- 聚合新增 queue-wait mean、cap-applied 比例、实际 `finish_reason=length` 的截断比例，以及被施加
  cap 的请求平均理论削减 token 数；完整性报告和 sweep CSV/JSON 同步新增这些字段。
- `configs/sweeps/expK-B-clipping.yaml` 用 FCFS 隔离截断效应：no-clip 与 learned cap
  1536/1024/512，各重放同 6 个 seed，共 24 点；realistic Poisson rps=1.2，每点 3 repetitions。

### 成功信号、失败诊断与当前结果

- 单测覆盖：默认关闭逐对象 parity、advertised 超阈才截、learned 只用可见特征、backend 发 cap 但
  不覆盖原目标、请求模型边界、记录和效用聚合、reclock 保留 cap、24 点 sweep 装配。
- 兼容性抽查成功读取 Week 1 的 75 条旧 JSONL 和 75 条旧 CSV；expK-B 四组首点解析为预期的
  no-clip/1536/1024/512 和 seed 20250825。
- `uv lock --check`、Ruff lint、Ruff format 和 126 个 pytest 全部通过。本切片尚未运行 expK-B，
  因此没有延迟改善、截断比例或效用损失的真机结论。

### 下一步

启动固定版本 vLLM，运行 expK-B 24 点并保留全部请求级事实和 GPU 样本；按 6 seed 聚合 queue mean/P99、
SLO、吞吐、cap-applied/realized-truncation rate 与 token reduction。只有在同时展示效用损失时，才解释
截断带来的延迟变化。之后再做 M/G/1 定性趋势桥接。
