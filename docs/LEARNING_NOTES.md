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
