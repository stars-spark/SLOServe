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
