# 第一周执行顺序

正式 GPU 工作全部在原生 Ubuntu 上完成。每一步通过验收后再进入下一步；失败输出也保留，
不得据此编写性能结论。

## 1. Ubuntu 环境核验与记录

- 检查 Ubuntu 版本、内核、文件系统、NVIDIA GPU、显存、驱动、CUDA 可见性和 Secure Boot。
- 安装 `uv`，用 CPython 3.11.14 创建项目环境。
- 在 Ubuntu 的 ext4 目录中同步仓库并运行锁文件、Ruff、pytest 和 CLI 检查。

验收：`nvidia-smi` 在 Ubuntu 可识别 RTX 4080 Laptop GPU；项目的四项 CPU 级检查全部通过；
环境信息写入版本化文件。

状态（2026-08-26）：已完成。正式工作区为 `/home/jiale/Desktop/SLOServe`（ext4）；锁文件、
Ruff lint、Ruff format、8 个 pytest 测试和 CLI 配置检查均已通过。完成本步骤时尚未安装
PyTorch/vLLM；随后已进入第 2 步。未启动 GPU 服务或性能实验。

## 2. 固定 vLLM、PyTorch、CUDA 用户态和模型版本

- 为 vLLM 建立独立环境，使用 vLLM 随附且兼容的 PyTorch/CUDA 依赖。
- 选定并锁定一个已在本机验证的 vLLM 发布版本，不使用浮动的 `latest` 或 nightly。
- 下载 `Qwen/Qwen3-0.6B`，记录模型 revision/commit 和 tokenizer revision。

验收：版本清单完整；重建命令不依赖未记录的全局 Python 包。

状态（2026-08-26）：已完成。独立 `.venv-vllm` 已安装并固定 vLLM 0.27.1、
PyTorch 2.13.0+cu130；模型与 tokenizer 均固定到 `Qwen/Qwen3-0.6B` commit
`c1899de289a04d12100db370d81485cdf75e47ca`。依赖、CUDA、文件哈希和离线加载检查均通过。

## 3. 跑通真实 vLLM 流式服务

- 先使用保守的单 GPU 参数启动 OpenAI-compatible server。
- 检查 `/v1/models`，完成一个非流式请求和一个流式请求。
- 保存启动命令、标准输出、失败日志和 GPU 显存占用，不做性能宣称。

验收：一条版本化命令可启动服务；流式响应能观察到首个 token/chunk 和结束事件。

状态（2026-08-27）：已完成。`scripts/serve.sh configs/base.yaml` 从版本化配置经
`sloserve serve-command` 生成 `vllm serve` 命令并启动服务；参数固定为 `--dtype bfloat16
--max-model-len 4096 --gpu-memory-utilization 0.5 --max-num-seqs 16 --enforce-eager`，模型与
tokenizer 均固定到 commit `c1899de289a04d12100db370d81485cdf75e47ca`。`/v1/models`、一次非流式
请求、一次流式请求（可见首个 chunk 与 `[DONE]`）均通过；启动日志、显存采样与原始响应保存在
`results/raw/week1-step3-smoke/`。启动前解决了两个 Python 3.11 环境阻塞：flashinfer 在 3.11 的
`array.array[int]` 导入错误（`scripts/patch_flashinfer_py311.py` 幂等修复），以及采样器 flashinfer
JIT 需要 `ninja`（改用 `VLLM_USE_FLASHINFER_SAMPLER=0` 原生采样器规避）。停止服务后无残留进程、
端点拒绝连接、显存回到 14 MiB。仅功能 smoke，未测任何性能指标。

## 4. 实现可测试的异步负载生成器

- 实现固定速率到达，随后再加入泊松和突发到达。
- 使用固定随机种子生成 interactive/batch 请求；请求大小来自 YAML 配置。
- 将 HTTP 流解析、到达过程和请求生成拆分，并先用可控的假后端测试。

验收：相同配置和种子产生相同请求序列；并发、错误和超时路径均有测试。

状态（2026-08-27）：最小可执行切片已完成。固定速率按 `i / request_rate_rps` 生成相对到达时间；
固定种子生成 interactive/batch 类别与对应 token 范围，并由该类端到端 SLO 推导 deadline。预热请求
作为 `total_requests` 之外的前置请求，与正式请求共享 RNG 流和从 0 开始的连续编号。异步分发经公共
`FcfsPolicy` 接口排序，以 `router.max_in_flight` 限制并发、以 `backend.request_timeout_s` 超时，并用
纯内存假后端覆盖 success/error/timeout/cancelled 及外部取消后的容量释放。新增 11 个聚焦测试通过，
全量 26 个测试及锁文件、Ruff lint/format 检查通过。Poisson 与 burst 仍为显式
`NotImplementedError`，真实 HTTP 流解析和完整指标事件属于后续步骤；本步未运行性能实验。

## 5. 建立原始指标事件与结果格式

- 请求级记录 arrival、enqueue、dispatch、first token、completion/error 时间戳。
- 保存输入/输出 token 数、状态、错误类型、策略、配置哈希、重复编号和环境版本。
- 从原始事件计算 TTFT、TPOT、端到端延迟、队列等待和 token 吞吐量；JSONL 为事实源，
  CSV 为派生的便捷格式。

验收：用手工可计算的小样例验证指标；P50/P95/P99、SLO 达标率、失败/超时数量和最长等待
均可从原始文件重算。公平性公式在实现前写入指标说明。

状态（2026-08-27）：最小可执行切片已完成。`docs/METRICS.md` 在实现前固定五个时间戳、
TTFT/TPOT/端到端/排队/吞吐公式、nearest-rank 分位数、SLO 判定及基于分类别达标率的差距和
Jain 指数。新增请求级不可变记录，复用 `RequestClass` 与 `DispatchStatus`，支持稳定配置 SHA-256、
由调用方传入的环境版本、JSONL 事实源逐字段 round-trip、对应 CSV 派生格式，以及由
`MetricsConfig` 控制目录和格式开关。纯函数可计算总体与分类别 SLO 达标率、四种终态计数、
P50/P95/P99、最长等待、吞吐和公平性，并覆盖空输入、全失败、无 TPOT 和零墙钟窗口。6 条手算
记录从 JSONL 读回重算得到预期 P95、`2/6` 总体达标率、1.5 秒最长等待、1.125 output token/s
测试值和 0.9 Jain 指数；这些数仅是公式单元测试数据，不是性能结果。新增 9 个聚焦测试通过，
全量 35 个测试及锁文件、Ruff lint/format 检查通过。本步没有接真实 dispatcher、HTTP、first-token
信号或 GPU 采样。

## 6. 实现外部 FCFS 准入与排队基线

- 在 vLLM 前实现有界队列、并发准入、超时和取消传播。
- FCFS 只决定哪些请求何时被送入 vLLM，不修改或冒充 vLLM 内部 token scheduler。
- 用假后端验证队列顺序，再连接真实 vLLM。

验收：并发请求严格按到达顺序准入；输入队列不会无限增长；超时/取消不会泄漏容量。

状态（2026-08-28）：最小可执行切片已完成。新增外部 `AdmissionQueue`，从现有版本化配置读取
`queue_capacity`、`max_in_flight` 与 `request_timeout_s`；在同一异步锁内预留空闲槽并经公共
`SchedulingPolicy.order()`/`FcfsPolicy` 按 arrival time、sequence ID 选择下一请求。等待队列满时立即
返回新增的 `REJECTED`，不入队、不启动后端。总超时覆盖 enqueue 到终态：排队态移除条目，in-flight
态取消并等待后端；调用方取消与整体关闭同样清理后端任务和槽位。9 个纯内存 `FakeBackend` 聚焦测试及
全量 44 个测试、锁文件、Ruff lint/format 均通过。该组件只控制请求何时送往外部后端，没有修改或
冒充 vLLM 内部 token scheduler；尚未连接真实 HTTP/vLLM、`RequestRecord`/analysis、first-token 或
GPU 采样，也没有产生性能数据。

## 7. 首个端到端 smoke benchmark

- 使用低请求量先预热，再按固定种子运行至少 3 次 FCFS smoke benchmark。
- 同步采集请求指标和 `nvidia-smi`/NVML 的 GPU 利用率、显存、功耗（硬件支持时）。
- 生成完整性报告，明确这只是管线验证，不作为策略性能比较。

验收：一条命令运行 benchmark 并生成带元数据的原始 JSONL/CSV；吞吐量、TTFT、TPOT、
P50/P95/P99、SLO 达标率、失败/超时、等待/公平性字段及 GPU 指标均不存在无说明的缺失值。
