# SLOServe 项目计划

## 1. 项目定位

项目名称：**SLOServe：面向混合长短请求的 LLM 推理调度与性能优化**

目标是在真实 LLM 推理引擎上实现并评测一个面向服务等级目标（SLO）的请求调度层，展示以下能力：

- Linux 与 GPU 推理环境部署；
- Python 异步服务与并发请求处理；
- LLM 推理性能指标采集和分析；
- 调度算法设计、实现和公平性处理；
- 可复现实验、技术写作和工程规范。

项目面向东南大学周睿婷老师课题组的 AI Infra、LLM 分布式推理、资源调度研究方向，同时服务于后续研究生申请和 AI Infra 实习简历。

## 2. 核心研究问题

在线推理服务通常同时处理两类负载：

- **交互请求**：输入和输出较短，对 TTFT、P95/P99 延迟敏感；
- **批处理请求**：输入或输出较长，更关注吞吐量，可以容忍一定等待。

默认 FCFS 策略可能让短请求排在长请求之后，导致交互请求尾延迟升高。本项目研究：

> 能否通过请求长度、等待时间和 SLO 共同驱动的调度策略，在不过度牺牲总体吞吐量的前提下，降低交互请求的尾延迟和 SLO 违约率？

## 3. 项目范围

### 3.1 MVP：必须完成

1. 部署一个可提供 OpenAI 兼容接口的 vLLM 服务；
2. 实现异步负载生成器，支持固定速率、泊松到达和突发流量；
3. 实现并比较三种策略：
   - FCFS；
   - Static Priority：交互请求静态优先；
   - SLO-Aware：结合预计计算量、剩余时间和等待时长动态排序；
4. 自动记录 TTFT、TPOT、端到端延迟、吞吐量和请求完成状态；
5. 采集 GPU 利用率和显存占用；
6. 完成参数扫描、结果绘图、README 和技术报告。

### 3.2 扩展目标：MVP 完成后再做

- 两张 GPU 上各部署一个 vLLM 实例；
- 比较 Round Robin、最短队列和预计完成时间路由；
- 加入 KV Cache/共享前缀感知调度；
- 尝试 vLLM 自定义 scheduler；
- 向 vLLM 提交文档、测试或小型修复 PR。

自定义 scheduler 接口目前不是稳定公共接口，因此首版优先采用“外部调度代理 + vLLM 内置能力”，避免把项目卡在框架内部兼容问题上。

## 4. 推荐实验环境

- 操作系统：Ubuntu 22.04 或更新版本；
- Python：3.11；
- GPU：NVIDIA GPU，建议至少 8 GB 显存；
- 模型：`Qwen/Qwen3-0.6B`，显存允许时再更换更大模型；
- 推理引擎：vLLM，固定具体版本或 Git commit；
- 服务与请求：FastAPI、Uvicorn、aiohttp；
- 数据处理：Pandas、NumPy；
- 绘图：Matplotlib、Seaborn；
- GPU 指标：NVML 或 `nvidia-smi`；
- 测试：pytest；
- 代码质量：Ruff。

最终实验必须记录 GPU 型号、显存、CUDA、驱动、PyTorch、vLLM 和模型版本。

## 5. 系统架构

```text
Request Profiles / Trace
          |
          v
Async Workload Generator
          |
          v
SLOServe Router
  |-- FCFS
  |-- Static Priority
  `-- SLO-Aware + Aging
          |
          v
vLLM OpenAI-Compatible Server
          |
          v
GPU + Model

Metrics Collector
  |-- request latency / TTFT / TPOT
  |-- throughput / SLO attainment
  `-- GPU utilization / memory
          |
          v
CSV/JSON Results -> Plots -> Report
```

## 6. 调度策略设计

### 6.1 FCFS

严格按照到达时间执行，作为基准。

### 6.2 Static Priority

- 交互请求设为高优先级；
- 批处理请求设为低优先级；
- 同一优先级内部保持 FCFS。

### 6.3 SLO-Aware with Aging

为每个请求计算调度分数，分数越小越优先：

```text
estimated_cost = a * input_tokens + b * max_output_tokens
slack = deadline - current_time - estimated_service_time
score = w1 * normalized_cost
      + w2 * normalized_slack
      - w3 * normalized_waiting_time
```

约束：

- 等待时间超过阈值的请求强制提升优先级；
- 记录最长等待时间，验证不存在明显饥饿；
- 所有权重写入配置文件，不硬编码；
- 首版可用 token 数近似计算量，后续再使用离线 profiling 拟合服务时间。

## 7. 请求负载

至少定义两类请求：

| 类型 | 输入长度 | 输出长度 | 目标 |
|---|---:|---:|---|
| Interactive | 64–256 tokens | 32–128 tokens | 低 TTFT、低 P99 |
| Batch | 512–2048 tokens | 128–512 tokens | 高吞吐量 |

工作负载变量：

- 到达过程：固定速率、泊松、突发；
- 请求比例：80/20、50/50、20/80；
- 请求速率：从低负载逐步增加到系统饱和；
- 并发上限：根据显存和模型大小确定；
- 随机种子固定；
- 每个配置预热后至少重复 3 次。

## 8. 评测指标

主要指标：

- TTFT：Time To First Token；
- TPOT：Time Per Output Token；
- 端到端延迟的 P50、P95、P99；
- 输入、输出及总 token 吞吐量；
- Interactive 请求的 SLO 达标率；
- 成功请求数、失败数和超时数；
- 最长等待时间与公平性；
- GPU 利用率和显存峰值。

必须同时报告延迟和吞吐量，不能只展示对本策略有利的单一指标。

## 9. 实验矩阵

### 实验 A：调度策略对比

- FCFS；
- Static Priority；
- SLO-Aware。

比较 P99 TTFT、SLO 达标率、总体吞吐量和最长等待时间。

### 实验 B：负载强度

从低请求速率逐步提高，找到系统开始排队和进入饱和的拐点。

### 实验 C：工作负载比例

测试不同 Interactive/Batch 比例，观察策略是否只在特定混合比例下有效。

### 实验 D：vLLM 参数

扫描：

- `max_num_seqs`；
- `max_num_batched_tokens`；
- chunked prefill 开启/关闭；
- 可选：prefix caching 开启/关闭。

### 实验 E：消融实验

分别移除长度估计、SLO slack 和 aging，解释每一部分的作用。

## 10. 三周里程碑

### 第一周：基线系统

- [ ] 确认 GPU、CUDA、Python 环境；
- [ ] 创建虚拟环境并锁定依赖；
- [ ] 部署模型和 vLLM 服务；
- [ ] 写异步负载生成器；
- [ ] 跑通 FCFS 基线；
- [ ] 正确计算 TTFT、TPOT 和端到端延迟；
- [ ] 保存原始 JSON/CSV 结果。

验收：一条命令启动服务，一条命令运行基准并生成结果文件。

### 第二周：调度实现

- [ ] 实现统一策略接口；
- [ ] 实现 Static Priority；
- [ ] 实现 SLO-Aware；
- [ ] 实现 aging 和超时处理；
- [ ] 增加 GPU 指标采集；
- [ ] 为队列排序、aging 和指标计算编写测试；
- [ ] 完成小规模正确性实验。

验收：三种策略可以通过配置切换，测试全部通过，无明显请求饥饿。

### 第三周：完整评测与交付

- [ ] 执行实验 A–E；
- [ ] 每组配置至少重复 3 次；
- [ ] 绘制吞吐量—延迟、请求速率—P99、SLO 达标率等图表；
- [ ] 记录失败实验和适用边界；
- [ ] 完善 README；
- [ ] 完成 4–6 页技术报告；
- [ ] 整理简历项目描述和联系导师邮件中的项目摘要。

验收：新机器按 README 能复现实验；图表可由原始数据自动生成。

## 11. 计划中的目录结构

```text
SLOServe/
├── PLAN.md
├── README.md
├── pyproject.toml
├── configs/
├── router/
│   ├── server.py
│   ├── policies.py
│   └── metrics.py
├── workload/
│   ├── generator.py
│   └── profiles.py
├── experiments/
│   ├── run_experiment.py
│   └── run_sweep.py
├── analysis/
│   └── plot_results.py
├── tests/
├── results/
└── report/
```

## 12. 最终交付物

1. 公开或可分享的 GitHub 仓库；
2. 清晰、可复现的 README；
3. 完整 commit 历史；
4. 原始实验数据和自动绘图脚本；
5. 4–6 页技术报告；
6. 3–5 张核心结果图；
7. 一段两分钟以内的演示视频，可选；
8. 简历项目描述和联系导师邮件中的项目摘要。

## 13. 简历描述模板

实验完成后再填写真实数字：

> 基于 vLLM 实现面向混合推理负载的 SLO 感知调度系统，构建异步压测、GPU 监控与参数扫描工具；对比 FCFS、静态优先级和自适应调度策略，在 `<负载条件>` 下将交互请求 P99 TTFT 降低 `<X%>`，同时将吞吐量变化控制在 `<Y%>`，并通过 aging 避免长请求饥饿。

不得在实验完成前填写或推测性能提升数字。

## 14. 风险控制

- **没有本地 GPU**：先在本地完成路由、负载生成器和测试，最终使用实验室或短期 GPU 环境执行正式实验；
- **显存不足**：从 0.6B 模型开始，优先验证方法而不是追求模型规模；
- **vLLM 接口变化**：固定版本，首版不依赖不稳定的自定义 scheduler 接口；
- **实验波动大**：预热、固定随机种子、重复实验并报告均值和误差；
- **项目范围失控**：先完成单 GPU MVP，再考虑多 GPU、KV Cache 和上游 PR；
- **只得到负结果**：保留并解释负结果，分析瓶颈和适用边界，同样具有研究价值。

## 15. 开始前需要确认

- [ ] 可使用的 GPU 型号、数量和显存；
- [ ] Linux 环境是否可用；
- [ ] 每周可投入时间；
- [ ] 当前 Python、Linux、PyTorch 和计算机网络基础；
- [ ] 项目目标优先级：本科科研申请、保研材料还是实习简历。

