# SLOServe 指标定义

本文档固定 Week 1 第 5 步离线指标切片的原始记录、公式和边界语义。该切片只处理
外部准入、排队和路由层可观察到的请求事件，不修改或推断 vLLM 内部 token scheduler
的状态。

## 原始请求记录

JSONL 是请求级事实源，每行一条记录；CSV 由同一组记录派生，只用于人工查看和通用工具
读取。记录保存请求标识、类别、实际 token 数、终态、策略、配置哈希、重复编号和由运行方
提供的环境版本。`env_version` 在本步测试中使用显式占位字符串；真实 GPU、驱动、CUDA、
PyTorch、vLLM、模型等值必须在第 7 步从实际运行环境采集并由调用方传入，禁止推测或填造。

五个时间戳都使用秒，且必须来自同一个单调时钟基准：

- `arrival_time_s`：请求按工作负载计划到达外部服务的时刻。
- `enqueue_time_s`：请求被外部准入队列接收的时刻，不得早于 arrival。
- `dispatch_time_s`：请求获得外部并发容量并被发送到后端的时刻，不得早于 enqueue。若请求被
  拒绝，或在排队中超时/取消而从未送入后端，则为 `null`。
- `first_token_time_s`：收到后端第一个输出 token 的时刻。若终态前没有收到 token，则为
  `null`。
- `completion_time_s`：请求到达终态的时刻。success 时表示完整响应结束；error、timeout、
  cancelled 或 rejected 时表示记录到相应错误/超时/取消/拒绝的时刻。有 dispatch 时它不得早于
  dispatch；若有 first token，也不得早于 first token；无 dispatch 时它不得早于 enqueue。

每条终态记录都有 arrival、enqueue 和 completion；dispatch 可选。无 dispatch 时 first token
也必须为 `null`，status 不得为 success。success 记录必须同时有 dispatch、first token，且
`output_tokens >= 1`。这样 rejected、排队中 timeout/cancelled 不需要伪造不存在的后端事件。

Week 6 的截断实验在同一事实记录中追加三个可选字段，并保持此前保存的 JSONL/CSV 可读：

- `requested_output_tokens`：工作负载生成的原始输出目标，截断时不得覆盖。
- `backend_max_output_tokens`：实际发给 vLLM 的 `max_tokens`；小于原始目标时表示外层应用了 cap。
- `finish_reason`：后端返回的停止原因。只有同时满足“应用了 cap”且 `finish_reason=length`，才记为
  实际因外部 cap 截断；仅应用 cap 但模型提前自然停止，不能算实际截断。

`max_tokens` 本身只表示生成上限，并不保证模型生成到该长度。需要把合成目标解释为实际输出长度的
实验必须在 `realistic_length.force_exact_output_tokens=true` 下运行；HTTP backend 此时发送
`min_tokens == max_tokens == effective cap`。该开关默认关闭，以保持早期实验行为不变，并写入 sweep
聚合元数据。未开启该开关的旧 expK-A 只能解释为“目标上限代理”实验，不能称作真实长度 oracle。

## 请求级指标

以下差值均以秒为单位：

- TTFT = `first_token_time_s - arrival_time_s`。
- TPOT = `(completion_time_s - first_token_time_s) / (output_tokens - 1)`，仅当
  `status=success` 且 `output_tokens >= 2` 时有定义。只有一个输出 token 的成功请求以及所有
  非成功请求从 TPOT 聚合中剔除。
- 端到端延迟 = `completion_time_s - arrival_time_s`。
- 排队等待 = `dispatch_time_s - arrival_time_s`，只对实际发生过 dispatch 的记录有定义；未
  dispatch 的 rejected、排队中 timeout/cancelled 不进入排队等待分位数或最长等待。最长等待是
  所有已 dispatch 记录的排队等待最大值，包含 success、error、timeout 和 cancelled，以便已送入
  后端的失败请求排队经历不会被隐藏。

TTFT、TPOT、端到端延迟和排队等待的 P50/P95/P99 只对 success 请求计算；TPOT 还应用上述
`output_tokens >= 2` 条件。空集合的分位数为 `null`。

排队等待均值与分位数使用同一 success 请求集合。截断实验额外报告：cap-applied count/rate、
realized-truncation count/rate，以及被施加 cap 的请求平均
`requested_output_tokens - backend_max_output_tokens`。两个 rate 的分母均为有 Week 6 截断字段的
全部正式 offered requests（包括失败或拒绝）；旧记录没有这些字段时 rate 为 `null`，不能误报为 0。
平均削减量表示配置允许少生成的上限，不等同于模型实际少生成量；必须和 `finish_reason` 指标一起解释。

## 分位数

所有分位数使用 **nearest-rank**，不做线性插值。对升序排列的 `n` 个值和比例 `p`，选择
1-based 排名 `ceil(p * n)` 的值。因此 P50、P95、P99 分别使用 `p=0.50、0.95、0.99`；只要
集合非空，高分位在很小样本上通常就是最大值。这一选择使手算和从原始文件重算完全一致。

## 吞吐量

token 吞吐量 = 所有 `status=success` 请求的 `output_tokens` 之和 / 墙钟窗口，其中：

`wall_clock_window_s = max(completion_time_s) - min(arrival_time_s)`

窗口由全部记录确定，而不只取成功请求，避免隐藏运行开头或结尾的失败。空输入时窗口和吞吐量
均为 `null`；非空输入但窗口为 0 时，吞吐量也为 `null`，因为不能除以 0。全失败但窗口大于
0 时，成功输出 token 数为 0，吞吐量为 `0.0`。

## SLO 达标、终态计数与公平性

每个请求按其 `RequestClass` 使用 `WorkloadConfig` 中对应 profile 的阈值。一个请求算 SLO
达标，当且仅当以下条件全部成立：

1. `status=success`；
2. TTFT `<= ttft_slo_ms / 1000`；
3. 端到端延迟 `<= end_to_end_slo_ms / 1000`。

边界值按达标处理。error、timeout、cancelled 和 rejected 一律为 SLO 未达标，即使取消前已经收到 first
token。输出总体及 interactive、batch 分类别的请求数、达标数、达标率，并分别报告 success、
error（失败）、timeout、cancelled 和 rejected 数量。这五类终态计数之和必须等于请求总数。
总体或某一类别没有请求时，其达标率约定为 `0.0`；
必须结合该项的 `request_count=0` 解读为“无观测”，不能解释成实际测得的 0% 服务水平。

公平性只基于**数据中实际出现过的** `RequestClass` 的 SLO 达标率 `x_i` 离线计算，不另设
代理指标。空类别（该轮没有任何请求）不参与公平性，也不出现在分类别报告里——否则会把“无观测”
误当成 0% 达标而虚报不公平（例如纯 interactive 负载）：

- 每个**出现过的** `RequestClass` 的达标率都单独报告（按 `RequestClass` 顺序，保证确定性）。
- 达标率差距 = `max(x_i) - min(x_i)`，0 表示各出现类别达标率完全相同；没有任何请求时约定为 `0.0`。
- Jain 公平性指数：`J = (sum(x_i))^2 / (n * sum(x_i^2))`，`n` 为**出现过的类别数**（0、1 或 2）。
  若所有 `x_i=0`（含空输入、全失败，或没有出现任何类别），定义 `J=1.0` 以避免 `0/0`；这只是
  公式边界约定，不代表已获得公平性的实验证据。Jain 指数作为单一标量报告。

## 确定性配置标识与文件写入

`config_hash` 对完整 `ExperimentConfig` 的 JSON 模式稳定序列化计算 SHA-256：字段名排序、紧凑
分隔符、UTF-8 编码，枚举和路径使用 JSON 值。同一配置对象得到同一 64 位十六进制哈希，任何
已序列化字段变化都会改变哈希。

统一写入入口必须使用 `MetricsConfig.output_directory` 决定目录，并分别服从 `save_jsonl` 和
`save_csv`。JSONL 读回后每个 dataclass 字段（包括复用的 `RequestClass` 和 `DispatchStatus`）
必须与写入前相等。CSV 列与 JSONL 字段一一对应，空的 `error_type`、dispatch 或 first-token 值写为空字段；
CSV 始终是可从同一内存记录重建的派生格式，不替代 JSONL 事实源。
