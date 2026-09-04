# Week 7 expL：自适应输出截断 cap 设计方案

## 修订记录（2026-09-04）

本次实现修订保留 §5.3 的低负载零截断硬门，但修正了使该硬门按构造不可达的控制律。原始
`results/raw/week7-lowload-depth-probe/probe.json` 在 `rps=0.08` 下测得 `Q>=1` 占 1.7148%、
`Q>=2` 占 0.6154%、非 L0 占 5.1155%，并实际施加 6 次 cap；即使把首个阈值抬到 2，低负载仍有
`Q>=2` 瞬态，不能靠阈值消除 Poisson 假阳性。expK-B no-clip 保存的 252 条请求中，目标长度大于
1536/1024/512 的比例分别为 27.38%/41.67%/71.43%，因此一次错误进入 L1 并非空动作。

控制器现增加收紧侧连续证据：`P(t)` 在当前档位的下一收紧阈值之上连续保持
`adaptive_clip_tighten_hold_s` 后才允许进入该档，跌回对应阈值下立即重置；多个档位各自计时，达成时
仍可直接跳到自身证据已满足的最紧档。默认 `8.0s` 来自实际 episode 分离：同一低负载 CPU 探针的
4.611161s、P95/max 为 6.151278s；expK-B `rps=0.17` no-clip 的每 repetition 主导 `Q>=1` episode
P50 为 13.324348s、P95/max 为 43.151063s。8 秒高于低负载观测上界，同时比高负载典型主导 episode
短 5.324348 秒。修改后相同低负载序列保存在
`results/raw/week7-lowload-depth-probe/probe-after-tighten-hold.json`：6 次降为 0 次，非 L0 时间降为
0，深度分布不变。两份都是基于 expK-B 服务时间拟合的确定性 CPU 模拟，不是真机测量。

以下为此前同日修订记录：

本次修订不改变方案的外部控制边界与预注册失败门，只吸收对
`results/raw/week6-expK-B/` 已保存请求级 JSONL 的离线预检事实：在 `rps=0.17` 下，no-clip 的稳定
等待深度峰值仅为 5/3/3，`Q>=8` 的时间占比为 0；clip512 又把 `Q>=2` 的时间占比从 10.6% 压到
0.8%。据此把不可触发的队列阈值重新标定到实测 0–5 动态范围，重新解释 EWMA 与慢放松的角色，把
档位驻留和反向切换周期提升为一等验收指标，并把可由确定性 CPU 测试证明的低负载命题移出 GPU
预算。正式矩阵由 30 点缩减为 21 点。本节引用的是 expK-B 既有原始数据的离线重建结果，不是 expL
的新实验结果或本方案提出的目标值。

本文定义 Week 7 的实现与实验契约。收紧侧迟滞、配置、CPU 回归和低负载探针现已实现并通过本地
验证；本文仍不包含 expL 真机实验结果。尚未执行的正式实验点和 GPU 验收门槛仍待真机证伪。

现状依据如下：`src/sloserve/router/clipping.py` 中的 `OutputClipper` 在 `clip_enabled: false` 时直接返回原 `RequestEnvelope`；固定截断只设置 `backend_max_output_tokens`，不覆盖原始 `max_output_tokens`。`src/sloserve/experiments/benchmark.py` 当前在创建 `AdmissionQueue` 之前对整批请求调用 clipper。expL 仍然只允许修改外部准入、排队和路由层，不修改 vLLM 内部调度器，也不尝试对已发往后端的生成做中途改 cap 或抢占。

历史锚点取自 `docs/REPORT.md` §J/§K/§L、`docs/PROGRESS.md` 和 `docs/LEARNING_NOTES.md` 末尾三条。Week 6 (expK-B) 已证明固定 cap 有效：cap=512 使平均排队等待从 2.78s 降到 0.26s（10.6x），interactive SLO 从 0.26 升到 0.73；代价是 43.1% 请求被平均砍掉 1019 token，token 吞吐降 36%。机制上，E[S^2] 降 2.94x，而平均服务时间只降 1.75x。Week 5 则证明“自适应”本身并不保证正确：`clamp(margin * median_queue_wait, floor, cap)` 的 interactive SLO 为 0.39，远低于固定方案的 0.83-0.91，且最长排队反而最长。expL 必须针对这个失败机制设计，而不能只更换被控制的参数名称。

## 1. 拥塞信号选择

### 1.1 选择标准与观测边界

本方案提出先把“拥塞”限定为外部准入队列承受的、尚未送入 vLLM 的工作积压。可观测量只能来自 `AdmissionQueue` 已有的外部状态、单调时钟、请求到达和完成事件；不读取、不修改 vLLM 内部 token scheduler、KV cache 或批调度状态。

候选信号按以下标准比较：

1. **方向正确**：Poisson 突发增加真实积压时，信号应不下降；积压持续消失时，信号才应下降。
2. **不被事件数量稀释**：同一时间段涌入更多新请求，不应仅因这些请求当前等待时间为零而把拥塞统计量拉低。
3. **反应及时**：收紧必须在排队尾部继续膨胀前发生；仅在请求完成后才得到的信号存在天然迟滞。
4. **采样定义稳定**：Poisson 到达使事件间隔不均匀，因此按事件次数做 EWMA 会过度加权突发中的密集事件；优先使用按真实时间更新的状态量。
5. **闭环可辨识**：cap 会改变服务时间和完成速率。若信号的分母本身又被 cap 改变，控制器容易形成自激反馈，事后也难区分负载变化与控制动作。
6. **可逐请求审计**：每次 cap 决策必须能从请求级原始事实或独立决策日志复算，不能只留下聚合曲线。

这里的“队列深度”必须精确定义。`AdmissionQueue.submit()` 会先把请求加入 `_waiting`，随后在同一把 `asyncio.Condition` 锁下调用 `_admit_available_locked()`。如果把这个瞬时的 append 当作一次深度为 1 的拥塞观测，那么每个低负载、可立即 dispatch 的请求都会制造假拥塞。因此，本方案提出观测**稳定外部等待深度** `Q(t)`：在同步填满所有可用 external in-flight slot 后仍留在 `_waiting` 的请求数。控制器更新和 cap 决策都在同一条件锁保护下进行；可立即 dispatch 的请求留下 `Q(t)=0`，只有没有外部 slot 可接纳的请求才形成正深度。

### 1.2 expK-B 稳定深度的既有离线事实与复算方法

以下分布由评审从 `results/raw/week6-expK-B/` 的现成请求级 JSONL 离线重建，属于运行 expL 之前
已经存在的事实，不是控制器阈值的设计输入伪装成实验结果。三种子合并后的墙钟时间占比如下；各行因
显示精度可能不能严格加总为 100%。

| expK-B arm | `Q=0` | `Q=1` | `Q=2` | `Q=3` | `Q=4` | `Q=5` | `Q>=2` | `Q>=4` | `Q>=8` | 各 seed 峰值 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| no-clip | 84.9% | 4.5% | 3.2% | 2.3% | 3.2% | 1.8% | 10.6% | 5.0% | 0.0% | 5 / 3 / 3 |
| clip512 | 97.3% | 1.8% | 0.2% | 0.6% | 0.0% | 0.0% | 0.8% | 0.0% | 0.0% | 3 / 3 / 1 |

可复算过程必须固定为：

1. 按 arm、seed、repetition 分组读取正式请求记录；对每条已 dispatch 请求取
   `enqueue_time_s=e_i` 与 `dispatch_time_s=d_i`，先硬校验二者存在且 `e_i<=d_i`。
2. 把每条等待区间写成两个状态事件 `(e_i,+1)`、`(d_i,-1)`。按时间排序；同一时间戳的增减先合并，
   因为零长度区间不贡献墙钟权重。扫描事件时，在应用当前时间戳的增减之前，把
   `[t_prev,t_current)` 的时长累计到当时的整数深度 `Q`。
3. 每个 repetition 的观测窗取首个 enqueue 到最后一个 dispatch。深度 `q` 的时间占比为
   `sum(duration with Q=q) / sum(observation-window duration)`；跨三种子合并时先累加各窗的时长和
   分母，再求比例，不能把不同时长运行的百分比直接等权平均。
4. 同一次扫描还应输出 `Q>=k` 的累计时长、每个 seed 的峰值，以及连续 `Q>=1`、`Q>=2` episode 的
   起止时间。后两项是 expL 分析档位驻留和自我熄火所需的审计量。任何负深度、未配对 dispatch 或
   repetition 跨界都必须硬失败，不能静默修补。

这个定义等价于 `Q(t)=|{r: enqueue_time_s <= t < dispatch_time_s}|`。结合
`AdmissionQueue` 当前在 `admission.py:134` append、在 `:143` 同锁内调用
`_admit_available_locked()` 的已核对实现事实，可立即 dispatch 的零时长成员不会贡献正深度；因此
稳定 `Q>0` 也意味着 external in-flight slot 已饱和。

### 1.3 候选总览

| 候选信号 | 突发时的理想反应 | 主要优点 | Poisson 场景下的主要风险 | 结论 |
| --- | --- | --- | --- | --- |
| 请求等待时间 EWMA/median | 等待升高后收紧 | 与目标延迟同量纲、直观 | zero-wait 新样本稀释；只有 dispatch/完成后才知道等待；事件频率偏置 | 不作主信号，可作诊断 |
| 稳定外部队列深度 `Q(t)` + 时间 EWMA | slot 用尽后，新增请求直接增加积压 | 状态量方向单调；无需等待请求完成；不以新请求数作分母 | 短暂尖峰、阈值附近抖动；已经 in-flight 的长请求无法追溯截断 | 推荐主信号 |
| 在途请求数 `I(t)` / `max_in_flight` | 并发槽被占满时升高 | 不受 zero-wait 稀释；实现简单 | 很快饱和成常数，不能区分“忙但无队列”和“深度过载” | 仅作解释/保护条件 |
| `rho_hat = lambda_hat / mu_hat` | 到达率超过服务能力时升高 | 有容量含义，可能提前于排队 | 小窗噪声、大窗滞后；低负载完成样本稀疏；cap 改变 `mu_hat`，形成内生反馈 | 对照臂/离线诊断，不作首选 |

### 1.4 候选一：请求等待时间的 median 或 EWMA

设第 `i` 个 dispatch 请求的等待为 `W_i = dispatch_time_i - enqueue_time_i`。可以对 `W_i` 求滑动 median，或做事件驱动 EWMA：

`W_bar_i = alpha * W_i + (1 - alpha) * W_bar_(i-1)`。

它的吸引力是直接对应平均排队等待和尾延迟。但其根本问题是 `W_i` 是**离开等待队列时才完整观测到的结果变量**，不是到达瞬间的积压状态。一个突发刚到时，新请求尚未产生正等待，控制器只能看到旧样本；等积压请求被 dispatch 并贡献较大的 `W_i` 时，控制动作已经落后一个或多个服务周期。

更严重的是，若把新到达且立即 dispatch 的请求也作为样本，则它重现 Week 5 的污染机制。假设统计窗口中已有 `n` 个正等待样本，突发或空闲 slot 又产生 `z` 个 zero-wait 样本；当零样本占到窗口的一半时，median 会直接变成零。对于事件 EWMA，密集到达使同一小段墙钟时间内执行 `z` 次更新，旧的正等待信号权重变为 `(1-alpha)^z`，会随 `z` 指数衰减。也就是说，**到达事件越密集，已有拥塞证据反而遗忘越快**。这与真实需求方向相反。

只对正等待样本做 EWMA 也不能彻底修复：它会产生选择偏差，低负载时没有新样本、信号长时间陈旧；突发开始时依然要等首个排队请求 dispatch 后才更新。按墙钟时间衰减可消除“事件频率加权”，但无法消除等待信号的结果滞后。

因此，本方案提出不再用 median-in-queue-wait 或 per-request wait EWMA 驱动 cap。它们保留为离线诊断：检查深度信号升高后，等待是否按预期升高；如果二者长期背离，说明深度阈值或服务状态需要重新校准。expL 不把该信号作为推荐控制器，也不通过“排除 zero 样本”伪装成已解决 Week 5 根因。

### 1.5 候选二：稳定外部队列深度与时间加权 EWMA

本方案提出把 `Q(t)` 定义为所有可用外部 in-flight slot 已同步填充后，仍在 `_waiting` 中的请求数。`Q(t)` 是分段常数状态。用真实时间常数 `tau` 更新 EWMA，而不是每来一个请求更新固定权重：

`Q_bar(t) = exp(-delta_t/tau) * Q_bar(t_prev) + (1 - exp(-delta_t/tau)) * Q(t_prev)`。

每次 enqueue、dispatch、backend terminal、timeout、cancel 或 close 导致队列状态变化时，先用上一状态持续的真实 `delta_t` 积分，再写入新 `Q(t)`。这样，一毫秒内发生一次事件与一百次事件不会仅因事件数不同而改变历史信号权重。

它在 Poisson 突发下的性质与等待 median 不同：

- 有空闲 slot 时，新请求被同步接纳，稳定深度仍为零。这表示系统尚未形成外部积压，“不收紧”是正确反应，不是信号被稀释。
- slot 已满时，每个不能立即接纳的新请求使 `Q(t)` 增加一。若突发增加 `z` 个到达、同期释放 `d` 个 slot，则稳定深度变化为 `Q_new = max(0, Q_old + z - d)`；在 slot 饱和区间内，它对 `z` 单调不减，没有 median 的分母稀释问题。
- 突发停止后，瞬时深度会随 dispatch 下降；`Q_bar(t)` 仍保存一段时间的拥塞记忆，使 cap 不会因一次 slot 释放立刻放松。
- 单个很短的尖峰可能提高瞬时 `Q(t)`，但对时间 EWMA 的贡献与其持续时间成比例。控制律再用收紧侧
  连续证据、阈值滞回和慢放松处理瞬态与抖动，而不是把统计定义改成容易受事件数污染的请求样本。

仅使用 `Q_bar(t)` 会有平滑迟滞。为避免严重突发在一个 `tau` 内继续堆积，本方案提出使用同一信号族的“双时间尺度压力”：

- `Q_inst`：当前稳定等待深度，用于及时开始收紧侧连续证据计时；
- `Q_bar`：按墙钟时间加权的深度，用于识别持续拥塞并延迟放松；
- 控制压力 `P(t) = max(Q_inst, Q_bar)`，但只有稳定状态进入 `Q_inst`，不记录 append 后立即 dispatch 的瞬态深度。

`max` 在这里不是新的请求样本平均：突发只能保持或提高 `P(t)`，不能通过加入 zero-wait 请求把它压低。缺点也需要保留：第一批已经进入 vLLM 的长请求无法被追溯截断；当它们占满 slot 时，控制器只能对之后排队、尚未 dispatch 的请求生效。这是“不修改 vLLM 内部调度器”边界带来的必然控制迟滞，不能通过文档承诺消除。expL 必须从决策日志测量“首次正深度到首次收紧”的延迟，并在分析中报告。

### 1.6 候选三：在途请求数 `I(t)`

`I(t)` 是 `_RequestState.IN_FLIGHT` 的外部请求数，可归一化为 `I(t) / max_in_flight`。它不会被 zero-wait 样本拉低：突发只会填充 slot，使比值上升或保持满载。因此它在方向上比等待 median 安全。

但 `I(t)` 的动态范围太小。expK-B 使用 `max_in_flight=4`；系统只要同时服务 4 个请求，信号就达到 1，而“4 个在途、队列为空”和“4 个在途、队列很深”得到完全相同的值。vLLM continuous batching 下，满 external in-flight 并不等价于不健康排队；仅按 `I(t)=max_in_flight` 收紧会在尚未形成外部等待时过早牺牲效用，破坏“低负载不截断”的核心主张。

Poisson 突发的具体失效模式是**饱和失真**而非向下稀释：信号快速到顶后不再表达突发规模或持续时间；一次短暂并发重叠也可能与持续过载完全相同。对它做 EWMA 只能平滑 0 到 1 的占用，不能恢复丢失的排队严重度。

因此，本方案提出把 `I(t)` 写入诊断日志，并允许它作为有效性保护条件——例如只有至少一个 slot 被占用时才解释深度——但不单独驱动正式推荐臂。若后续发现 `Q(t)` 与延迟脱钩，`I(t)` 可帮助区分“没有请求”与“后端忙但尚无外部积压”，不能取代 `Q(t)`。

### 1.7 候选四：到达率/容量比 `rho_hat`

一种更主动的信号是按墙钟窗口估计到达率 `lambda_hat`，再除以服务容量滑动估计 `mu_hat`。Poisson 突发会增加单位时间到达计数，因而不会像等待 median 那样被 zero-wait 新请求向下拉；从这个角度看，它规避了一类污染。

但它有三类新的失效模式：

1. **窗口偏差**：短窗口中的 Poisson 计数方差大，`rho_hat` 会在档位间来回穿越；长窗口降低方差，却可能在队列已经形成后才收紧，并在负载下降后继续截断。
2. **稀疏与陈旧容量样本**：低负载时完成事件少，在线 `mu_hat` 可能长期沿用旧温度、旧输出分布或旧 cap 下的容量。机器的降频档容量约 0.230 req/s 是 expK-B 条件下的校准事实，不应被误当成跨功耗状态永久不变的常数。
3. **内生闭环**：收紧 cap 会缩短服务时间、提高观测完成速率，从而提高 `mu_hat`、降低 `rho_hat`，促使控制器放松；放松后长输出又降低 `mu_hat`、促使再次收紧。信号分母被控制动作直接改变，容易产生 limit cycle，且难以判断是负载变化还是控制器自造的容量变化。

若只用固定的 0.230 req/s 作分母，第三个问题减弱，但环境漂移风险增大；若在线重估分母，必须按 cap 档位分层估计并等待足够完成样本，复杂度与 expL 的首轮目标不相称。本方案提出把 `rho_hat` 作为一个对照臂和离线解释量：固定容量版本用于测试“到达率前馈”是否有价值，在线容量版本暂不进入正式推荐实现。任何 `rho_hat` 对照都必须在日志中同时保留 `lambda_hat`、容量来源和当前 cap，不能只记录一个无法复算的比值。

### 1.8 为什么推荐信号不会重蹈 Week 5 覆辙

Week 5 的根因不是“median 这个单词不好”，而是信号的统计方向在突发下错了。`median_queue_wait` 把每个新请求当作一个等权样本；Poisson 突发恰好一次加入大量等待为零的样本，使中位数下降。控制律又让更低中位数产生更低 aging ceiling，于是最需要保持 SLO 区分时反而把更多请求送入 class-blind tier，造成排序 thrash。

本方案提出的时间加权稳定队列深度在数学上排除了这条因果链：

- **没有按请求数量归一化的样本集合。** `Q(t)` 是一个计数状态，不计算“新请求等待值”的 median/mean。
- **在饱和区间对突发规模单调。** 新增而未被 slot 吸收的请求只能使 `Q(t)` 增加；不存在加入更多 zero-wait 项后统计量下降的可能。
- **按时间而不是按事件平滑。** 历史权重由 `delta_t/tau` 决定，不由一小段时间内来了多少请求决定；密集 burst 不能靠多次更新把拥塞记忆冲掉。
- **未排队的新请求不制造假阳性，也不构成稀释。** 有空闲 slot 时稳定 `Q(t)=0`，因为外部系统确实没有等待工作；这与“窗口中加入零值导致已有正等待统计下降”是两件不同的事。
- **控制状态不直接改请求排序。** expL 使用 FCFS 隔离截断效应；cap 只改变尚未发送请求的 backend 工作量，不把请求批量推入 class-blind aging tier，因此不会重现 Week 5 的排序翻转路径。
- **滞回后，下降需要持续证据。** 即使瞬时 `Q_inst` 下降，控制器也只能在 `Q_bar` 低于更低的放松阈值并持续 hold 时间后逐档放松，不能在同一阈值两侧频繁反转。

这不等于预先宣称控制器一定成功。队列深度可能太迟、深度阈值可能误设、首批 in-flight 长请求可能已经决定尾延迟，也可能出现“深度不高但服务时间方差很大”的场景。区别在于这些失败会表现为可审计的响应迟滞或阈值错配，而不会由 zero-wait 新请求把信号本身推向与需求相反的方向。

### 1.9 最终推荐

本方案提出以**稳定外部等待深度 `Q_inst` 与其墙钟时间 EWMA `Q_bar` 构成的队列深度压力**为 expL 的主信号，`P(t)=max(Q_inst,Q_bar)`；在途数与 `rho_hat` 只作诊断或对照。推荐理由是：它直接来自 SLOServe 已拥有的外部准入状态，在 backlog 区间对 Poisson burst 单调，不以 zero-wait 请求作等权样本，能在请求完成前反应，并能在不接触 vLLM 内部调度器的前提下逐请求记录和复算。

## 2. 控制律设计

### 2.1 离散档位而非连续 cap

本方案提出使用有限状态机，而不是把 `P(t)` 连续映射为任意整数 cap：

| 档位 | backend cap | 含义 |
| --- | ---: | --- |
| L0 | no-clip | 保持 `backend_max_output_tokens` 的现有语义，不人为截断 |
| L1 | 1536 | 轻度收紧 |
| L2 | 1024 | 中度收紧 |
| L3 | 512 | 最强允许收紧 |

这些档位复用 expK-B 已经定义过的 no-clip/1536/1024/512 边界，便于比较；这只是设计选择，不暗示 1536 或 1024 在 expL 中会产生任何特定结果。512 是本方案提出的最小可用 cap，控制器不得低于它。2048 是本次 realistic workload 的原始上限；高于它不产生有效截断，因此 L0 必须表示真正的 no-clip，而不是写入一个更大的“伪 cap”。

固定 cap 的长度资格判断继续沿用 `OutputClipper`：只有可见长度估计和原始目标都严格大于当前数值 cap 时，才设置 `backend_max_output_tokens`；原始 `max_output_tokens` 永远保留。L0 不设置新 cap。

### 2.2 阈值、滞回与不对称速度

原默认收紧阈值 `[2.0,4.0,8.0]` 与 expK-B 的实测动态范围不匹配：no-clip 的 `Q>=4` 仅占
5.0%，`Q>=8` 为 0.0%，三个 seed 的峰值又只有 5/3/3。于是 L3 永不可达，expK-B 唯一已经证明
有效的 512 档不会启用；按 `Q_inst` 看，控制器约 84.9% 时间在零深度，另有 4.5% 时间只有深度 1，
旧 L1 门槛也会忽略这部分压力，合计约 89.4% 时间不具备进入 L1 的资格；旧 L2 只有约 5.0% 时间
具备资格。这属于**由配置制造的空结果**，必须在 GPU sweep 前修正，不能等实验结束后才归因于信号
无效。

本方案重新提出默认 queue-depth 参数：收紧阈值 `[1.0,2.0,3.0]`，放松阈值
`[0.25,0.75,1.5]`，墙钟 EWMA 时间常数 `6.0s`，连续高压收紧保持时间 `8.0s`，连续低压放松保持
时间 `30.0s`。三个收紧阈值全部落在
既有 0–5 动态范围内。下表直接由 §1.2 的边际分布求和；它表示 `Q_inst` 达到对应门槛的**触发资格
时间占比**，不是对尚未运行的 adaptive arm 档位驻留比例的预测。由于 `P=max(Q_inst,Q_bar)`，EWMA
与两侧 hold 会改变切换资格和档位记忆，实际驻留必须由 decision sidecar 测量。

| 默认目标档位 | 收紧门槛 | no-clip 中 `Q_inst` 达标 | clip512 中 `Q_inst` 达标 | 可达性结论 |
| --- | ---: | ---: | ---: | --- |
| L1 / cap1536 | `P>=1` | `Q>=1`：约 15.1% | `Q>=1`：约 2.7%（分项舍入和约 2.6%） | 两臂均可达 |
| L2 / cap1024 | `P>=2` | 10.6% | 0.8% | 两臂均可达，但收紧后资格显著减少 |
| L3 / cap512 | `P>=3` | `Q>=3`：7.3% | `Q>=3`：0.6% | no-clip 三个 seed 峰值均达 3；clip512 三个 seed 中两个可达 |

保守队列臂也不能保留超出实测范围的 `[4,8,16]`。其收紧阈值改为 `[2.0,3.0,4.0]`、放松阈值
改为 `[0.5,1.5,2.5]`：no-clip 下三档资格时间分别为 10.6%、7.3%、5.0%。L3 确实可达，但按
5/3/3 的峰值只会在峰值 5 的 seed 出现；clip512 下 `Q>=4` 为 0，因此该臂有意形成“可达但稀疏”的
保守对照，而不是再次设置一个整个实验都不可能触发的档位。

`tau=2.0s` 也不再合适。no-clip 有 84.9% 墙钟时间 `Q=0`，且 `rps=0.17` 对应平均到达间隔
`1/0.17=5.88s`；若一段时间内深度回到零，2 秒 EWMA 经过一个平均到达间隔只保留
`exp(-5.88/2)=5.3%` 的旧压力。此时 `Q_bar` 显著低于上升沿的 `Q_inst`，`P=max(...)` 的收紧几乎
完全由 `Q_inst` 主导。**上升沿由瞬时深度主导是期望行为**，因为它提供快速收紧；但若 EWMA 在一次
典型到达间隔内几乎清空，它就无法承担“慢放松的拥塞记忆”职责。默认 `tau` 因此调到 `6.0s`，约等于
高负载一个平均到达间隔，此时同样间隔后仍保留 `exp(-5.88/6)=37.5%` 的旧压力。EWMA 仍不负责压过
`Q_inst` 的上升沿，只负责让下降沿有可观测记忆；保守臂用 `12.0s` 检验更长记忆的效用代价。

- 每个更紧档在 `P(t)` 达到自己的收紧阈值时独立开始计时；连续保持 `8.0s` 才具备进入该档的资格，
  期间跌回该阈值以下便清零该档证据。一次更新直接跳到自身证据已达成的最紧档，严重且持续的 burst
  不必逐档等待；刚刚才越过的更高阈值不能借用低档已经累计的时间。
- 放松采用更低阈值，且 `P(t)` 必须连续低于目标放松阈值达到 `30.0s`；每次最多放松一档，随后重新计时。从 L3 完全回到 L0 至少经历三次独立 hold，而不是一次计时连续跨三级。
- 收紧仍快于放松，但不再是纯瞬时：8 秒过滤低负载 Poisson 瞬态，30 秒放松 hold 防止控制动作
  自我熄火。代价是高负载首次收紧至少晚 8 秒，并在离散事件驱动下最晚到下一个更新才被观察，即
  延迟上界为 `tighten_hold + one update interval`。expL 用“首次正深度到首次收紧”、queue P95/P99、
  interactive SLO 和相对 fixed-512 的等待/SLO 收益保留率共同检验这项反应速度损失。
- 决策只在队列状态事件下执行，所有状态更新都在同一条件锁内；同一请求只在 dispatch 前决定一次，发送后不再变更。

控制器必须记录 `decision_time_s`、`Q_inst`、`Q_bar`、旧档位、新档位、选定 cap 和触发原因。收紧侧
原因枚举为 `TIGHTEN_HOLD_STARTED/PENDING/RESET/ELAPSED`，与放松侧
`RELAX_HOLD_STARTED/PENDING/RESET/ELAPSED` 对称。分析应计算每次收紧延迟、档位驻留时间和相邻反向
切换次数。若出现刚收紧便放松的频繁反转，视为控制 thrashing，而不是用聚合均值掩盖。

### 2.3 瞬态误收紧、自我熄火与两侧 hold 的依据

原低负载探针已把瞬态误收紧从假设变成事实：`Q>=1` episode 的 P50 为 4.611161 秒，P95/max 为
6.151278 秒；旧瞬时控制下非 L0 episode 的 P50 为 0.794899 秒，但 P95/max 达 42.797650 秒，长尾
来自 30 秒放松记忆，而不是持续高压。默认收紧 hold 取 8 秒，覆盖所有已观测低负载正深度 episode。
高负载侧直接重建 expK-B 三 seed × 三 repetition 的正式 no-clip 等待区间：每个 repetition 最长的
主导 `Q>=1` episode 为 10.422251–43.151063 秒，nearest-rank P50 为 13.324348 秒。因此 8 秒仍让
典型高负载积压撑过；反应代价相对瞬时控制固定增加至少 8 秒、至多再加一个更新间隔。

expK-B 给出的反事实量级很强：no-clip 的 `Q>=2` 占 10.6%，而固定 clip512 下只占 0.8%；`Q=0`
又从 84.9% 升到 97.3%。这说明强收紧一旦减少后端工作量，驱动信号本身会迅速塌向零。若仍用
`tau=2s`、`hold=10s`，控制器很可能把“本次控制动作刚刚有效”误判为“外生拥塞已经结束”，随即
放松并重新注入长尾，形成收紧→放松→再收紧的自我熄火/极限环。慢放松不是实现细节，而是方案能否
成立的关键控制条件。

高负载全部 177 个 `Q>=1` episode 中包含可立即 dispatch 所形成的几十微秒记录间隔；保留这些原始
间隔时 P95 为 10.422251 秒、max 为 43.151063 秒。默认 `adaptive_clip_relax_hold_s=30s` 高于该实测
P95，但不覆盖最大 episode；它仍是防止短暂归零自我熄火的保守记忆，而不是保证覆盖每个极端事件。
若从 L3 开始，逐档重新计时使完全放松至少需要 90 秒。代价是拥塞真正结束后可能继续截断，因此低
负载确定性零截断门和真机吞吐等价区间必须共同约束它。

### 2.4 控制时点与外部边界

现有固定 clip 在生成请求后、建队列前应用；此时还没有真实队列压力。自适应 cap 必须在 `AdmissionQueue._admit_available_locked()` 选中请求后、创建 `backend.send()` 任务前决定。由于这是外部 dispatch 边界，它只改变即将发送的 `RequestEnvelope.backend_max_output_tokens`。

本方案明确排除三类做法：修改 vLLM scheduler、读取 vLLM 内部队列来驱动控制、对正在生成的请求中途停止或改 cap。已经 in-flight 的请求不可追溯控制是本方案的已知限制，必须在结果中报告而非隐藏。

## 3. 配置字段设计

### 3.1 `configs/base.yaml` 的新增字段

字段继续放在现有 `admission:` 下；当前实现的默认值如下：

```yaml
admission:
  clip_enabled: false
  clip_max_tokens: 2048
  clip_source: advertised
  clip_estimator_path: null
  adaptive_clip_enabled: false
  adaptive_clip_signal: queue_depth_ewma
  adaptive_clip_caps: [1536, 1024, 512]
  adaptive_clip_tighten_thresholds: [1.0, 2.0, 3.0]
  adaptive_clip_relax_thresholds: [0.25, 0.75, 1.5]
  adaptive_clip_ewma_tau_s: 6.0
  adaptive_clip_tighten_hold_s: 8.0
  adaptive_clip_relax_hold_s: 30.0
  adaptive_clip_capacity_rps: null
```

字段契约如下：

| 字段 | 类型/校验 | 默认值 | 作用 |
| --- | --- | --- | --- |
| `adaptive_clip_enabled` | `bool` | `false` | 总开关；默认整体不生效 |
| `adaptive_clip_signal` | enum：`queue_depth_ewma`、`in_system`、`rho_hat` | `queue_depth_ewma` | 选择实验信号；推荐值为前者 |
| `adaptive_clip_caps` | 3 个严格递减正整数 | `[1536, 1024, 512]` | L1-L3 cap；末项也是最小可用 cap |
| `adaptive_clip_tighten_thresholds` | 3 个严格递增有限非负浮点数 | `[1.0, 2.0, 3.0]` | L0→L1→L2→L3 的收紧门槛；按 expK-B 既有 0–5 动态范围标定 |
| `adaptive_clip_relax_thresholds` | 3 个有限非负浮点数，逐档小于对应收紧门槛 | `[0.25, 0.75, 1.5]` | 滞回放松门槛 |
| `adaptive_clip_ewma_tau_s` | 有限正浮点数 | `6.0` | 墙钟 EWMA 时间常数；约一个高负载平均到达间隔 |
| `adaptive_clip_tighten_hold_s` | 有限非负浮点数 | `8.0` | 收紧前连续高压保持时间；由低负载 P95/max 6.151278s 与高负载主导 episode P50 13.324348s 分离得到 |
| `adaptive_clip_relax_hold_s` | 有限非负浮点数 | `30.0` | 放松前连续低压保持时间；约两倍典型 episode 中心尺度 |
| `adaptive_clip_capacity_rps` | 正浮点数或 `null` | `null` | 仅 `rho_hat` 对照需要；推荐信号不得依赖它 |

`clip_source` 和 `clip_estimator_path` 继续决定哪些长度信息可用于判断请求是否值得截断；expL 与 expK-B 一样使用 learned source 和独立 predictor artifact，不能读取未来真实输出。`adaptive_clip_enabled: true` 时要求 `clip_enabled: true`，并校验最小 cap 不低于 512、最大数值 cap 不高于 workload 的有效上限。后一个跨配置校验若不能通用于所有 workload，则以“不得高于该 workload 的 advertised/clamp 上限”为 expL sweep 校验，属于本方案提出、待评审确认的实现选择。

### 3.2 默认关闭与严格 parity

“默认关闭”不只表示结果大致相同。本方案提出以下兼容契约：

1. `adaptive_clip_enabled: false` 时不构造控制器，不增加 clock 调用，不改变 RNG 调用，不改变 queue mutation 次序。
2. 关闭自适应且 `clip_enabled: false` 时，仍由当前 `OutputClipper.apply()` 直接返回同一对象。
3. 关闭自适应且固定 `clip_enabled: true` 时，继续走现有“建队列前一次性 clip”的路径；不能借 Week 7 重构顺便改变固定 cap 的判断时点。
4. 新字段（包括 `adaptive_clip_tighten_hold_s`）在关闭状态下从 legacy config-hash 投影中排除，避免
   仅因新增默认字段改变历史 `config_hash`；基准哈希继续为
   `520091cbcb2669534c962b25c86c8f6fd23e7cd1e3191f7afe36a672b53201a6`。
5. 自适应决策字段写入独立的 `*-adaptive-cap-decisions.jsonl`；关闭状态不创建该 sidecar。现有 request JSONL/CSV 的字段顺序、缺省键和数值格式保持不变。

parity 测试采用确定性 fake backend、fake clock、相同 seed 和完全相同 trace，在修改前保存 no-clip 与固定 cap=512 两份 golden JSONL。修改后分别运行关闭路径，对完整文件做 `cmp` 和 SHA-256 校验；同时逐对象断言 request identity、dispatch 顺序、有效 cap、状态和所有时间戳一致。任何一个字节或逐请求字段不同都算失败。真实 GPU 运行含不可控计时，不适合做逐字节 parity 证明，只用于补充请求语义抽查。

## 4. 实验设计 expL

### 4.1 正式对照臂

本方案提出 5 个正式 arm，统一使用 FCFS、`max_in_flight=4`、realistic heavy-tailed workload、`force_exact_output_tokens: true` 和与 expK-B 相同的 learned predictor 信息边界：

1. `no-clip`：无截断基线。
2. `fixed-512`：当前最优固定方案基线。
3. `adaptive-q-default`：推荐的队列深度压力，使用第 2 节默认阈值。
4. `adaptive-q-conservative`：同一信号但按下表提高收紧门槛、延长放松证据，用于检验低负载效用保护与高负载反应迟滞的权衡。
5. `adaptive-rho-fixed-capacity`：按墙钟到达率 EWMA 除以显式容量常数的对照；容量来源记录为本机降频档约 0.230 req/s，不在线用完成速率回授，以避免 cap→容量估计→cap 的内生环。

为避免看到结果后再调参，本方案提出在生成正式 sweep 前冻结以下 override；这些数值是实验自变量，不是实验结果：

| adaptive arm | signal | tighten | relax | EWMA `tau` | tighten hold | relax hold | capacity |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| `adaptive-q-default` | `queue_depth_ewma` | `[1.0, 2.0, 3.0]` | `[0.25, 0.75, 1.5]` | `6.0s` | `8.0s` | `30.0s` | `null` |
| `adaptive-q-conservative` | `queue_depth_ewma` | `[2.0, 3.0, 4.0]` | `[0.5, 1.5, 2.5]` | `12.0s` | `8.0s` | `45.0s` | `null` |
| `adaptive-rho-fixed-capacity` | `rho_hat` | `[0.70, 0.85, 1.00]` | `[0.60, 0.75, 0.90]` | `10.0s` | `8.0s` | `10.0s` | `0.230` |

三个 adaptive arm 都使用 `[1536, 1024, 512]`。`rho_hat` arm 的 `tau` 用于按墙钟统计到达率；分母固定为配置中的容量，禁止在正式运行途中用受 cap 影响的完成率回写。

不设置 median-queue-wait 正式 arm：Week 5 已经给出该污染机制的直接负证据，重复消耗真机预算没有新增辨识价值。若 CPU 仿真/确定性 trace 无法复现其反向响应，应先修测试，而不是增加 GPU 点。

### 4.2 负载点、种子与预算

本方案提出只保留两个负载点：

- `rps=0.08`：低负载效用保护点，目标是直接证伪“控制器会不会在没有持续积压时也截断”。这是设计负载，不是已观测结果。
- `rps=0.17`：复用 expK-B 的高负载比较点，且低于本机降频档约 0.230 req/s 的容量。

每个保留的 arm/负载组合使用 3 个 seed：`20250825`、`11`、`202`；每点仍保留现有
`repetitions: 3`、`total_requests: 24`、`warmup_requests: 4`。不加第三个中负载点；控制拐点先由
决策日志解释，只有正式矩阵结论不清时才另立后续实验，不能事后悄悄混入 expL 主结论。

正式矩阵改为非对称设计：

- `rps=0.17` 保留全部 5 arm × 3 seed = 15 点，因为 fixed-512、两个 queue-depth 参数组与 rho 对照
  都用于辨识高负载延迟—效用 Pareto 和控制机制。
- `rps=0.08` 只保留 `no-clip` 与推荐的 `adaptive-q-default`，即 2 arm × 3 seed = 6 点。低负载的
  随机性问题只是推荐控制器相对 no-clip 的吞吐等价区间；fixed-512 已知会主动截断，保守 queue arm
  与 rho arm 又不是最终“零效用代价”主张的对象，把它们各跑三次不会增强该主张。

因此正式矩阵从 30 点降为 `15+6=21 points`。没有保留任何被删除的低负载 arm；它们仍全部保留在
高负载矩阵中，所以没有丢失 fixed cap 基线、阈值敏感性或替代信号对照。

按本轮预算输入重算：低负载每点约 16 分钟，6 点为约 96 分钟；高负载沿用已知每点约 7–8 分钟，
15 点为 105–120 分钟；纯运行合计 201–216 分钟，即 3 小时 21 分至 3 小时 36 分。原 30 点“含冷却
约 7.6 小时”与纯运行中值约 352.5 分钟之差约 103.5 分钟，折合每点约 3.45 分钟的启动、检查和冷却
开销；按 21 点线性折算约 72.5 分钟。修订后的总预算因此约 274–289 分钟，即 **4 小时 34 分至
4 小时 49 分（计划窗口按约 4.6–4.8 小时，保守预留 5 小时）**。这是从给定单点时长和原 7.6 小时
总预算推导的调度估算，不是新增性能观测；若冷却门实际等待更久，以硬门为准，不为赶预算缩短冷却。

执行顺序按 seed 做区组并在区组内轮换 arm，避免温度或时间漂移总与某一 arm 重合。所有点沿用 expK-B 的功耗档、冷却门和硬停止条件；中断点保留原始数据，不删除，也不与完整点混算。

## 5. 关键指标

### 5.1 延迟收益与效用代价必须联报

每个保留的 arm/负载/seed 同时报告：

- 延迟：`queue_wait_mean_s`、queue P50/P95/P99、`longest_queue_wait_s`、端到端 P50/P95/P99、interactive SLO、batch SLO 和 overall SLO。
- 效用：`clip_applied_rate`、`realized_truncation_rate`、`mean_cap_reduction_tokens`、完成输出 token 总数、`token_throughput_per_s`。
- 健康度：success/error/timeout/cancelled/rejected、GPU 利用率/显存/功耗、每档驻留时间、档位切换次数、首次正深度到首次收紧的响应延迟。
- 机制：按实际输出拟合服务时间，并报告平均服务时间与 E[S^2]；`queueing.py` 的 concurrency-scaled M/G/1 仍只作方向核对，不把绝对预测秒数当成验收标准。

所有均值与区间按 seed/repetition 分层，使用 paired seed 差值或 cluster bootstrap，不能把同一次运行中的请求当作相互独立样本来夸大显著性。

### 5.2 控制稳定性是一等验收指标

档位行为不再只是第 6 节 thrash 失败门的附属诊断。每个 adaptive arm/负载/seed 必须直接报告：

- L0/L1/L2/L3 的墙钟驻留占比，以及每档连续驻留 episode 的 count、P50、P95、max；
- 每种有向切换（尤其“收紧→放松”和“放松→再收紧”）的次数、相邻切换间隔与完整
  `tighten -> relax -> retighten` 周期的 count、P50、P95、min；
- 首次 `Q_inst>0` 到首次收紧的延迟，以及收紧后 `Q_inst/Q_bar` 归零到真正放松的延迟；
- 分别以 `adaptive_clip_tighten_hold_s`、`adaptive_clip_relax_hold_s` 为尺度归一化相应方向的切换
  延迟与周期长度，明确有多少反向周期短于 1×hold、位于 1–2×hold，或长于 2×hold。

验收不只看切换总数。若推荐臂的短周期反向切换反复出现，或大量周期接近 hold 边界，说明闭环正在
自我熄火/形成极限环；即使平均等待或吞吐有利也不能通过。反之，长时间驻留某一档也不能自动算稳定，
必须结合 §2.2 的触发资格占比判断它是合理记忆，还是阈值/hold 导致的锁死。

### 5.3 “低负载没有效用代价”的 CPU/GPU 分工验证

“低负载没有效用代价”拆成两个确定性命题和一个随机性命题，不再让 GPU 重复证明状态机语义：

旧控制律已真实触发过本节硬门：`probe.json` 的 `clip_applied_count=6`，所以“瞬时越阈即收紧”不能
进入 GPU 阶段。硬门本身不修改；修复的是控制器的可达性条件。相同 trace 在 8 秒收紧 hold 下重跑后，
`probe-after-tighten-hold.json` 给出 `clip_applied_count=0`、`non_l0_time_fraction=0`，而
`depth_ge_1_time_fraction=0.01714843218197302`、`depth_ge_2_time_fraction=0.006154110178295506`
保持不变。这说明门槛现在可达是因为瞬态没有撑过持续时间判据，而不是换了更温和的随机序列或掩盖了
正深度。

1. **CPU 硬门 1**：Batch 1/2 用 fake clock 重放固定的 `rps=0.08` trace；推荐配置必须满足
   `clip_applied_count == 0` 且 `realized_truncation_count == 0`，否则禁止进入 GPU 阶段。
2. **CPU 硬门 2**：同一 trace 经 fake backend 后，每条请求必须满足
   `backend_max_output_tokens == requested_output_tokens`，decision sidecar 的档位始终为 L0；按
   `request_id + repetition + seed` 与 no-clip trace 配对后，请求次序、有效 cap 和 fake 输出 token
   总数逐请求一致。这是确定性语义证明，也必须在 GPU 前通过。
3. **GPU 随机性门**：正式低负载只跑推荐 arm 与 no-clip。paired token-throughput ratio 的 95% 区间
   必须落在 `[0.95,1.05]`；这是本方案提出的等价性门槛，不是历史结果。若 CPU 两门通过而该区间
   不通过，应诊断运行时噪声、温度、失败请求或真实到达时序，不能宣称效用无损。

这种拆分不削弱“低负载零效用代价”主张：cap 是否被应用、后端 cap 是否保持原目标、档位是否恒为 L0
都是给定 trace、clock 与状态机下的确定性命题，CPU 可以逐事件严格证明；真实 GPU 仍保留推荐 arm 与
no-clip 的三 seed 配对，检验执行时序与测量噪声下的吞吐等价区间。确定性部分由 CPU 证明，随机性部分
由 GPU 证明，证据边界比用 15 个 GPU 点混在一起更清晰。

GPU 原始验证仍从 request JSONL 重算 finish reason、token totals 和 throughput，再与 sweep CSV 对账；
decision sidecar 用于语义抽查和重放，但不把已经由 CPU 硬门证明的两个命题重新包装成 GPU 预算目标。
分析脚本必须在缺 sidecar、重复决策、请求对不上、聚合与原始事实不一致时硬失败。

### 5.4 高负载的 Pareto 目标

高负载不要求自适应 arm 在延迟上超过 fixed-512，因为 fixed-512 始终支付最大效用成本。目标是形成 no-clip 与 fixed-512 之间可解释的 Pareto 点：相对 no-clip 明确改善平均排队等待和 interactive SLO，同时相对 fixed-512 减少截断或恢复 token 吞吐。只报告延迟改善、不同时展示效用恢复，或只报告吞吐恢复而延迟退回 no-clip，都不构成 Week 7 成功。

## 6. 预期失败模式与诚实证伪

以下阈值均为本方案提出的预注册失败规则，不是对结果的预测：

1. **低负载零效用门失败**：CPU 固定 trace 出现任意 cap applied/realized truncation、任一后端 cap
   不等于请求目标或任一档位不为 L0，立即停止，不能用 GPU 结果覆盖确定性错误。CPU 两门通过后，
   GPU paired throughput ratio 的 95% 区间越出 `[0.95,1.05]`，核心主张仍失败。
2. **阈值或 hold 造成空结果**：GPU 前用 §1.2 重建的 no-clip trace 回放 queue-depth 参数；若默认
   L1/L2/L3 任一档没有非零越阈资格，或高负载持续 episode 全部撑不过 tighten hold，说明配置未覆盖
   已知动态范围，禁止进入正式 sweep。正式 adaptive 运行中 L3 驻留为零不自动算失败，因为收紧本身
   可能把深度压低；必须联合 `Q_inst/Q_bar` 越线、`TIGHTEN_HOLD_*` 轨迹与低档控制效果解释，不能把
   “没有越阈”“越阈但未持续”和“触发后有效”混为一谈。
3. **高负载没有延迟收益**：相对 no-clip，平均排队等待下降或 interactive SLO 上升的 paired 95%
   区间包含零，则不能声称稳定收益；两者任一方向变坏则记为明确负结果。
4. **相对 fixed-512 保留收益不足**：定义等待收益保留率
   `(W_no_clip - W_adaptive) / (W_no_clip - W_fixed512)` 和 SLO 收益保留率
   `(S_adaptive - S_no_clip) / (S_fixed512 - S_no_clip)`。任一分母非正时该项不可解释并触发诊断；
   否则推荐 arm 的两项保留率若任一低于 `0.75`，视为高负载控制过松。
5. **没有恢复效用**：相对 fixed-512，推荐 arm 的 token throughput 未提高且 realized truncation
   rate 未下降，说明自适应只增加复杂度、没有形成新的 trade-off 点。
6. **控制自我熄火或 thrashing**：tighten→relax 间隔短于 `adaptive_clip_relax_hold_s`，或
   relax→tighten 间隔短于 `adaptive_clip_tighten_hold_s`，属于实现错误；档位切换数超过正式
   dispatch 数的 `10%`，或
   §5.2 的 `tighten -> relax -> retighten` 短周期反复出现，属于控制不稳定，即使聚合延迟好看也
   不能通过。必须报告驻留时间和周期分布，不能只给切换总数。
7. **方向异常**：Poisson burst 中 `Q_inst` 增加而记录的 `P(t)` 或目标档位变松，说明稳定深度定义、
   更新时间或锁内顺序有 bug；不得继续跑完整 GPU sweep。

若自适应再次输给固定 cap，优先按以下假设定位：

- **控制太迟**：首批长请求已经占满 in-flight slot，排队形成后才有正 `Q`，或 8 秒 tighten hold 吃掉
  了可干预窗口；看“首次正深度→首次收紧”、`TIGHTEN_HOLD_*` 与这些请求的 dispatch 时间。
- **阈值太高/平滑或收紧 hold 太慢**：高负载长期 L0/L1，收益保留率低；看档位驻留、
  `Q_inst/Q_bar` 越线记录，以及越阈 episode 是否在 8 秒前结束。
- **阈值太低/放松太慢**：低负载出现 cap 或吞吐损失；看短 burst 后 L1-L3 驻留时间。
- **深度不是工作量**：少量超长请求与大量短请求得到相同 count，`Q` 低但 E[S^2] 仍高；按 predictor bucket 分层检查等待和被截请求。
- **预测器误判**：控制器正确收紧，但 learned source 没有命中真正长尾；比较 predicted bucket、原目标、实际输出和 finish reason，不能偷偷换成未来真实长度。
- **外生环境漂移**：功耗档、温度、服务版本或 GPU 状态变化造成容量改变；按 seed 区组检查环境 metadata 与 GPU 样本。
- **`rho_hat` 自激或窗口噪声**：若对照臂频繁切换，检查到达窗口、固定容量来源和 cap 档位；不把它归咎为 queue-depth 推荐信号失败。
- **效用度量错位**：cap applied 不等于 realized truncation；必须同时查看 `finish_reason=length`、实际输出和理论 cap reduction。

任何失败都保留 request JSONL、decision sidecar、GPU samples、完整性报告和 sweep 配置。不得删除失败点、修改门槛后只展示更有利的子集，或把减少生成工作量描述为免费调度收益。

## 7. 任务契约拆分

### Batch 1：配置模型与纯控制器（CPU）

本方案提出修改/新增路径：`src/sloserve/config.py`、`configs/base.yaml`、`src/sloserve/router/adaptive_clipping.py`、`tests/test_adaptive_clipping.py`、`tests/test_config.py`。

产出：严格配置校验、时间加权 `Q_bar`、L0-L3 状态机、带连续证据的直接跳档收紧、逐档慢放松、可
注入 clock。测试覆盖 burst 单调性、同墙钟区间不同事件数量得到相同 EWMA、阈值边界、收紧 hold 的
开始/进行中/重置/达成、达成后的 direct jump、逐档放松、每档重新 hold、最小/最大 cap，以及固定
低负载压力 trace 全程保持 L0。

验收：

```bash
uv run pytest tests/test_adaptive_clipping.py tests/test_config.py
uv run ruff check src/sloserve/config.py src/sloserve/router/adaptive_clipping.py tests/test_adaptive_clipping.py tests/test_config.py
```

该批禁止访问 GPU；若确定性 trace 中加入更多 zero-wait arrival 会降低压力或放松档位，立即失败。
推荐配置在原探针同一固定 `rps=0.08` 压力序列上必须全程 L0 且 `clip_applied_count=0`；持续高压测试
仍须进入压力对应最紧档，收紧延迟不得超过 `adaptive_clip_tighten_hold_s + one update interval`。
任一条件不满足都作为 CPU 硬门失败，不能进入后续真机批次。

### Batch 2：外部 dispatch 接入、审计事实与 parity（CPU）

本方案提出修改路径：`src/sloserve/router/admission.py`、`src/sloserve/router/clipping.py`、`src/sloserve/experiments/benchmark.py`、请求/序列化相关模块；新增 decision-sidecar writer 与对应测试。自适应只在 dispatch 前作用于尚未发送请求；固定路径保持当前 pre-queue 行为。

产出：逐请求决策事实（含四种 `TIGHTEN_HOLD_*` 原因）、join 后正确的 effective cap、关闭状态无
sidecar、no-clip 与 fixed-512 golden fixtures，以及 fake clock + fake backend 下的低负载零截断配对
事实。

验收：

```bash
uv run pytest tests/test_admission.py tests/test_clipping.py tests/test_benchmark.py tests/test_metrics_records.py
cmp tests/fixtures/parity/no-clip-before.jsonl tests/fixtures/parity/no-clip-after.jsonl
cmp tests/fixtures/parity/fixed-512-before.jsonl tests/fixtures/parity/fixed-512-after.jsonl
sha256sum tests/fixtures/parity/*-before.jsonl tests/fixtures/parity/*-after.jsonl
```

文件名可按仓库 fixture 习惯在实现评审时调整，但“完整字节相等 + hash 相等 + 不增加 clock/RNG 调用”验收不可降低。该批不需要 GPU。

另加低负载硬验收：推荐配置的 `clip_applied_count == 0`、`realized_truncation_count == 0`，每条
`backend_max_output_tokens == requested_output_tokens`，所有 decision 均为 L0，且与 no-clip 的固定
trace 逐请求相等。任一不满足即停止；这两项不再留给 GPU 批次证明。

### Batch 3：expL sweep、分析与失败门（CPU）

本方案提出新增 `configs/sweeps/expL-adaptive-clipping.yaml`、`src/sloserve/analysis/expl.py`、`scripts/analyze_expl.py`、`tests/test_expl_analysis.py`，并按现有 sweep override 风格扩展 `src/sloserve/experiments/sweep.py`。

产出：21 点非对称矩阵装配；按 seed/负载/arm 聚合；paired 区间、低负载吞吐等价门、收益保留率、
效用恢复、档位驻留分布、反向切换周期和 thrash 检测；原始事实与聚合对账。CPU fixture 还必须按
§1.2 重放 expK-B 深度 trace，确认默认三档均有触发资格、L3 可达。

验收：

```bash
uv run pytest tests/test_sweep.py tests/test_expl_analysis.py
uv run sloserve config-check --config configs/base.yaml
```

fixture 必须断言恰好 21 个唯一 label：高负载 5 arm × 3 seed = 15 点，低负载 2 arm × 3 seed =
6 点；每个保留的 arm/负载拥有相同三个 seed，且所有正式点为 FCFS、`max_in_flight=4`、
`repetitions=3`、`force_exact_output_tokens: true`。该批只用既有原始数据和合成 fixture，不产生或
伪造性能结果。

### Batch 4：全量静态与回归检查（CPU）

产出：关闭状态兼容、旧 expK-B 分析可读、所有测试和格式检查通过。

验收：

```bash
uv lock --check
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run python scripts/analyze_expkb.py
```

最后一条只重算已保存的 expK-B 数据，不启动服务。若它改变历史结论或无法读取旧数据，Week 7 不得进入 GPU 阶段。

### Batch 5：单点真机语义 smoke（必须真实 GPU）

本批必须由人在与 expK-B 一致的真实 GPU 机器、降频与功耗档下运行，不能在纯 CPU/无 GPU 环境模拟或补造。

产出：一个不计入正式结论的短 smoke 目录，证明真实 vLLM 收到 decision sidecar 记录的 effective cap；L0 请求保持原目标，收紧请求在 `force_exact_output_tokens: true` 下的 `min_tokens == max_tokens == effective cap`；服务停止后无遗留进程。

验收：服务健康检查、请求 JSONL 与 decision sidecar 一一对应、完整性报告无无法解释缺失、错误/超时/取消/拒绝均为零。若真实 cap 与日志不一致，停止，不执行正式 sweep。

### Batch 6：expL 正式矩阵（必须真实 GPU）

本批必须由人在真实 GPU 机器执行：

```bash
uv run sloserve sweep --config configs/sweeps/expL-adaptive-clipping.yaml
uv run python scripts/analyze_expl.py --results results/raw/week7-expL
```

产出：`results/raw/week7-expL/` 下 21 个完整点的请求级 JSONL/CSV、decision sidecar、GPU samples、
completeness report、环境 metadata、sweep CSV/JSON 和 `analysis.json`。每点之间执行既有冷却门；
热保护触发就中止并保留部分结果。

验收：21 个唯一 label 全部存在，且满足高负载 15 点、低负载 6 点的预注册矩阵；每个点正式请求数、
terminal 状态之和、decision 映射和 config hash 完整；同一负载的保留 arm 使用相同 trace seed；分析
脚本从原始数据复算后与 sweep 聚合一致。只有全部通过才计算第 5、6 节的成功/失败门。

### Batch 7：图表、结论与交付（CPU，依赖 Batch 6 真机事实）

产出：延迟—效用 Pareto 图、低负载截断/吞吐等价图、cap 档位随队列压力的时间线；更新进度、学习笔记和报告中的 expL 小节。图与文字只能从 Batch 6 原始事实生成。

验收：图表可由一条版本化脚本从保存数据重建；正文同时报告延迟、效用、失败状态和边界；若预注册门槛失败，标题和结论明确写负结果，并保留对 fixed-512 的诚实比较。该批不需要再次运行 GPU，但没有 Batch 6 的真实数据时不得编写任何性能数字。
