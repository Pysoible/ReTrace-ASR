# ReTrace-ASR 本周工作进展

> 周期：2026-08-14 ～ 2026-08-21　|　作者：ReTrace-ASR 项目组

---

## 一、项目介绍

### 1.1 研究问题

流式对话 ASR 往往会在上下文尚未充分展开时提交一个听起来合理、但可能错误的人名、
术语或指代。后续对话虽然可能提供决定性证据，传统流式字幕却已经没有回溯路径。**ReTrace-ASR**
研究的问题是：当流式 ASR 产生这类错误时，系统能否像人类一样“存疑 → 重听 → 修正”，
同时保留原始观察、控制误改风险，并让每次字幕变化都可以解释、回放和撤销？

ReTrace-ASR 是一个 Agent-first 的实时语音识别回溯修订原型。Qwen 系列 ASR 负责首遍
识别，Context Judge 负责结合未来上下文和双层记忆提出结构化判断，确定性服务组件负责
关系仲裁、声学验证、证据门控和事件回放。系统不直接覆盖首遍文本，而是把最终字幕视为
不可变 raw turn 与追加式 revision ledger 的投影。

### 1.2 本周总体进展

本周项目从“回溯修订机制可运行”推进到“声学证据、记忆检索、服务化评测和决策可观测性
协同工作”的阶段，主要完成以下四条主线：

1. **声学证据链增强**：接入可选的第二 ASR 意见、字级声学置信度、分歧区间和漏字/截断
   检测，并将这些信号路由到定向重听。
2. **高置信字级修订**：在严格保守约束下支持只修正少量高置信分歧字符，避免整段二次
   识别覆盖首遍结果。
3. **记忆与决策链升级**：完成混合语义记忆检索、已验证实体接入 Agent、证据门控和前端
   decision trace 展示，进一步强化“可审计而非黑箱改写”的论文主线。
4. **真实音频评测闭环**：新增 RAMC 数据集服务化评测和后文支持错误代理指标脚本，能够
   记录首遍/回放 CER、RTF、修订、延迟和音频访问等字段，并暴露当前没有 revision gold
   时的指标边界。

---

## 二、核心工作详述

### 2.1 双卡 Qwen ASR 与流式处理稳定化

- Qwen ASR 支持 GPU 0/1 各自启动 worker，音频 chunk 按 round-robin 分片并行推理。
- `_OrderedChunkProcessor` 对双卡可能乱序返回的 chunk 做缓冲，仍按音频 index 顺序进入
  Agent 分析，避免并发推理破坏会话时间线。
- 保留单卡 in-process 路径，双卡服务通过状态接口报告加载状态、worker 数和 GPU 配置。
- 音频上传与后台分析继续采用 SSE 逐 turn 推送，前端可以在长音频尚未处理完时看到已经
  完成的 turn 和决策轨迹。

本周的重点从“能并行”进一步转向“并行后仍然保持因果顺序、版本一致性和可诊断性”。

### 2.2 漏字/截断检测与当前窗口恢复

针对长音频 chunk 中“语音存在但首遍文本过短、尾部被截断或重复退化”的问题，本周完成了
声学覆盖证据路由：

- 使用 VAD 估计有效语音时长，并以字符密度识别可能的漏字/截断窗口。
- 对疑似截断窗口触发受限 plain 重听，仅在重听结果不再退化且覆盖更合理时采纳。
- 对重复填充词和循环输出执行 degeneration 检测；若重听结果仍退化，则从首次重复位置
  裁剪尾部，保留有效前缀。
- 将覆盖风险、低置信字符、第二 ASR 分歧等信息写入 turn uncertainty，供后续 Agent 统一
  判断，而不是在 ASR 层直接覆盖 raw text。

这条路径与历史冲突修订保持隔离：当前窗口恢复只能修复当前 turn，不会因为一次声学重听
自动改写任意历史 turn。

### 2.3 第二 ASR 与字级高置信度纠正

本周接入可选的 Paraformer 第二声学意见，保存 `paraformer_text`、字符级置信度和分歧
区间。编排层在发现低置信字符或独立 ASR 分歧时，才把对应窗口交给重听路径。

新增的 `high_conf_correction` 遵循以下保守策略：

- 只在第二 ASR 的字符置信度达到门槛、且与首遍文本存在明确字级分歧时尝试替换或插入。
- 低置信字符不作为修订依据；对于第二 ASR 的漏字，默认保留首遍内容。
- 字符置信度必须与文本严格一一对应，否则直接返回首遍文本，避免索引错位造成误改。
- 少量字符修订不再被“整体文本相似度过高”的普通重听门槛错误拒绝，但仍必须通过后续
  声学和证据门控。

这使声学意见从“附加日志”变成了 Agent 可以消费的证据，同时保留了首遍 raw observation
不可修改的协议约束。

### 2.4 混合语义记忆与已验证实体记忆

- 长期记忆检索从单一词面匹配升级为中文字符 n-gram、BM25、可选 dense embedding 和
  alias/entity 信息的混合排序。
- 只有经过置信度、独立来源或声学验证门槛的 belief 才能进入 stable long-term memory；
  被新证据推翻的事实保留 supersedes/provenance 链，而不是物理删除。
- 已验证实体及其来源、会话和 turn 信息被纳入 Context Judge 的输入，用于候选解释和定向
  重听，但不再作为无条件热词覆盖当前识别。
- 短期记忆继续保存近期 turn、依赖 turn、WorkingBeliefs 和 OpenHypotheses，从而区分
  “当前工作假设”和“跨来源稳定事实”。

论文中的双时标记忆设计已经有了对应实现：记忆是仲裁证据，而不是把所有字幕强行归一化
到某个历史实体的命令。

### 2.5 证据门控、版本一致性与前端可观测性

本周进一步强化了 `CONFLICT` 之后的决策链：

- Context Judge 输出 `CONSISTENT / NOVEL / CONFLICT / UNCERTAIN`，并将冲突区分为
  `MUTUALLY_EXCLUSIVE / COEXIST / TEMPORAL_CHANGE`。
- 只有互斥且有明确 focus 的冲突才进入历史音频闭集验证；无音频、验证失败、候选不一致或
  margin 不足时保持 `KEEP_OLD` 或 `DEFER`。
- 每次分析携带 observed version，提交前重新检查 session 是否已经变化；过期任务重新基于
  最新 projection 判断，普通事件保持幂等。
- revision ledger、rollback、accepted facts、pending hypotheses 和 status counts 统一
  进入 decision state trace，并在前端展示，便于检查“为什么改、改了什么、证据来自哪里”。
- 前端保留三面板实时渲染和内部滚动/折叠，长会话分析过程中可以同步观察字幕、Agent 判断
  和记忆状态。

这部分对应论文中的核心主张：显示文本是 `Replay(O1:t, Et)` 的可回放投影，不是被模型
偷偷覆盖后的新观察。

---

## 三、评测与量化结果

### 3.1 工程基准上的声学纠正结果

在此前固定的 100 号长音频、142 个 chunk 的整段拼接测试中，逐步加入声学覆盖检测和字级
高置信纠正后得到：

| 配置 | 整段 CER | 相对首遍变化 | REVISE 事件 |
| --- | ---: | ---: | ---: |
| 首遍 ASR，无漏字重听 | 32.35% | - | 0 |
| 加入漏字/覆盖重听 | 23.35% | -9.00 个百分点 | - |
| 再加入字级高置信纠正 | 22.06% | -10.29 个百分点 | 47 |

该结果说明声学覆盖检测对当前测试样本有明显诊断价值，字级高置信路径还能进一步修正少量
分歧字符。但这是工程验证样本，不是 RAMC/AliMeeting 正式测试集结果，不能据此宣称统计
显著性或泛化性能。

### 3.2 RAMC 服务化评测闭环

本周新增 `scripts/run_ramc_dataset_eval.py`，通过运行中的 ReTrace 服务提交音频，等待
会话完成后保存每个样本和数据集汇总结果。当前输出已覆盖：

- raw/final CER、字符 F1、RTF 和处理耗时；
- revision、audio verification、defer、rollback 事件计数；
- relisten 时间代理、revision delay 等成本/时序字段；
- revision precision、recall、F2、overcorrection 和 rollback success 等待补充 gold 后
  计算的字段。

最近的短样本检查已经验证评测服务链路可以完整跑通：30 秒 `sample30` 产出 raw/final CER
为 97.45%、RTF 约 0.58～1.37，并记录 2 个 revision event；843 秒样本产出 RTF 0.45、
29 个 revision event。当前这些运行的 final CER 与 raw CER 相同，说明“能够产生事件”并不
等价于“事件已经改善参考文本”，后续需要逐事件检查触发、证据和目标 span。

同时，评测脚本明确记录了当前协议限制：RAMC TXT 提供的是 transcript/reference，不提供
人工标注的 revision gold，因此 revision precision/recall、F2、overcorrection 和 rollback
success 目前不能报告为有效数值。Embedding/MENLI 等可选指标也因环境中未安装对应依赖而
保持空值。

### 3.3 论文实验状态

论文已经形成完整的方法和实验协议：Raw Qwen、Current-turn LLM、Sliding-context LLM、
Memory-only、Always Re-listen 与完整 ReTrace 的 capability ladder，以及去除短期记忆、
长期记忆、关系仲裁、选择性重听和 rollback/revalidation 的机制消融矩阵。

目前仍处于“协议和脚本可运行、正式结果待执行”的阶段：bootstrap 门槛尚未在开发集上冻结，
后文支持错误 slice 仍是自动代理而非人工歧义标注，论文表格中的正式比较数字暂不填写。

---

## 四、本周问题定位与经验沉淀

1. **声学证据必须进入决策链，而不是只写日志**：第二 ASR 的分歧只有经过窗口定位、字符
   对齐和门控，才有资格影响字幕投影。
2. **事件数不等于修订收益**：服务可以产生 revision event，但若目标 span、候选 winner
   或参考对齐不正确，最终 CER 不会改善。因此评测必须同时看 raw/final、事件证据和逐样本
   轨迹。
3. **记忆需要独立性约束**：词面命中或模型高置信度不能单独把 belief 提升为长期事实；跨
   turn、跨 session 或音频验证才构成更可靠的支持。
4. **评测标签决定指标上限**：没有 span-level revision gold 时，不能把自动重复形式代理
   直接当成人工语义歧义结论，必须将 CER、过度修订和后文支持 slice 分开报告。
5. **并发系统需要版本重验证**：双卡乱序、后台 SSE 和慢速 Agent 分析同时存在时，observed
   version 检查是避免旧分析覆盖新状态的必要条件。

---

## 五、当前风险与未完成事项

- RAMC/AliMeeting 开发集上的 `suspect`、`relisten`、`revise`、`long_memory` 和音频
  margin 门槛尚未完成真实标定，当前仍是 bootstrap 配置。
- 尚未建立人工 revision/ambiguity gold，因此 revision recall、F2/F3、overcorrection、
  rollback success 和 Memory 污染率尚不能作为正式论文结果。
- 最新 RAMC 检查中部分样本的 revision event 尚未带来 CER 改善，需要继续检查事件触发、
  目标 span、候选闭集和参考对齐。
- 完整 capability ladder、机制消融、配对 bootstrap 置信区间和显著性检验尚未执行。
- 真实音频复听秒数、DeepSeek 调用次数、端到端延迟分布和任务合并策略仍需补齐。
- 高层设计与代码已经基本对齐，但前端历史快照断言和全量回归仍需跟随最新 UI/trace 契约
  同步验证。

---

## 六、下周计划

1. **完成开发集门槛标定**：在 RAMC/AliMeeting 开发集冻结 `K`、revision threshold、音频
   margin 和长期记忆门槛，记录 overcorrection cap 下的选择过程。
2. **建立 revision gold**：从后文支持错误代理中抽取样本，补充目标 span、关系类型、正确
   候选和是否允许历史修订的人工标注。
3. **逐事件诊断 RAMC 评测**：重点分析“有 revision 但 CER 不变”的样本，区分发现错误、候选
   验证、resolver 拒绝和 reference 对齐问题。
4. **运行正式指标与消融**：输出最终 CER、revision precision/recall、F2/F3、overcorrection、
   rollback success、audio access 和 RTF，并执行记忆/声学验证/回滚/版本重验证消融。
5. **补齐成本和稳定性报告**：统计复听秒数、judge 调用次数、端到端延迟、并发顺序和长会话
   SSE 行为，更新前端回归测试。

---

## 七、关键经验

- **“重听不同”不代表“重听更正确”**：只有在声学置信度、候选一致性和上下文证据共同满足
  门槛时，二次识别才可以改变显示投影。
- **把模型判断和状态变更分开**：LLM 负责提出结构化证据，resolver、decision engine 和
  ledger 负责决定是否改变字幕，才能保留可审计、可回滚的边界。
- **先把测量链路做实，再讨论效果**：raw/final CER、事件轨迹、音频访问和延迟必须来自同一
  会话版本，指标缺失时明确标记为 unavailable，而不是用代理数字填空。
