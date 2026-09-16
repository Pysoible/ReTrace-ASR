# ReTrace 不确定性决策层设计

> 已由 `2026-08-07-retrace-selective-audio-verification-design.md` 取代。
> 本历史设计包含人工确认，不属于当前全自动方法。

## 目标

在不改变 ReTrace「不可变原始 ASR 观测、后文原始证据验证、事件回放和可撤销修订」边界的前提下，吸收 TAMA 的不确定性管理能力。系统应在提交修订前显式保存竞争候选、证据来源与下一步动作；前端应可解释这些未决状态并允许对高风险候选做人工确认。

## 非目标

- 不引入领域词库、热词二遍转写或外部 Claude SDK 编排。
- 不允许模型、记忆或人工操作覆写 `Turn.raw_text`。
- 不允许已修订的 `current_text` 成为新的提交证据。

## 架构

ReTrace 在每个新 Turn 到达后运行三个阶段：

1. **不确定性发现**：从 ASR 低置信度、N-best 分歧、码切换片段和隔离记忆近似匹配中产生 `SuspiciousSpan`。已有上游 `text_candidates` / `entity_candidate_ids` 继续作为候选来源。
2. **候选与策略**：每个 `Hypothesis` 保存候选、分数、支持/反驳原始证据、风险及动作。当前控制器选择 `WAIT`、`ASK_USER` 或 `COMMIT`；音频复听和记忆检索仅以证据需求记录，尚未作为已执行工具声明。只有低不确定性、明显领先且存在已验证后文原文引句时才允许 `COMMIT`。
3. **证据化提交**：`COMMIT` 仍走现有 `RevisionEvent`。控制器逐项验证 target/source 时序、原文引用、候选集合和分数。事件回放从 raw turns 导出显示字幕；undo 仅停用事件。

## 数据模型与 API

- `CandidateState`：`text`、可选 `entity_id`、`score`、`supporting_evidence`、`contradicting_evidence`。
- `Hypothesis`：保留原有 span/candidate/entity 字段，增加 `candidates`、`risk`、`decision`、`decision_rationale` 和 `evidence_packet`。
- `SuspiciousSpan`：记录文本位置、分数和触发原因；保存在 `Turn.meta`，不改变 raw 文本。
- `RevisionEvent.evidence` 继续持久化为 `turn_id:quote`。新增证据包只用于解释和候选策略；提交事件必须包含来自后续 raw turn 的逐字引句。
- text-turn API 可继续接收现有字段，并新增可选 `nbest`、`speaker`、`risk`。音频适配器传入可获得的 N-best、置信度和时间戳。
- 新增确认 API：人工选择候选或保持原文时，追加审计事件；确认仍须调用控制器，不直接改写字幕。

## 动作语义

- `COMMIT`：候选的后文原始证据完整且领先，追加 `REVISE_TEXT` 或 `REVISE_ENTITY`。
- `WAIT`：没有可用的追加证据，保留候选以等待未来 Turn。
- `ASK_USER`：风险为 high 时优先，或竞争候选长期接近时触发；用户可确认候选或保持原文。

## 前端

维持 Research Notebook 的时间线和 Undo。在 Agent Note 增加未决候选卡：显示候选排名、动作、原因，以及当前可得的 ASR 与后文原文证据。`ASK_USER` 卡提供确认候选和保持原文按钮；已确认/拒绝操作在审计时间线中可见。

## 安全与正确性约束

- 证据校验统一从 `Turn.raw_text` 读取，绝不聚合 `current_text` 作为判定事实。
- 低置信度或候选领先本身都不能形成修订；它们只决定是否需要进一步取证。
- 高风险候选没有人工确认时不得 `COMMIT`。
- 所有新状态、人工确认和撤销操作均写入会话持久化文件。

## 测试与验收

单元测试覆盖自动 span 发现、竞争候选不提交、后文证据后提交、高风险请求确认、raw-only 证据约束和 undo 回放。API 测试覆盖扩展字段与确认端点。前端测试要求候选、动作、证据包和确认控件出现，同时保持现有 revision/undo 行为。完整 Python 测试和前端生产构建必须通过。
