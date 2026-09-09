# ReTrace 方法详说 PPT 设计

## 目标

面向论文汇报，将现有 ReTrace-ASR 演示文稿的方法部分更新为与当前代码一致的、可追溯的 Agent 架构说明。听众应理解：ReTrace 不是让 LLM 直接改写字幕，而是让 LLM 对后续对话作反思提案，由控制器以不可变原始 ASR 证据验证，再以事件回放重建当前字幕。

## 叙事主线

采用“证据驱动的回溯修订”主线。一个小型对话贯穿方法章节：早期 ASR 观测保留为原始记录；后续 Turn 使 Agent 对旧文本提出替代解释；只有逐字可核的后续原文证据通过验证后，修订事件才进入当前显示；撤销则停用事件并重放。

## 方法章节（五页）

1. **Method Overview**：说明不可变观测与可更新解释的分离；展示 ICO → RHR → EPV → EBR。
2. **RHR — Retrospective Hypothesis Reflection**：每个新 Turn 触发 DeepSeek 对此前原始 Turn 的受约束提案。提案字段为目标 Turn、旧片段、候选替换、后文引用、分数与理由；没有足够证据时返回空集合。
3. **EPV — Evidence-Grounded Proposal Verification**：控制器逐条检查目标在当前 Turn 之前、替换片段存在、分数不低于 0.7、所有引用均来自目标之后且不晚于当前的原始 Turn，且引用文本逐字存在。失败即拒绝，不改变可见字幕。
4. **EBR — Event-Sourced Belief Replay**：Turn.raw_text 从不覆写；被接受的提案成为带 source/target/evidence/rationale 的 RevisionEvent。显示文本由 raw turns 与 active events 重放导出；undo 停用对应事件并重新回放。
5. **Worked Example**：用“涂/屠”仅作为可视化例子，不将其表述为特定实体知识。突出后文原始引文、修订理由、事件记录和撤销能力。

## 准确性约束

- 不宣称系统解决所有歧义。
- 不依赖用户输入领域词库或实体画像。
- DeepSeek 的角色是受约束的反思/提案器；Controller 才是验证与提交边界。
- 保留旧代码兼容接口不作为核心方法叙述。

## 视觉与交付

基于用户提供的 PPTX 进行模板跟随编辑，保留已有母版、页码与论文汇报风格。方法页采用简洁流程、逐字证据标注和时间线，不以密集仪表盘或无依据性能图取代方法论。输出为新的可编辑 PPTX，并逐页渲染检查。
