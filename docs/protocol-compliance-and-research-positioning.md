# ReTrace-ASR 协议合规与研究定位

## 已实现的第一阶段协议

- 原始 `raw_text` 不可修改，`current_text` 只由 raw 与只追加账本回放得到。
- 文本快链路返回 `202 queued`；慢链路按会话串行、跨会话并行。
- 每个慢任务携带 `observed_version`；过期任务基于最新 Session 重新判断，重复任务幂等。
- Memory 分为近期上下文、WorkingBeliefs、OpenHypotheses、DependencyIndex 和跨会话稳定信念。
- Context Judge 先输出 `CONSISTENT | NOVEL | CONFLICT | UNCERTAIN`，并可输出带 Turn 证据的结构化信念。
- 冲突显式区分 `MUTUALLY_EXCLUSIVE | COEXIST | TEMPORAL_CHANGE`。
- 只有互斥文本解释会进入闭集历史音频验证；缺音频、验证失败或 margin 不足均 `DEFER`。
- `ACCEPT_NEW | KEEP_OLD | REVISE_CURRENT | REVISE_HISTORY | COEXIST | DEFER | ROLLBACK` 均可审计。
- 长期信念带置信度、时间有效性、来源会话、创建/更新时间版本和 supersedes 链。
- 全局 N-gram、写死中文二元词表、静态 entity/quarantine 旧链路均已退出。

## 尚不能宣称完成的研究部分

- 没有本地连续语音数据集和 ASR 模型，尚无 CER/WER、显著性或置信区间结果。
- 当前概率模型和门槛是 bootstrap 配置；代码支持开发集选择门槛，但尚未完成真实标定。
- 长期 Memory 相关性目前是词面与结构匹配，尚未与稠密检索或学习式 retriever 比较。
- 尚未实现指向同一 hypothesis 的任务合并，只实现了会话级顺序和 observed-version 幂等。
- 尚未报告 Memory 污染率、真实音频复听成本和端到端延迟曲线。

因此，当前代码可以支撑方法原型、消融设计和离线合成测试，但不能仅凭这些测试声称达到顶会实验结论。

## 三个可投稿的核心创新假设

### 1. 对称的时间化 ASR 信念仲裁

系统不预设新文本或旧 Memory 谁更可信，而是把新旧解释建模为可竞争、可共存、可随时间变化的信念。后续证据既能修订历史 Turn，也能通过 superseding event 自动回滚先前修订。创新重点不是单纯利用未来上下文，而是对 ASR 文本、会话事实和现实变化进行统一的可逆仲裁。

### 2. 与字幕因果依赖耦合的双时标 Memory

短期 Memory 保存工作信念、竞争假设和 Turn 依赖；长期 Memory 只提升具有独立来源或声学验证的版本化信念。长期知识不会直接批量改写历史字幕，每一处文本变化仍需独立证据事件，从而把 Memory 帮助与 Memory 污染风险同时变成可测量对象。

### 3. Agent 触发的选择性历史声学验证

Context Judge 在任何全局候选扫描之前决定是否存在冲突，并只对目标 span 请求历史音频闭集验证。执行层使用开发集标定概率、最佳/次佳 margin 和乐观版本重验证，在高修订召回目标下限制过度修订。该设计把语义发现、声学授权和实时并发一致性分离，而不是让 LLM 直接生成并覆盖字幕。

## 必要实验

上述三点必须分别通过消融验证：去掉未来证据、去掉短期 Memory、去掉长期 Memory、去掉声学验证、去掉回滚、去掉版本重验证，并在相同过度修订率下比较历史修订召回率、F2/F3、最终 CER/WER、回滚成功率、Memory 污染率和延迟。
