# ReTrace 选择性历史音频复核设计

## 目标

将 ReTrace 改造成无人工干预的流式会话 ASR 消歧系统：后文语义仅负责触发和约束候选，目标历史音频负责最终验证。系统自主选择 `WAIT`、`RELISTEN` 或 `REVISE`，并将所有自动修订记录为可回放事件。

## 方法边界

- 不提供 `ASK_USER`、人工确认 API 或前端确认控件。
- LLM 不可自由生成替换文本；其输出必须从初始 ASR N-best / 不确定候选组成的闭集选择。
- 后文证据必须逐字定位到不可变 `raw_text`；音频证据必须关联目标 chunk 的音频路径及时间窗口。
- 未达到双证据阈值的候选保持原文，不进行猜测性修订。

## 状态与动作

每个可疑 span 保存 `(turn_id, span, candidates, confidence, nbest, audio_ref, semantic_evidence, acoustic_evidence, decision)`。其中 `audio_ref` 包含原始音频路径和 chunk 起止秒数。

1. `WAIT`：没有后文语义证据，或候选分数不足。
2. `RELISTEN`：后文逐字引文支持候选，但初始 ASR 不确定；系统调用候选条件化音频验证器。
3. `REVISE`：同一候选同时通过后文语义验证和音频验证，并满足分数与 margin 阈值；控制器写入 `RevisionEvent`。

音频验证器是可注入接口，输入为音频路径、时间窗口和候选文本，输出闭集候选分数。生产适配器使用 Qwen-Omni 音频提示；测试可用确定性 stub。

## 提交约束

令 `s(c)` 为语义支持分数，`a(c)` 为音频复听分数，`p(c)` 为初始 ASR 候选先验。控制器计算：

`F(c) = alpha * a(c) + beta * s(c) + gamma * p(c)`。

仅在 `F(c*) >= threshold`、`F(c*) - max(F(other)) >= margin` 且候选有逐字后文证据时提交。缺少任一条件时保留 `WAIT`。默认 alpha、beta、gamma、threshold、margin 可配置并持久化在事件理由中。

## 工程改动

- `retrace.py`：删除人工确认路径；保存音频引用；在后文反思后生成 `RELISTEN`；调用验证器并在双证据通过后提交。
- `audio_verifier.py`：裁剪读取目标音频窗口，提示 Qwen 在闭集候选中输出 JSON 分数；失败时不提交。
- `server.py`：删除确认端点；在音频 chunk 进入服务时保存可复听元数据。
- `frontend`：删除确认按钮，展示自动动作、音频/语义证据和 wait 状态。
- `paper/`：生成匿名 ICASSP 风格 Overleaf 项目，包括方法、实验协议、局限和 BibTeX；不编造实验数值。

## 验收

- 高风险和低置信度均无人工路径。
- 有语义证据但无音频验证时只产生 `RELISTEN`，不改字幕。
- 双证据支持唯一候选时追加修订事件；无唯一 margin 时保留原文。
- 所有现有 undo/raw-text 不变量继续成立。
- LaTeX 可由 `latexmk` 或 `pdflatex` 成功构建，或在运行时未安装 TeX 时通过结构检查。
