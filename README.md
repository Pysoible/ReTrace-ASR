# ReTrace-ASR

ReTrace-ASR 是一个 Agent-first 的实时 ASR 回溯修订原型。它保存不可修改的首遍 ASR，结合短期与长期 Memory 判断新信息与旧解释的关系，并在上下文和历史音频共同支持时自动修订或回滚字幕。

## 核心流程

1. `OBSERVE`：立即持久化 `raw_text`，文本接口返回 `202 queued`。
2. `MEMORY RETRIEVE`：读取近期 Turn、未决假设、依赖 Turn 和长期稳定事实。
3. `CONTEXT JUDGE`：Agent 输出 `CONSISTENT | NOVEL | CONFLICT | UNCERTAIN`；只有冲突或不确定时才能提出闭集候选。
4. `TARGETED RELISTEN`：仅验证 Agent 指定的历史音频窗口和候选，不做全局 N-gram 扫描。
5. `EVENT REPLAY`：以不可修改的 raw Turn 和只追加事件重建当前字幕。新版证据可以触发 `ROLLBACK`，无需用户确认。

Memory 分为两层：会话内短期 Memory 保存近期上下文、工作事实和未决假设；长期 Memory 只接收达到置信门槛且拥有独立来源或音频验证的事实。长期存储故障时，服务自动降级到短期 Memory。

## 当前环境边界

仓库当前不附带数据集或本地 ASR 模型。所有核心测试使用注入式 Context Judge、音频验证器和合成会话，因此可离线运行，但不能据此宣称真实 CER 提升或门槛已经完成数据标定。

`backend/asr_agent/calibration.py` 中的门槛是显式 bootstrap 配置，目标偏向提高错误发现与修正召回率，并允许少量误改。接入开发集后应重新标定 `suspect`、`relisten`、`revise`、`long_memory` 和音频 margin，再报告 revision recall、F2/F3、overcorrection 与 rollback 指标。

未配置 DeepSeek 时，系统只读取 ASR 显式置信度/N-best 信号，不自行猜测错误 span；未配置历史音频或音频验证失败时，动作固定为 `DEFER`，不会只凭文本改写字幕。

## 运行

```bash
uv sync --group dev
cd frontend && npm install && npm run build && cd ..
uv run uvicorn asr_agent.server:app --reload
```

可选配置：

- `DEEPSEEK_API_KEY`：启用结构化 Context Judge。
- `ASR_AUDIO_ENABLED=1` 与 `ASR_MODEL_PATH`：启用本地 Qwen 音频转写。

## API

- `POST /api/sessions/{id}/turns`：提交不可修改的文本观察，立即返回 `202`。
- `GET /api/sessions/{id}`：读取版本、分析状态、两层 Memory、未决假设和修订账本。
- `POST /api/sessions/{id}/audio/upload`：模型可用时上传音频并按 chunk 建立 Turn。
- `GET /api/integrations/status`：检查 Qwen 与 DeepSeek 是否就绪。

## 验证

```bash
.venv/bin/pytest -q
cd frontend && npm run build
```
