# ReTrace-ASR 问题与解决方案复盘

> 项目：ReTrace-ASR
> 
> 工作区：`/home/ma-user/work/dataset/sjk_data/pyb/ReTrace-ASR`
> 
> 记录范围：2026-08 期间的 ASR、Context Judge、证据修订、RAMC 评测、服务稳定性和工程运维问题。

## 1. 项目目标

ReTrace-ASR 的目标不是简单地用第二次识别覆盖第一次识别，而是让流式 ASR 具备：

```text
首遍观察 -> 保留原始文本 -> 结合后文发现疑点 -> 定向重听 -> 证据门控 -> 修订或删除 -> 可回放、可解释、可撤销
```

核心约束是：

- 首遍 `raw_text` 不可变。
- `current_text` 只能通过 revision ledger 投影得到。
- LLM 负责提出判断和候选，不能直接覆盖字幕。
- 声学证据、上下文证据和关系判断必须分开记录。
- 证据不足时宁可 `DEFER`，不能为了产生事件而强行修订。
- 删除幻觉片段与替换近音词是两种不同操作。

## 2. 总体结论

本轮迭代已经验证了以下能力：

- Qwen ASR 可以在双 GPU 上并行处理长音频。
- Context Judge 可以回溯历史 turn，并通过 memory 和后文提出 focus。
- 音频 verifier 可以对候选进行闭集判断。
- 普通 `REPLACE` focus 可以被重新分类为 `DELETE`。
- 对真实 RAMC 片段 `t063`，Qwen 对 `[DELETE]` 返回了最高分。
- resolver 和 service 的删除写回链路已经通过集成测试。
- Qwen worker 的异常可以在单 GPU 范围内隔离和重建。

但以下目标尚未完成：

- 完整 RAMC 样本的 final CER 尚未低于 raw CER。
- 正向替换尚未在精确时间窗口上稳定证明。
- revision gold 尚未建立，因此 HRA、Revision Precision 等指标不能作为正式人工标注指标。
- RAMC 的 timestamp overlap 对齐仍可能跨 speaker 或跨 turn，影响逐事件评估。
- 多个 Qwen verifier 请求的并发稳定性仍需继续加强。

最重要的工程结论是：**产生 revision event 不等于识别质量改善，必须同时检查候选、音频窗口、resolver 决策和 raw/final CER。**

## 3. ASR 推理与服务问题

### 3.1 单 GPU 长音频运行时间过长

**现象**

31 分钟 RAMC 样本在单 GPU 上约需 45 分钟，迭代成本过高。

**原因**

原始实现主要使用单个 Qwen/vLLM engine，长音频 chunk 只能串行处理。

**解决方案**

- 每张 GPU 启动一个独立 Qwen worker。
- chunk 按 round-robin 分配到 GPU 0 和 GPU 1。
- 使用 `_OrderedChunkProcessor` 按 chunk index 重新排序。
- 音频推理可以并行，但 Agent 分析仍按时间顺序提交。

**验证结果**

- 双 GPU 后同一 31 分钟样本约需 23 到 24 分钟。
- 已记录过 `RTF=0.7404` 和 `RTF=0.7184` 的运行结果。
- `/api/integrations/status` 能报告 `workers=2` 和 `gpus=["0", "1"]`。

**残余风险**

双卡 worker 之间仍有模型加载、显存和进程生命周期成本；并行推理不能破坏 session version 和 turn 顺序。

### 3.2 双卡乱序破坏会话顺序

**现象**

两个 GPU 的 chunk 返回速度不同，后返回的早期 chunk 可能被后续 chunk 先分析，造成 memory 和 revision 顺序错误。

**解决方案**

`_OrderedChunkProcessor` 缓存乱序结果，只把当前期待的 index 交给 Agent：

```text
pending[index] -> next index -> analyze -> next index
```

**结果**

ASR 推理保持并行，决策和持久化保持时间顺序。

### 3.3 CPU Paraformer 声学分歧导致长任务过慢

**现象**

第二 ASR Paraformer 在 CPU 上对长音频执行分歧检测，耗时显著增加，完整评测几乎无法迭代。

**解决方案**

- 增加 `ASR_ACOUSTIC_DISAGREEMENT` 开关。
- 长样本验证阶段关闭该开关：`ASR_ACOUSTIC_DISAGREEMENT=0`。
- 保留声学分歧代码，以便短样本或正式实验中按需启用。

**取舍**

关闭第二 ASR 可以显著降低运行时间，但失去一部分独立声学证据。它适合工程迭代，不代表正式实验应该永久关闭。

### 3.4 `.env` 覆盖 shell 环境变量

**现象**

命令行中导出的 GPU、模型长度或声学开关与实际服务配置不一致。

**原因**

项目会读取 `.env`，而且环境变量加载顺序使 `.env` 中的值保留下来。

**解决方案**

- 检查 `.env` 和 `run_server.sh` 的实际值。
- 通过 `/api/integrations/status` 验证生效配置。
- 关键配置包括：

```text
ASR_GPU=0,1
ASR_GPU_MEM_UTIL=0.85
ASR_MAX_MODEL_LEN=4096
ASR_MAX_NUM_SEQS=1
ASR_ACOUSTIC_DISAGREEMENT=0
ASR_STRICT_REVISION=1
```

**经验**

不能只看启动命令；必须读取服务状态接口确认真正生效的配置。

### 3.5 服务重启后仍运行旧代码

**现象**

已经修改 resolver，但完整评测中仍然出现旧行为：`清高的爱拉 -> 清高的安拉`，没有 `[DELETE]`。

**原因**

长驻 uvicorn 进程启动时间早于代码修改，没有自动加载新模块。评测实际调用的是旧代码。

**解决方案**

- 检查 uvicorn 的 PID 和启动时间。
- 停止旧进程。
- 重新启动 `run_server.sh`。
- 等待 Qwen worker 完成 preload。
- 通过状态接口确认 `loaded=true`、`workers=2` 后再测试。

**经验**

修改服务端代码后，必须把“代码时间”和“服务启动时间”放在一起核对。只看到 HTTP 200 不能证明新代码已经加载。

## 4. 修订逻辑问题

### 4.1 语义上合理的替换造成错误修订

**案例**

```text
清高的爱拉 -> 清高的安拉
```

“安拉”符合对话主题，且与“爱拉”近音，因此 Context Judge 认为替换合理。

**问题**

RAMC groundtruth 显示，这个位置更可能是首遍 ASR 插入的幻觉片段，正确处理应该是删除，而不是替换。语义合理不是声学证据。

**解决方案**

- 引入显式 `operation`：`REPLACE` 或 `DELETE`。
- 增加特殊候选 `[DELETE]`。
- 普通 `REPLACE` focus 自动加入 `[DELETE]`，候选集合变为：

```text
原文 / 替换文本 / [DELETE]
```

- 若 `[DELETE]` 是音频候选中的最高分，则动态把 `effective_operation` 改为 `DELETE`。
- 删除操作最终写入 `replacement=""`。

### 4.2 DELETE 候选在普通 REPLACE 分支中丢失

**现象**

最初实现只对显式 `operation=DELETE` 生成 `[DELETE]`。如果模型输出普通 `REPLACE`，删除候选不会进入 verifier。

**解决方案**

对所有普通替换 focus 自动追加：

```python
if operation == "REPLACE" and DELETE_CANDIDATE not in candidates:
    candidates.append(DELETE_CANDIDATE)
```

并根据 verifier 的最高候选动态决定最终操作。

**验证**

服务级测试证明：模型只提出 `清高的爱拉 -> 清高的安拉`，verifier 选择 `[DELETE]` 时，最终文本会删除原 span。

### 4.3 三候选闭集返回不完整

**现象**

真实 Qwen 对三候选请求有时没有返回完整的：

```json
{"scores":{"原文":...,"替换":...,"[DELETE]":...}}
```

而是返回缺少 key 的 JSON，或者返回纯文本，导致：

```text
invalid closed-set audio result
```

**解决方案**

`audio_verifier.py` 增加兼容解析：

- 接受完整 `scores`。
- 接受 `{"choice": 1}`。
- 接受 `{"selected": "[DELETE]"}`。
- 接受只包含一个明确候选的纯文本。
- 只有候选唯一匹配时才接受纯文本，避免模糊猜测。

如果三候选仍然返回不完整结果：

1. 先用 `原文/[DELETE]` 二候选重试。
2. 删除二候选也失败时，才回退旧的替换候选。

**验证**

- `choice` 和 `selected` 解析测试通过。
- 三候选失败后 DELETE 二候选重试测试通过。
- 真实服务对 t063 三候选返回：

```json
{
  "清高的爱拉": 0.0,
  "清高的安拉": 0.0,
  "[DELETE]": 1.0
}
```

### 4.4 整段音频窗口污染局部判断

**现象**

对 t063 的完整约 23 秒窗口验证：

```text
清高的爱拉：0.0
清高的安拉：0.0
[DELETE]：1.0
```

这个结果可能受到整段窗口中其它语音的影响，不能直接说明目标短语在音频中不存在。

缩短到末尾约 5 秒后，结果变为：

```text
清高的爱拉：0.3333
清高的安拉：0.6667
[DELETE]：0.0
```

**解决方案**

根据 focus span 在 turn 文本中的相对字符位置估算音频中心，并把长窗口缩短到自适应的 3 到 7 秒：

```text
center = start + text_position / text_length * duration
width = clamp(duration * 0.22, 3, 7)
```

短窗口保持原始边界。

**当前结论**

focus 窗口定位确实会改变模型判断，但文本字符位置不等于真实语音时间位置。要稳定支持短词正向替换，还需要 token timestamp、forced alignment 或更细粒度的声学定位。

## 5. 声学证据问题

### 5.1 第二 ASR 只能作为证据，不能直接覆盖

**问题**

第二 ASR 也可能犯错，例如 Paraformer 对某些专有名词同样听不清。如果直接用第二 ASR 覆盖 Qwen，会把一个错误变成另一个错误。

**解决方案**

- 保存 `paraformer_text`、分歧区间和低置信字符。
- 将分歧作为 `EvidenceFeatures` 输入 resolver。
- 分歧不是无条件修订命令。
- 只有候选、音频 margin、上下文和策略门槛共同满足时才提交。

### 5.2 “重听不同”不等于“重听更正确”

**问题**

同一个 ASR 模型重听时可能输出不同文本，但不同不代表新结果更接近 groundtruth。

**解决方案**

开放式 relisten 只有在以下条件之一满足时才采纳：

- 命中已验证的领域实体。
- 恢复了退化或截断文本。
- 通过候选证据门控。

否则保留首遍文本并记录 `DEFER`。

### 5.3 Paraformer 与 Qwen 同时不确定

**现象**

某些专有名词在两个声学模型中都存在混淆，例如“雷恩加尔”可能被识别成“连家”。

**处理**

此时不强行改写。该情况是系统设计边界：语义和世界知识可能认为某个候选合理，但声学证据不足以安全提交。

## 6. Context Judge 与记忆问题

### 6.1 LLM 过度生成 focus

**现象**

模型把普通词、功能词或语义通顺片段当成疑点，产生大量候选，导致 false revision 超过 correct revision。

**解决方案**

Prompt 和协议中增加约束：

- 只在具体证据存在时提出 focus。
- 普通词的近音差异不能自动当作错误。
- 历史回溯必须指向真实历史 turn。
- `target_turn_id`、`span`、`proposed_text`、`alternatives`、`evidence_turn_ids` 必须完整。
- `operation=DELETE` 时 `proposed_text` 必须为空。
- 候选 span 必须是当前文本中原样存在的最短片段。

### 6.2 音近检测误报：时候 vs 狮子狗

**现象**

“时候”和“狮子狗”拼音相似度较高，简单滑窗会错误生成：

```text
时候 -> 狮子狗
```

最终可能产生类似“狮子狗刚出的狮子狗”的荒谬文本。

**解决方案**

- 当前 turn 已经出现的实体不再参与音近比对。
- stopwords 从完全相等改为包含子串即跳过。
- 增加“实体已在原文出现”的保护。
- 保留真实案例“连加额 -> 雷恩加尔”的检测能力。

### 6.3 历史音频整窗中出现正确词，污染 verifier

**现象**

目标 span 与 proposed text 都在同一个长 turn 的其它位置出现，例如目标是“时候”，候选是“狮子狗”。Verifier 听完整窗口时容易被其它位置的“狮子狗”误导。

**解决方案**

当 span 和 proposed text 都已出现在 raw text 中时，resolver 拒绝该低置信名称互换，并返回：

```text
proposed replacement already appears elsewhere in the same turn
```

### 6.4 Context Judge 输出格式不稳定

**现象**

模型可能返回：

- `focus` 为 dict 而不是 list。
- 使用 `closed_set` 而不是 `alternatives`。
- 缺少 confidence。
- 携带协议之外的未知字段。

**解决方案**

- dict focus 自动包装成单元素 list。
- `closed_set` 兼容为 `alternatives`。
- 对 confidence 提供按 outcome 的保守默认值。
- 忽略未知字段。
- 对 LLM 失败增加 cooldown 和 fallback，避免每个 turn 重试造成延迟放大。

## 7. 持久化、版本与事件问题

### 7.1 事件数很多但 CER 不变

**现象**

早期完整 RAMC 运行出现：

```text
revision_events=74
committed_revisions=1
audio_verified_events=1
raw CER=0.1775852734
final CER=0.1775852734
```

**原因**

事件中有大量 `DEFER`，且唯一提交的修订是错误的“爱拉 -> 安拉”。“事件数量”不能当作系统效果。

**解决方案**

评测同时记录：

- `raw_edits`
- `current_edits`
- `edit_change`
- `committed_revisions`
- `audio_verified_events`
- `defer_count`
- `rollback_count`
- `relisten_seconds`

每个 revision 都要检查它是否让目标参考距离下降。

### 7.2 过期分析覆盖新状态

**问题**

后台 SSE、双卡推理和慢速 LLM 分析并发进行时，旧分析可能在新 turn 或新 revision 之后写回。

**解决方案**

- 每次分析携带 `observed_version`。
- 写入前重新加载 session 并检查 version。
- 发生冲突时基于最新 projection 重试。
- ledger 事件追加写入，不直接覆盖 raw observation。

### 7.3 revision event 与 committed revision 混淆

**问题**

`DEFER`、`KEEP_OLD`、音频验证事件和实际文本改变都曾被混在一起统计。

**解决方案**

区分：

```text
decision event       任何一次决策尝试
verified event       通过音频验证的事件
committed revision   真正改变 current_text 的事件
```

正式指标只能使用与指标定义对应的事件集合。

## 8. 评测与指标问题

### 8.1 RAMC 没有 revision gold

**现象**

RAMC TXT 可以作为 transcript groundtruth，但没有人工标注“哪里原 ASR 错了、是否应该回溯、正确候选是什么”。

**影响**

以下指标不能直接视为正式人工标注结果：

- HRA
- Revision Precision
- Revision Recall
- F2
- False Revision Rate
- CCR
- EGR
- Rollback success

**解决方案**

- 将 RAMC TXT 用于 raw/final CER 和字符级参考比较。
- 使用 `edit_change` 作为修订是否改善参考距离的离线 proxy。
- 对 revision gold 缺失的指标明确输出 `null` 或 `not available`。
- 后续从后文支持错误中抽取片段建立人工 revision/ambiguity gold。

### 8.2 timestamp overlap 对齐不可靠

**问题**

当前 `reference_for_turn()` 主要按照时间重叠选择 TXT 参考，可能跨 speaker 边界或把多个参考句拼在一起。

**影响**

逐 turn 的 `edit_change` 可能不是纯粹的 ASR 改善度量。

**后续方案**

- 引入 speaker-aware 对齐。
- 使用全局序列对齐，而不是只按 overlap。
- 单独报告边界重叠样本。
- 不把跨 speaker 的局部 proxy 当成人工 revision label。

### 8.3 可选指标依赖和网络问题

**现象**

Embedding 指标尝试访问 Hugging Face 时遇到代理超时；MENLI 缺少 `gdown`。

**处理**

- 评测脚本捕获异常并写入 `metric_errors`。
- CER 和事件指标不因可选指标失败而中断。
- 网络恢复并安装依赖后再补算，不回填伪结果。

典型错误：

```text
ProxyError: Cannot connect to proxy
ModuleNotFoundError: No module named 'gdown'
```

## 9. 磁盘与运维问题

### 9.1 磁盘配额导致 HTTP 500

**现象**

评测请求返回 HTTP 500，服务端错误：

```text
OSError: [Errno 122] Disk quota exceeded
```

甚至评测脚本写 `error.json` 时再次失败，因此一度没有任何 v2 结果文件。

**原因**

可再生的临时文件占用大量用户配额：

- `/tmp/retrace_chunks_*`
- `/tmp/vllm-wheels*`
- `/tmp/node-compile-cache`
- 其它历史临时音频和缓存

**解决方案**

清理明确可再生文件：

```bash
rm -rf /tmp/retrace_chunks_* /tmp/vllm-wheels* /tmp/node-compile-cache
```

同时删除失败的空评测目录，保留代码、模型和有效历史结果。

清理后观察到：

```text
/tmp：约 2.8G -> 670M
根文件系统可用空间：约 35G
```

**预防措施**

- 长音频任务前检查 `df -h`。
- 评测失败时保留最小错误信息，不让 error handling 再次因配额失败。
- 任务结束后清理临时 chunk。
- 区分持久化结果和临时音频。

### 9.2 监视命令误导运行状态

**问题**

曾使用：

```bash
tail --pid=$(ps ... ) -f /dev/null
```

该命令不能可靠证明目标评测仍在运行，尤其在 PID 匹配为空或匹配错误时。

**改进**

后续状态报告必须同时核对：

- `ps` 中的精确命令行和 PID。
- 目标输出目录中的最终文件。
- `per_file_metrics.jsonl` 是否存在。
- 服务日志中的 HTTP 状态。
- session 文件中的最终事件。

只有进程退出且指标文件成功生成，才能说评测完成。

## 10. Worker 异常隔离

### 10.1 单个 worker 失败影响整个服务

**现象**

局部 verifier 并发请求时曾出现 HTTP 502，worker 退出或通信异常，影响后续请求。

**解决方案**

`_RemoteEngine` 增加：

- 保存 worker 的 GPU 和配置。
- 捕获 `EOFError`、`OSError`、`BrokenPipeError`。
- 关闭失效 worker。
- 尝试重建同一 GPU worker。
- 当前请求返回错误，不直接让整个服务崩溃。

**已验证**

代码通过编译、诊断和关键链路测试。

**尚未充分验证**

还需要注入真实 worker crash，确认：

- 另一个 GPU worker 仍可用。
- 重建后的 worker 可以继续推理。
- session 不会重复提交或丢失事件。

## 11. 关键真实验证结果

### 11.1 t063 删除候选

音频：

```text
CTS-CN-F2F-2019-11-04-387.wav
时间：1801.984 - 1825.344 秒
```

三候选真实 Qwen 返回：

```json
{
  "清高的爱拉": 0.0,
  "清高的安拉": 0.0,
  "[DELETE]": 1.0
}
```

二候选真实 Qwen 返回：

```json
{
  "清高的爱拉": 0.0,
  "[DELETE]": 1.0
}
```

服务级 resolver 集成测试将普通替换 focus 转换为删除，并得到：

```text
清高的爱拉他曾经说
->
他曾经说
```

### 11.2 RAMC 完整样本历史结果

当前已完成的完整样本运行曾得到：

```text
raw CER：0.1775852734
final CER：0.1775852734
raw edits：1312
final edits：1312
```

早期严格策略运行：

```text
revision_events：74
committed_revisions：1
audio_verified_events：1
Revision Precision proxy：0.0
False Revision Rate proxy：1.0
```

新候选更严格的 v3 运行：

```text
revision_events：70
committed_revisions：0
audio_verified_events：0
raw CER：0.1775852734
final CER：0.1775852734
```

这些结果说明：

- 严格门控可以降低错误提交。
- 但“减少错误修订”还不等于“产生正确修订”。
- 完整 CER 改善必须等候选窗口、真实时间定位和 verifier 稳定后再重新测量。

## 12. 已验证的代码与测试

### 12.1 关键测试

已通过的关键回归包括：

```text
显式 DELETE 写回
普通 REPLACE 自动重分类为 DELETE
三候选失败后 DELETE 二候选重试
choice 候选索引解析
selected 候选文本解析
真实 t063 score 流经 ReTraceService
```

最近的窄测试结果：

```text
8 passed
```

另一次 verifier 和删除链路回归结果：

```text
7 passed
```

### 12.2 静态检查

已通过：

```text
compileall
VS Code diagnostics
 git diff --check
```

### 12.3 GitHub 同步

相关代码已经推送到：

```text
https://github.com/Pysoible/ReTrace-ASR.git
```

分支：

```text
feature/qwen-evolve-integrations
```

已推送提交：

```text
6127e64 feat: strengthen evidence-gated revision candidates
c069a97 fix: isolate ASR worker failures and focus audio windows
```

## 13. 当前未完成问题

按优先级排序：

### P0：真实 focus 时间定位

文本字符比例只是近似。需要：

- Qwen/Paraformer token timestamp。
- forced alignment。
- VAD + 字符/音素时长估计。
- 目标词前后短窗口搜索。

### P0：正向替换稳定性

删除已经在真实 t063 上证明有效，但 `浴巾 -> 育经` 仍受窗口和模型输出影响。必须在候选目标附近取得足够精确的时间窗口，避免将整段语音误判为删除。

### P1：worker crash 回归

补充可控的 worker 崩溃注入测试，验证单 GPU 重建和另一 GPU 继续服务。

### P1：speaker-aware reference alignment

改进 RAMC TXT 与 ASR turn 的对齐，避免用错误的 reference overlap 判断 revision quality。

### P1：revision gold

建立人工标注：

```text
是否 ASR 错误
错误 span
正确操作：KEEP / REPLACE / DELETE
正确候选
是否允许回溯
支持证据
```

### P2：正式实验

待上述问题稳定后，再执行：

- capability ladder
- memory / acoustic / rollback 消融
- bootstrap 置信区间
- HRA、Revision Precision、CCR、EGR、RTS 等正式指标

## 14. 后续推荐工作流

今后的开发验证建议遵循以下顺序：

```text
1. 先用 mock verifier 写短测试
2. 用真实服务验证一个关键 3-30 秒窗口
3. 检查候选、时间窗、score keys 和最终 resolver action
4. 用 3-5 个关键片段做小集合评测
5. 只在局部结果稳定改善后运行完整样本
6. 完成后同时读取 raw/final CER、事件轨迹和日志
```

每次报告必须明确区分：

```text
已实际读取的结果
后台进程状态
尚未验证的假设
```

不能把“启动了后台命令”表述为“已经持续监督”，也不能把“产生了 revision event”表述为“识别效果改善”。

## 15. 最终经验

1. **语义合理不是声学正确。** “安拉”符合主题，但不代表音频中说了“清高的安拉”。
2. **删除是独立操作，不是空替换的附属情况。** 候选生成必须显式包含 DELETE。
3. **验证窗口决定证据质量。** 整段窗口容易污染短语判断，字符比例定位只能作为临时近似。
4. **严格拒绝比错误改写更安全，但还不够。** ReTrace 的价值最终必须体现为正确修订带来的 raw/final 改善。
5. **事件、证据和最终文本必须分层统计。** decision event、verified event 和 committed revision 不能混为一谈。
6. **服务状态必须用可验证信号确认。** 精确 PID、worker 状态、输出文件和日志要互相印证。
7. **评测协议决定结论可信度。** 没有 revision gold 时，应诚实报告 CER 和 proxy，不能把 proxy 当人工标签。
8. **长任务不是调试工具。** 先用关键窗口和短测试解决控制流问题，再做完整样本。
