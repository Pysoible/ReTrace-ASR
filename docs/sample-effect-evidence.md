# ReTrace-ASR 样本效果证据

> 目的：用已有真实音频样本验证 ReTrace-ASR 是否能改善 ASR 输出。
>
> 统计原则：同时比较首遍 ASR 的 `raw` 与 ReTrace 当前投影的 `final`，以字符编辑距离计算 CER。事件数量只作为辅助信息，不能代替 CER 改善。

## 1. 结论

已有真实 Alimeeting 样本显示，ReTrace 的开放式音频重听路径能够在出现严重退化、重复或截断时恢复更多有效语音内容，并降低字符错误率。

四个已有样本/切片合计：

```text
参考字符数：10713
raw edits： 6448
final edits：5469
加权 raw CER：0.6019
加权 final CER：0.5105
绝对改善：0.0914（9.14 个百分点）
错误数减少：979
```

因此可以证明一个明确但有边界的结论：

> 在已有的退化/覆盖不足样本上，ReTrace 的“检测异常 -> 音频重听 -> 更新当前投影”路径能够产生可量化的 ASR 改善。

这还不能证明所有类型的 ASR 错误都能被修正，也不能代表正式测试集上的统计显著性。

## 2. 样本结果

| 样本 | 时长/规模 | raw CER | final CER | 绝对改善 | raw edits | final edits | 修订数 | 主要路径 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `R0004_M0025_rerun` | 16 turns | 82.68% | 31.27% | 51.41 个百分点 | 587 | 222 | 4 | 退化 turn 开放式重听 |
| `R0015_M0135_rerun` | 13 turns | 56.13% | 45.25% | 10.87 个百分点 | 733 | 591 | 3 | 退化/截断恢复 |
| `R0020_M0176_rerun` | 16 turns | 72.79% | 62.79% | 10.00 个百分点 | 808 | 697 | 2 | 退化 turn 开放式重听 |
| `R0003_M0046_20min_newcode` | 20 分钟、103 turns | 56.94% | 52.18% | 4.76 个百分点 | 4320 | 3959 | 5 | 长音频退化恢复 |

## 3. 逐样本证据

### 3.1 R0004_M0025

原始结果文件：

```text
retrace_state/alimeeting_eval/R0004_M0025_rerun.summary.json
```

结果：

```text
raw CER：0.8267605634
final CER：0.3126760563
CER 下降：0.5141
revision：4
处理时间：315.5 秒
```

修订集中在严重重复/退化 turn：

- `t011`：首遍文本大量重复“嗯”，重听恢复完整的会议室讨论内容。
- `t013`：首遍只保留很短片段，重听恢复后续会议内容。
- `t014`：首遍退化，重听恢复“玻璃隔断”等内容。
- `t015`：首遍大量重复，重听恢复工位规划内容。

该样本的错误数从 587 降至 222，减少 365 个编辑错误，是当前最强的退化恢复证据。

### 3.2 R0015_M0135

原始结果文件：

```text
retrace_state/alimeeting_eval/R0015_M0135_rerun.summary.json
```

结果：

```text
raw CER：0.5612557427
final CER：0.4525267994
CER 下降：0.1087
revision：3
处理时间：249.5 秒
```

关键修订包括：

- `t007`：首遍“遥”重复，重听恢复为“骁龙”相关内容。
- `t008`：首遍大量“对/嗯”重复，重听恢复关于处理速度的完整讨论。
- `t012`：首遍内容不足，重听恢复“曲屏/瀑布屏”等后续内容。

该样本错误数从 733 降至 591，减少 142 个编辑错误。

### 3.3 R0020_M0176

原始结果文件：

```text
retrace_state/alimeeting_eval/R0020_M0176_rerun.summary.json
```

结果：

```text
raw CER：0.7279279279
final CER：0.6279279279
CER 下降：0.1000
revision：2
处理时间：443.9 秒
```

关键修订包括：

- `t009`：首遍出现长串重复“嗯”，重听恢复实际会议内容。
- `t015`：首遍严重退化，重听恢复后续完整语句。

该样本错误数从 808 降至 697，减少 111 个编辑错误。

### 3.4 R0003_M0046_20min

原始结果文件：

```text
retrace_state/alimeeting_eval/R0003_M0046_20min_newcode.summary.json
```

结果：

```text
时长：20 分钟
turn：103
raw CER：0.5693950178
final CER：0.5218136286
CER 下降：0.0476
revision：5
处理时间：2917.9 秒
```

关键修订包括：

- `t019`：重复退化恢复节假日和日期讨论。
- `t062`：重复退化恢复花朵颜色和布置内容。
- `t087`：重复退化恢复结婚红包相关内容。
- `t093`：短文本通过重听补回上下文。

该样本错误数从 4,320 降至 3,959，减少 361 个编辑错误。它证明改善并不只发生在很短的片段上，20 分钟长音频也能保留正向收益。

## 4. 这些样本证明了什么

### 已证明

- ReTrace 可以识别部分明显的退化、重复和覆盖不足。
- 定向开放式重听可以恢复首遍 Qwen 输出中丢失的内容。
- 当前投影的字符错误数可以低于 raw observation。
- 改善可以在 13 到 103 turn、短片段到 20 分钟样本上观察到。
- ReTrace 不需要修改不可变的 raw observation，也能改善 final transcript。
- revision 的价值可以通过 `raw CER -> final CER` 和 `edit_change` 量化。

### 尚未证明

- 所有普通近音词都能被正确替换。
- 所有历史冲突都能被准确回溯。
- `[DELETE]` 在完整长样本上已经带来整体 CER 改善。
- 当前结果可以直接代表 RAMC 正式测试性能。
- HRA、Revision Precision、CCR、EGR 等需要 revision gold 的人工指标已经可用。

## 5. 删除与替换的专项证据

### 删除

真实 RAMC `t063` 窗口：

```text
音频：CTS-CN-F2F-2019-11-04-387.wav
时间：1801.984 - 1825.344 秒
```

Qwen 闭集验证返回：

```json
{
  "清高的爱拉": 0.0,
  "清高的安拉": 0.0,
  "[DELETE]": 1.0
}
```

服务级 resolver 集成测试进一步证明：

```text
清高的爱拉他曾经说
->
他曾经说
```

因此“候选生成、音频判断、resolver 重分类、删除写回”整条链路已经被验证。

### 替换

已有 Alimeeting 样本中的正向效果主要来自开放式重听对退化内容的恢复，而不是稳定的近音词替换。对 `浴巾 -> 育经` 的测试受到窗口污染，Qwen 选择了 `[DELETE]` 或出现 502，不能作为正向替换成功证据。

因此当前应该把“退化恢复”作为已经证明的能力，把“精确近音替换”列为仍需专项验证的能力。

## 6. 复现方式

已有 summary 文件可以直接复核：

```bash
cd /home/ma-user/work/dataset/sjk_data/pyb/ReTrace-ASR
python - <<'PY'
import json
from pathlib import Path
for path in [
    Path('retrace_state/alimeeting_eval/R0004_M0025_rerun.summary.json'),
    Path('retrace_state/alimeeting_eval/R0015_M0135_rerun.summary.json'),
    Path('retrace_state/alimeeting_eval/R0020_M0176_rerun.summary.json'),
    Path('retrace_state/alimeeting_eval/R0003_M0046_20min_newcode.summary.json'),
]:
    data = json.loads(path.read_text())
    raw = data['cer_asr']
    final = data['cer_current']
    print(path.name, raw['cer'], final['cer'], raw['cer'] - final['cer'], data['n_revisions'])
PY
```

按参考字符加权汇总：

```text
raw edits / total reference chars = 6448 / 10713 = 0.6019
final edits / total reference chars = 5469 / 10713 = 0.5105
```

## 7. 方法有效性的谨慎表述

论文或汇报中建议使用：

> 在四个已有 Alimeeting 真实音频样本上，ReTrace 的异常检测与定向开放式重听路径将加权字符错误率从 60.19% 降至 51.05%，绝对下降 9.14 个百分点，共减少 979 个字符编辑错误。该结果支持 ReTrace 对退化、重复和覆盖不足 ASR 输出的恢复能力，但不等同于对所有近音词修订、历史冲突回溯或正式 RAMC revision gold 指标的全面证明。

不建议使用：

- “ReTrace 已经全面提升 ASR 准确率”。
- “所有 revision 都是正确的”。
- “RAMC 上已经证明 HRA 或 Revision Precision”。
- “单个 t063 删除案例代表完整数据集效果”。

## 8. 下一步样本验证

建议继续采用小样本分层方式：

1. 退化/重复样本：验证开放式重听恢复，并记录 CER gain。
2. 删除幻觉样本：验证原文、替换、DELETE 三候选。
3. 近音实体样本：使用更精确的 token timestamp 或 forced alignment，再验证 REPLACE。
4. 负样本：构造语义合理但声学不支持的候选，确认系统保持 `DEFER`。
5. 只有前四类稳定后，才运行完整 RAMC 样本。
