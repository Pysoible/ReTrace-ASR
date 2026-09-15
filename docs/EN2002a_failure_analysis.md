# EN2002a English ASR / ReTrace Failure Analysis

Date: 2026-08-28

## Executive Summary

This report analyzes the real end-to-end result for the AMI English meeting sample `EN2002a`.
The conclusion is not that Qwen-Omni is intrinsically unusable. The result is dominated by three separate problems:

1. The AMI reference is speaker-grouped STM, while the system output is a single mixed-audio stream split into chunks. Comparing a chunk with all text that overlaps its time window is not a valid ordinary WER calculation.
2. The original ASR prompt was Chinese-oriented. The old result therefore contains Chinese translations and should not be used as a baseline for the current code.
3. ReTrace applied open-vocabulary whole-turn recovery to several English turns. Four `low_diversity` decisions rewrote healthy or partially correct English instead of making a verified local edit. This is a real ReTrace regression, independent of the scoring problem.

The current report separates the facts. It does not present the old `ami_time_sorted_metrics.json` number as a trustworthy AMI WER.

## Files And Provenance

### Ground truth

The original AMI ground truth is unchanged and remains the authoritative source:

- Full STM: [EN2002a.stm](../../AMI/STM/EN2002a.stm)
- Lines: 746
- Recording: `EN2002a`
- Speakers: A, B, C, D
- Time range: approximately 0.000 to 2142.704 seconds
- Format: `recording channel speaker start end transcript`

The STM is speaker-grouped rather than globally chronological. The chronological derived copy used by the evaluator is generated separately and must never replace the original:

- Derived sorted STM: [EN2002a.time_sorted.stm](../retrace_state/ami_eval/EN2002a_fixed/EN2002a.time_sorted.stm)

### Audio and system result

- Audio: [EN2002a.wav](../../AMI/Audio/EN2002a.wav)
- Full API result: [result.json](../retrace_state/ami_eval/EN2002a_fixed/result.json)
- Evaluator report: [ami_time_sorted_metrics.json](../retrace_state/ami_eval/EN2002a_fixed/ami_time_sorted_metrics.json)
- Evaluator: [run_ami_eval.py](../scripts/run_ami_eval.py)

The full API result contains every system turn, raw text, current text, metadata, uncertainty signals, and revision events. The STM file contains every ground-truth utterance. These two files are linked above rather than silently copied into a second mutable source of truth.

## Dataset Structure

AMI is not a single-speaker read-speech benchmark. It contains a mixed meeting signal with multiple participants and overlapping speech. The reference has 746 STM utterances from four speakers. A single system stream cannot be scored as ordinary single-speaker WER unless the system output is first diarized or assigned to speakers with sufficiently fine-grained timestamps.

The tested recording is 2142.709 seconds long. The ASR service produced 144 chunks and 144 turns. The model output is chunk-level text with chunk start/end times, not word-level timestamps and not speaker-attributed words.

The first few reference utterances, in the original speaker-grouped file, are:

```text
EN2002a 1 A 0.944 7.068 Wonder how much of the meetings is talking about the stuff at the meetings Yeah
EN2002a 1 A 21.400 23.328 Not a lot No
EN2002a 1 A 25.344 30.166 Hmm Okay Sounds like you've done some stuff So
EN2002a 1 A 32.484 32.871 'Kay
EN2002a 1 A 106.656 107.034 Hmm
```

The chronologically sorted file starts with several speakers at the same time:

```text
EN2002a 1 D 0.000 1.901 Funky sh stuff like that
EN2002a 1 C 3.492 5.446 Yeah exactly yeah yeah yeah
EN2002a 1 D 3.612 12.288 Yeah Yeah Look at all this stuff man Okay Right
EN2002a 1 C 13.120 13.722 Should we
EN2002a 1 D 14.959 15.625 Has anybody
```

This is why sorting is necessary but not sufficient: chronological order does not remove overlapping speakers.

## Model And Runtime

The observed run used:

| Item | Value |
|---|---|
| Model | `/home/ma-user/work/dataset/sjk_data/sjk/model_demo/checkpoint-793-merged` |
| Model type | `qwen3_omni` |
| Backend | Qwen Omni vLLM |
| GPU workers | 2 |
| GPUs | 0, 1 |
| Sampling rate | 16 kHz |
| Chunk maximum | 15 seconds in the current chunk configuration |
| Audio duration | 2142.709 seconds |
| ASR chunks/turns | 144 / 144 |
| ASR elapsed time | recorded in `result.json` under `asr.elapsed_sec` |
| Swift source | `/home/ma-user/work/algorithm/qwen_omni_algorithm_asr_sjk/ms-swift-bk` |

The service was started with the Swift source explicitly added to `PYTHONPATH`. The model loaded successfully with two workers.

## Prompt History

The old AMI result was produced while the ASR prompt still contained Chinese-specific instructions such as:

```text
转写这段中文语音。
```

That prompt is invalid for an English AMI recording and explains the old large Chinese sections. The current prompt was changed to language-neutral English instructions:

```text
Transcribe the audio verbatim in the language that is spoken.
For English speech, output English only; for Chinese speech, output Chinese only.
Never translate, interpret, summarize, or add content.
```

The current prompt also requires uncertainty candidates to use the same language/script as the source span.

## Full Run Inventory

The current full-run API result has:

| Quantity | Value |
|---|---:|
| ASR turns | 144 |
| Revision events, active | 154 |
| Submitted revisions | 6 |
| Raw Latin characters | 13,156 |
| Raw CJK characters | 15 |
| Current Latin characters | 11,922 |
| Current CJK characters | 15 |

The CJK count is tiny relative to the old `IS1009a` result, so the previous large-scale English-to-Chinese pollution is substantially reduced in the current run. The exact text for all 144 raw/current turns is stored in the linked [result.json](../retrace_state/ami_eval/EN2002a_fixed/result.json).

## Real Sample Output

The following excerpts are copied from the actual `EN2002a_fixed/result.json`.
`RAW` is the immutable first-pass recognition and `CURRENT` is the ReTrace
projection. The complete 144-turn sample is in the linked JSON; these excerpts
show every important failure type without creating a second copy of the result.

### t001: repeated-tail case

```text
Time: 1.0-30.4
RAW: Wonder how much of the meetings is talking about the stuff of the meetings. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah.
CURRENT: Wonder how much of the meetings is talking about the stuff of the meetings.
```

The prefix is plausible English; the repeated `Yeah` suffix is a genuine
degeneration pattern. Removing only that suffix is defensible.

### t008: unchanged Chinese contamination

```text
Time: 115.8-116.4
RAW: 哦，WeChat。
CURRENT: 哦，WeChat。
```

This is an ASR language error, not a ReTrace-created error, because both values
are identical.

### t013: unchanged Chinese contamination

```text
Time: 177.4-177.9
RAW: 对，我感觉。
CURRENT: 对，我感觉。
```

Again, ReTrace did not change this Turn.

### t014: English whole-turn rewrite

```text
Time: 182.3-197.2
RAW: So otherwise you just have a right click or two. Yeah, yeah, but um, I don't know about you, but usually in Windows, right click. Doesn't do anything, does it? It opens a menu. A menu, yeah. Yeah. So. Yeah. So that would be. Yeah.
CURRENT: Yeah, yeah, but I don't know.I don't know about you, but usually in Windows, right click doesn't do anything, does it? It opens a menu.Maybe yeah. Yeah. So, but yeah.
```

The result remains English but is an unconstrained second transcription. No
local error span justifies replacing the original Turn.

### t021: English content deleted

```text
Time: 298.2-304.6
RAW: It's true. You know how to yeah. Involve the speaking characters. Yeah, that's true. It would be nice to have it in connect interconnect. Yeah.
CURRENT: Yeah. Yeah. Yeah, that's true. Yeah.
```

Most of the raw English content disappears. This is a false positive, not a
successful correction.

### t041: unchanged mixed-script fragment

```text
Time: 591.4-591.9
RAW: 五G是。
CURRENT: 五G是。
```

This remains an English-session language error, but it was not introduced by
ReTrace.

### t054: unsupported English insertion

```text
Time: 769.6-780.5
RAW: Yeah, that's true. They're They're definitely Yeah, they're definitely going to be in the same window. And probably mutually. Yeah, that that was a.
CURRENT: Yeah, that's true. They're they're definitely using the same window, and so we can get a lot. And probably mutually. Um. Dispute. Yeah, that that was a.
```

`Dispute` is new content, not a bounded replacement supported by a local span.

### t071: strongest hallucination example

```text
Time: 962.9-982.0
RAW: And browser. Yes. And for meeting. Hope you never have to look in my directory. It's just pure chaos. No, I'm I'm extremely organized. I've got folders. I think I love folders. Mine is serious chaos. Files everywhere.
CURRENT: M browser. Yes. M for meeting.Monitor it, but it's just pure chaos.I've got balls. I think I love balls.[Laughter]
```

`I've got balls` and `[Laughter]` are not present in the raw output. This is a
clear hallucinating whole-turn recovery and should never be accepted as an
ordinary revision.

### t086: second repeated-tail case

```text
Time: 1194.0-1208.2
RAW: Which is only a page long, but it's quite difficult to write a page. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah.
CURRENT: Which is only a page long, but it's quite difficult to write a page.
```

This is the second bounded repeated-tail cleanup.

### Complete real output

For the full real sample, including every timestamp, raw recognition, current
projection, uncertainty metadata, and revision event, open:

- [EN2002a result.json](../retrace_state/ami_eval/EN2002a_fixed/result.json)

The corresponding complete ground truth remains:

- [EN2002a.stm](../../AMI/STM/EN2002a.stm)

No ground-truth text has been rewritten into the report. The report embeds only
verbatim system-output excerpts and links the authoritative source files.

## Actual ReTrace Revisions

Only the following six active revisions changed the current transcript.

### t001: repeated-tail cleanup

Raw output contained the valid opening sentence followed by a long repeated tail:

```text
RAW: Wonder how much of the meetings is talking about the stuff of the meetings. Yeah. Yeah. Yeah. Yeah. Yeah. ...
CURRENT: Wonder how much of the meetings is talking about the stuff of the meetings.
```

This is the one case where a repeated-tail cleanup is well motivated. The actual event is `audio-degeneration-recovery` with reason `repeated_tail`.

### t014: unsafe whole-turn open recovery

```text
RAW: So otherwise you just have a right click or two. Yeah, yeah, but um, I don't know about you, but usually in Windows, right click. Doesn't do anything, does it? It opens a menu. A menu, yeah. So, but yeah.
CURRENT: Yeah, yeah, but I don't know.I don't know about you, but usually in Windows, right click doesn't do anything, does it? It opens a menu.Maybe yeah. Yeah. So, but yeah.
```

The output is English, but it is a second free-form rewrite. It is not a bounded correction of a known span, and it can delete or duplicate valid material.

### t021: unsafe whole-turn open recovery

```text
RAW: It's true. You know how to yeah. Involve the speaking characters. Yeah, that's true. It would be nice to have it in connect interconnect. Yeah.
CURRENT: Yeah. Yeah. Yeah, that's true. Yeah.
```

The current text deletes most of the raw content. This is not an acceptable ASR correction without a timestamped local replacement and decisive evidence.

### t054: unsafe whole-turn open recovery

```text
RAW: Yeah, that's true. They're They're definitely Yeah, they're definitely going to be in the same window. And probably mutually. Yeah, that that was a.
CURRENT: Yeah, that's true. They're they're definitely using the same window, and so we can get a lot. And probably mutually. Um. Dispute. Yeah, that that was a.
```

The replacement introduces a new interpretation (`Dispute`) and changes the sentence structure. It is not a local evidence-backed repair.

### t071: hallucination-producing whole-turn recovery

```text
RAW: And browser. Yes. And for meeting. Hope you never have to look in my directory. It's just pure chaos. No, I'm I'm extremely organized. I've got folders. I think I love folders. Many folders.
CURRENT: M browser. Yes. M for meeting. Monitor it, but it's just pure chaos. I've got balls. I think I love balls. [Laughter]
```

`I've got balls` is a clear hallucination relative to the raw text and the meeting topic. This is the strongest direct evidence that whole-turn open recovery is unsafe.

### t086: repeated-tail cleanup

```text
RAW: Which is only a page long, but it's quite difficult to write a page. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah.
CURRENT: Which is only a page long, but it's quite difficult to write a page.
```

This is another reasonable repeated-tail removal, provided the repeated suffix detector has verified the suffix boundary.

## Chinese Output In The English Run

The current full run has only 15 CJK characters in raw and current text. Examples are short fragments such as:

```text
哦，WeChat。
对，我感觉。
嗯。
五G是。
第一。
JH: 不, 在中间。
```

These are not large translated paragraphs. They should still be treated as language errors in an English-only evaluation, but they are categorically different from the old result where whole English turns became Chinese paragraphs such as:

```text
是项目融资,呃,当然,我们设计这个东西的时候,我们都要考虑。
```

The current language-preserving prompt and candidate filtering therefore fixed the major cross-language failure, but the run still needs a final English-only output validator if the product requirement is absolute.

## Why The Old Metrics Were Misleading

The old `ami_time_sorted_metrics.json` contains fields such as:

```json
"raw_asr": {
  "cer": 0.5174206968278731,
  "wer": 0.5714285714285714,
  "reference_words": 2009,
  "hypothesis_words": 1143
}
```

Those numbers are not reliable ordinary WER/CER for this recording. The per-turn details show why:

```text
t003: reference_words=91, hypothesis_words=11
t029: reference_words=1,  hypothesis_words=19, WER=18.0
t030: reference_words=111, hypothesis_words=10
t031: reference_words=48,  hypothesis_words=4
```

A single ASR chunk is being compared with multiple speakers' utterances that merely overlap its time range. A one-word reference compared with 19 system words produces a mathematically valid edit-distance ratio but not a meaningful ASR WER. The high aggregate number therefore mixes:

- actual ASR omissions;
- wrong-speaker assignment;
- overlapping speech;
- chunk/reference boundary mismatch;
- old Chinese translation behavior;
- ReTrace whole-turn rewrites.

Sorting STM by time changes the file order, but it does not solve these other problems.

## Correct Metric Interpretation

The following metrics must be kept separate:

1. **Official AMI multi-speaker metric**: use system CTM with word-level times and speaker labels, reference STM, and the official AMI/NIST scoring recipe (`asclite`/the corresponding recipe). The current Qwen response does not contain a valid system CTM, so an official cpWER cannot honestly be reported yet.
2. **Chunk-aligned diagnostic**: useful for debugging, but must identify ambiguous overlap turns and must not call its result official WER.
3. **English-only content metrics**: calculate on turns whose reference and hypothesis are both English and whose alignment is not ambiguous. This is useful for isolating language failure, but it is not a full AMI score.
4. **ReTrace delta**: compare raw versus current on exactly the same eligible aligned units. ReTrace should not be credited or blamed using a score that changes the reference alignment between the two runs.

## What The Current Run Actually Shows

The evidence supports these conclusions:

- Qwen-Omni recognizes many English spans well. Examples in the result include long passages such as the discussion of the remote control, the light sensor, and the meeting plan.
- The original baseline had severe under-transcription on some chunks and occasional language switching.
- The current prompt substantially reduces large English-to-Chinese conversions.
- ReTrace currently has negative risk on English meetings because `low_diversity` can trigger an open whole-turn recovery.
- Two repeated-tail cleanups are defensible; the four low-diversity whole-turn replacements are not.
- The `t071` hallucination proves that “a different retranscription was returned” is not enough evidence to replace a transcript.
- A correct ReTrace evaluation should report both a protected baseline and a revision delta. The protected baseline should retain raw text whenever no local verified correction exists.

## Required Code Changes

### 1. Disable low-diversity whole-turn replacement

`low_diversity` is a suspicion signal, not proof of truncation. It should route to `DEFER` or targeted local verification. It must not create a whole-turn `REVISE_CURRENT` event.

### 2. Keep only bounded repeated-tail cleanup

Repeated-tail cleanup may remain, but it must:

- identify the repeated suffix precisely;
- preserve the non-repeated prefix byte-for-byte;
- verify that the suffix is genuinely repetitive;
- never replace the complete turn with a free-form second transcript.

### 3. Require local alignment for retranscription

A retranscription candidate must be converted into a local span proposal. If no bounded alignment exists, return `DEFER` and keep raw text.

### 4. Enforce language after every recovery path

For an English-only turn:

- reject Chinese recovery text;
- reject mixed-script candidates that introduce CJK content;
- record `language_mismatch` as a metric;
- keep raw text unchanged.

### 5. Add a no-regression metric report

Every evaluation should include:

```text
raw_text_metrics
current_text_metrics
eligible_turn_count
ambiguous_turn_count
unscorable_turn_count
language_mismatch_count
whole_turn_recovery_count
local_revision_count
repeated_tail_cleanup_count
```

### 6. Add an official scoring export

When Qwen can expose word-level timestamps, export:

```text
<recording> <channel> <speaker> <start> <duration> <confidence> <word>
```

as CTM and run the official AMI scorer. Until then, label all local reports as diagnostics.

## Recommended Acceptance Gates

For an English AMI run, do not accept a ReTrace revision unless all conditions hold:

- same language/script as the target span;
- target span is present verbatim in the target current text;
- replacement is local, not a whole-turn rewrite;
- candidate set is closed and acoustically verified;
- context evidence points to the same span;
- no ambiguity caused by unresolved speaker overlap;
- raw text is not replaced merely because another ASR produced a different sentence.

A repeated-tail deletion is a special controlled exception, not a general retranscription permission.

## Final Assessment

The old result file is not a valid basis for claiming that Qwen-Omni has 57% WER or that ReTrace improves it. The current evidence says:

- the English prompt and language boundary fix worked substantially;
- the AMI scoring path is still not official because system word-level speaker-attributed CTM is missing;
- ReTrace is not yet demonstrating reliable English improvement;
- open whole-turn recovery is the main product risk;
- the safest current policy is to preserve raw English and defer unless a local, same-language, audio-verified correction exists.

The most important next experiment is not another unqualified full-file WER. It is an ablation on the same aligned English turns:

```text
raw ASR
raw + repeated-tail cleanup only
raw + local verified revisions
raw + open whole-turn recovery (for diagnosis only)
```

This will show exactly whether the ReTrace mechanism helps, instead of hiding ASR, diarization, alignment, and hallucination errors inside one aggregate number.
