# ReTrace ICASSP full-manuscript design

## Objective

Expand the standalone `ReTrace-ICASSP-Paper` manuscript from a two-page method overview into a four-page ICASSP technical manuscript plus references, without fabricating empirical results.

## Scope

The manuscript will retain the IEEE conference class and the KDD-reference-inspired `main.tex` plus `body/*.tex` organization. It will describe the implementation currently on `feature/qwen-evolve-integrations`:

1. immutable streaming ASR turns with timestamps and uncertainty/candidate information;
2. local-first deterministic nomination from later candidate mentions and near-form conflicts;
3. localized historical audio verification with strong-score thresholds;
4. sparse DeepSeek escalation only for weak but top-ranked audio candidates and explicit triggers;
5. a separately labelled open re-transcription route for degenerate first-pass turns; and
6. append-only event replay and a symbolic entity fallback reported separately from audio verification.

## Manuscript structure

- Introduction: concrete streaming ambiguity problem, safety/cost motivation, and precise contributions.
- Related work: ASR rescoring, contextual correction, acoustic-conditioned correction, and audio grounding.
- Problem formulation: notation for turns, evidence, candidate sets, event state, and resolver paths.
- Method: deterministic nomination, local-window construction, strong/weak audio gates, sparse LLM adjudication, degenerate recovery, audit semantics, and complexity/cost.
- Experimental protocol: benchmark construction, annotation schema, baselines, metrics, ablations, statistical tests, and reporting templates with no invented values.
- Analysis and limitations: resolver-path error taxonomy, example trace, failure modes, and limitations.
- Conclusion.

## Claims and exclusions

- Do not claim that every revision is strictly an original ASR/N-best closed-set decision: later evidence can create a bounded decision-time candidate pair, and degenerate recovery is open re-transcription.
- Do not claim that every revision is audio-grounded: symbolic entity fallback is disclosed and evaluated separately.
- Do not report CER/WER, accuracy, cost, or significance figures until experiments are run.
- Do not portray manual confirmation or manual undo as part of the method.

## Acceptance criteria

1. The manuscript compiles with pdfLaTeX/BibTeX using `IEEEtran[conference]`.
2. It is approximately four IEEE two-column pages before references, with no overfull boxes or undefined citations.
3. Every algorithmic path in the paper maps to a named current resolver path in code.
4. The evaluation section is executable as a protocol and uses result placeholders rather than fictional outcomes.
