# ReTrace V2 Agentic Acoustic Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an independent, auditable ReTrace V2 agent that selectively acquires acoustic evidence and achieves measurable natural AISHELL-4 CER reduction over frozen Raw MOSS.

**Architecture:** A pure-Python core operates on immutable typed records and exposes ASR, overlap, and separation tools through protocols. A bounded agent loop chooses relistening, expansion, separation, comparison, keep, commit, rollback, or stop. Inference artifacts are finalized before a separate scorer may read references.

**Tech Stack:** Python 3.10+, dataclasses, JSONL, NumPy, python-Levenshtein, soundfile, pytest; optional FunASR/Paraformer and GSS/MVDR server adapters.

---

## File map

- `retrace_v2/schemas.py`: immutable V2 records and model roles.
- `retrace_v2/text.py`: normalization, alignment, and edit replay.
- `retrace_v2/candidates.py`: closed candidate graph.
- `retrace_v2/tools.py`: evidence-ASR and separation protocols.
- `retrace_v2/agent.py`: bounded Agent state machine and risk decisions.
- `retrace_v2/artifacts.py`: inference manifests and JSONL artifacts.
- `retrace_v2/scoring.py`: offline CER, S/D/I, oracle, and revision metrics.
- `retrace_v2/adapters/`: Paraformer, RTTM, and separation adapters.
- `scripts/run_retrace_v2.py`: inference-only CLI with no GT argument.
- `scripts/score_retrace_v2.py`: offline reference-aware scorer.
- `tests_v2/`: isolated unit and integration tests.

### Task 1: Immutable schemas and model isolation

**Files:**
- Create: `retrace_v2/__init__.py`
- Create: `retrace_v2/schemas.py`
- Test: `tests_v2/test_schemas.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_raw_segment_is_immutable():
    row = RawSegment("s1", 0.0, 1.0, "spk0", "你好", "moss")
    with pytest.raises(FrozenInstanceError):
        row.text = "再见"


def test_manifest_rejects_same_baseline_and_evidence_identity():
    with pytest.raises(ValueError, match="independent"):
        RunManifest.create(
            run_id="r1",
            models={ModelRole.BASELINE_ASR: "moss", ModelRole.EVIDENCE_ASR: "moss"},
            rttm_source="predicted",
        )
```

- [ ] **Step 2: Run the test and see the expected import failure**

Run: `.venv/bin/python -m pytest -q tests_v2/test_schemas.py`  
Expected: FAIL with `ModuleNotFoundError: retrace_v2`.

- [ ] **Step 3: Implement the frozen records and validation**

```python
class ModelRole(str, Enum):
    BASELINE_ASR = "baseline_asr"
    EVIDENCE_ASR = "evidence_asr"
    SEMANTIC_RANKER = "semantic_ranker"
    SEPARATION_BACKEND = "separation_backend"


@dataclass(frozen=True)
class RawSegment:
    segment_id: str
    start_sec: float
    end_sec: float
    speaker: str
    text: str
    model_id: str


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    models: Mapping[ModelRole, str]
    rttm_source: str

    @classmethod
    def create(cls, *, run_id, models, rttm_source):
        if models[ModelRole.BASELINE_ASR].lower() == models[ModelRole.EVIDENCE_ASR].lower():
            raise ValueError("baseline and evidence ASR must be independent")
        if rttm_source not in {"none", "predicted", "oracle"}:
            raise ValueError("invalid rttm_source")
        return cls(run_id, MappingProxyType(dict(models)), rttm_source)
```

- [ ] **Step 4: Run `.venv/bin/python -m pytest -q tests_v2/test_schemas.py`; expect PASS.**

- [ ] **Step 5: Commit**

```bash
git add retrace_v2/__init__.py retrace_v2/schemas.py tests_v2/test_schemas.py
git commit -m "feat(v2): add immutable experiment schemas"
```

### Task 2: Explicit character edits and closed candidates

**Files:**
- Create: `retrace_v2/text.py`
- Create: `retrace_v2/candidates.py`
- Test: `tests_v2/test_candidates.py`

- [ ] **Step 1: Write failing tests for all CER edit types**

```python
def test_candidates_cover_all_cer_edit_types():
    sub = candidates_from_hypothesis("方按", "方案", source="paraformer")
    extra = candidates_from_hypothesis("这个个方案", "这个方案", source="paraformer")
    missed = candidates_from_hypothesis("这个案", "这个方案", source="paraformer")
    assert {edit.kind for edit in sub[1].edits} == {"SUB"}
    assert "INS" in {edit.kind for edit in extra[1].edits}
    assert "DEL" in {edit.kind for edit in missed[1].edits}
    assert apply_edits("方按", sub[1].edits) == "方案"
    assert sub[0].candidate_text == "方按" and sub[0].edits == ()
```

- [ ] **Step 2: Run `.venv/bin/python -m pytest -q tests_v2/test_candidates.py`; expect missing-module FAIL.**

- [ ] **Step 3: Implement alignment, replay, and Raw preservation**

```python
def align_edits(raw: str, candidate: str) -> tuple[CharacterEdit, ...]:
    edits = []
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, normalize(raw), normalize(candidate), autojunk=False).get_opcodes():
        if tag != "equal":
            kind = {"replace": "SUB", "delete": "INS", "insert": "DEL"}[tag]
            edits.append(CharacterEdit(kind, i1, i2, candidate[j1:j2]))
    return tuple(edits)


def candidates_from_hypothesis(raw, hypothesis, *, source):
    keep = EditCandidate.keep(raw)
    if normalize(raw) == normalize(hypothesis):
        return (keep,)
    return (keep, EditCandidate(raw, hypothesis, align_edits(raw, hypothesis), (source,)))
```

- [ ] **Step 4: Run `.venv/bin/python -m pytest -q tests_v2/test_candidates.py`; expect PASS.**

- [ ] **Step 5: Commit**

```bash
git add retrace_v2/text.py retrace_v2/candidates.py tests_v2/test_candidates.py
git commit -m "feat(v2): build closed acoustic candidate graph"
```

### Task 3: Evidence tools and predicted RTTM routing

**Files:**
- Create: `retrace_v2/tools.py`
- Create: `retrace_v2/adapters/__init__.py`
- Create: `retrace_v2/adapters/rttm.py`
- Test: `tests_v2/test_tools_and_rttm.py`

- [ ] **Step 1: Write failing protocol and overlap tests**

```python
def test_predicted_rttm_yields_overlap(tmp_path):
    path = tmp_path / "predicted.rttm"
    path.write_text(
        "SPEAKER rec 1 1.0 2.0 <NA> <NA> spk0 <NA> <NA>\n"
        "SPEAKER rec 1 2.0 2.0 <NA> <NA> spk1 <NA> <NA>\n"
    )
    assert overlap_windows(parse_rttm(path, expected_recording="rec")) == [(2.0, 3.0)]


def test_rttm_rejects_wrong_recording(tmp_path):
    path = tmp_path / "predicted.rttm"
    path.write_text("SPEAKER other 1 0 1 <NA> <NA> spk0 <NA> <NA>\n")
    with pytest.raises(ValueError, match="recording"):
        parse_rttm(path, expected_recording="rec")
```

- [ ] **Step 2: Run `.venv/bin/python -m pytest -q tests_v2/test_tools_and_rttm.py`; expect missing-adapter FAIL.**

- [ ] **Step 3: Implement typed tool protocols and overlap extraction**

```python
class EvidenceASR(Protocol):
    model_id: str
    def transcribe(self, audio_path: Path, start_sec: float, end_sec: float) -> tuple[EvidenceHypothesis, ...]: ...


class Separator(Protocol):
    backend_id: str
    def separate(self, audio_path: Path, start_sec: float, end_sec: float) -> tuple[Path, ...]: ...


def overlap_windows(turns):
    events = sorted((x, delta) for t in turns for x, delta in ((t.start_sec, 1), (t.end_sec, -1)))
    return intervals_where_active_count_is_at_least_two(events)
```

- [ ] **Step 4: Run `.venv/bin/python -m pytest -q tests_v2/test_tools_and_rttm.py`; expect PASS.**

- [ ] **Step 5: Commit**

```bash
git add retrace_v2/tools.py retrace_v2/adapters tests_v2/test_tools_and_rttm.py
git commit -m "feat(v2): add acoustic tools and RTTM routing"
```

### Task 4: Bounded Agent loop and failure-to-KEEP policy

**Files:**
- Create: `retrace_v2/agent.py`
- Test: `tests_v2/test_agent.py`

- [ ] **Step 1: Write failing Agent action tests with fake tools**

```python
def test_agent_relistens_then_commits_supported_candidate():
    tools = FakeTools(evidence=[hypothesis("方案", score=0.92)])
    result = ReTraceAgent(policy=ThresholdPolicy(commit_margin=0.2)).run(region("方按"), tools)
    assert [a.kind for a in result.actions] == ["RELISTEN", "COMPARE_EVIDENCE", "COMMIT"]
    assert result.final_text == "方案"


def test_agent_separates_only_after_overlap_conflict():
    tools = FakeTools(evidence=[hypothesis("方按", score=0.51)], separated=[hypothesis("方案", score=0.95)])
    result = ReTraceAgent(policy=ThresholdPolicy(commit_margin=0.2)).run(region("方按", overlap=0.9), tools)
    assert "SEPARATE" in [a.kind for a in result.actions]
    assert result.final_text == "方案"


def test_tool_failure_keeps_raw():
    result = ReTraceAgent().run(region("不可改"), FailingEvidenceTools())
    assert result.final_text == "不可改"
    assert result.decision.kind == "KEEP"
```

- [ ] **Step 2: Run `.venv/bin/python -m pytest -q tests_v2/test_agent.py`; expect missing-Agent FAIL.**

- [ ] **Step 3: Implement the budgeted state machine**

```python
while not state.terminal and state.tool_calls < policy.max_tool_calls:
    action = policy.next_action(state)
    try:
        state = execute(action, state, tools)
    except ToolFailure as error:
        state = state.record_failure(action, str(error)).keep("tool_failure")
if not state.terminal:
    state = state.keep("budget_exhausted")
return state.result()
```

`COMMIT` requires localized edits, non-semantic acoustic support, and a score
margin. Separation is legal only after high overlap probability or an observed
speaker conflict.

- [ ] **Step 4: Run `.venv/bin/python -m pytest -q tests_v2/test_agent.py tests_v2/test_candidates.py`; expect PASS.**

- [ ] **Step 5: Commit**

```bash
git add retrace_v2/agent.py tests_v2/test_agent.py
git commit -m "feat(v2): add bounded evidence acquisition agent"
```

### Task 5: Inference artifacts and offline diagnostic scoring

**Files:**
- Create: `retrace_v2/artifacts.py`
- Create: `retrace_v2/scoring.py`
- Test: `tests_v2/test_artifacts_and_scoring.py`

- [ ] **Step 1: Write failing leakage, CER, and oracle tests**

```python
def test_inference_manifest_rejects_reference_keys(tmp_path):
    with pytest.raises(ValueError, match="reference"):
        ArtifactWriter(tmp_path).write_manifest({"reference_dir": "/secret", "run_id": "r"})


def test_oracle_candidate_exposes_generator_headroom():
    row = scored_row(reference="方案", raw="方按", final="方按", candidates=["方按", "方案"])
    report = score_rows([row])
    assert report.raw.edits == 1
    assert report.final.edits == 1
    assert report.oracle_candidate.edits == 0


def test_pooled_cer_is_total_errors_over_total_reference_chars():
    rows = [scored_row("甲乙", "甲", "甲", ["甲"]), scored_row("丙", "", "", [""])]
    assert score_rows(rows).raw.cer == pytest.approx(2 / 3)
```

- [ ] **Step 2: Run `.venv/bin/python -m pytest -q tests_v2/test_artifacts_and_scoring.py`; expect missing-module FAIL.**

- [ ] **Step 3: Implement atomic artifacts and pooled metrics**

```python
def write_json_atomic(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def pooled_metric(pairs):
    totals = Counter()
    for reference, hypothesis in pairs:
        totals.update(character_edit_counts(reference, hypothesis))
        totals["reference_chars"] += len(normalize(reference))
    totals["cer"] = sum(totals[x] for x in ("substitutions", "deletions", "insertions")) / totals["reference_chars"]
    return totals
```

The oracle scorer runs only after inference artifacts are closed and writes only
under `scoring/`.

- [ ] **Step 4: Run `.venv/bin/python -m pytest -q tests_v2/test_artifacts_and_scoring.py`; expect PASS.**

- [ ] **Step 5: Commit**

```bash
git add retrace_v2/artifacts.py retrace_v2/scoring.py tests_v2/test_artifacts_and_scoring.py
git commit -m "feat(v2): isolate inference artifacts from scoring"
```

### Task 6: Separate inference and scoring CLIs

**Files:**
- Create: `scripts/run_retrace_v2.py`
- Create: `scripts/score_retrace_v2.py`
- Create: `tests_v2/fixtures/raw_moss.jsonl`
- Create: `tests_v2/fixtures/evidence.jsonl`
- Create: `tests_v2/fixtures/reference.jsonl`
- Test: `tests_v2/test_cli.py`

- [ ] **Step 1: Write a failing end-to-end CLI test**

```python
def test_inference_has_no_reference_option_and_scorer_is_separate(tmp_path):
    run = subprocess.run([
        sys.executable, "scripts/run_retrace_v2.py",
        "--raw", "tests_v2/fixtures/raw_moss.jsonl",
        "--evidence", "tests_v2/fixtures/evidence.jsonl",
        "--output", str(tmp_path / "run"),
    ], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    help_text = subprocess.run(
        [sys.executable, "scripts/run_retrace_v2.py", "--help"],
        capture_output=True, text=True,
    ).stdout.lower()
    assert "reference" not in help_text
    score = subprocess.run([
        sys.executable, "scripts/score_retrace_v2.py",
        "--run", str(tmp_path / "run"),
        "--reference", "tests_v2/fixtures/reference.jsonl",
    ])
    assert score.returncode == 0
```

- [ ] **Step 2: Run `.venv/bin/python -m pytest -q tests_v2/test_cli.py`; expect missing-script FAIL.**

- [ ] **Step 3: Implement independent entry points**

```python
# run_retrace_v2.py
parser.add_argument("--raw", required=True)
parser.add_argument("--evidence", required=True)
parser.add_argument("--predicted-rttm")
parser.add_argument("--output", required=True)
parser.add_argument("--max-tool-calls", type=int, default=3)

# score_retrace_v2.py
parser.add_argument("--run", required=True)
parser.add_argument("--reference", required=True)
```

Precomputed evidence lets policy experiments remain reproducible without rerunning
the acoustic model.

- [ ] **Step 4: Run `.venv/bin/python -m pytest -q tests_v2`; expect PASS.**

- [ ] **Step 5: Commit**

```bash
git add scripts/run_retrace_v2.py scripts/score_retrace_v2.py tests_v2
git commit -m "feat(v2): add reproducible inference and scoring CLIs"
```

### Task 7: Real Paraformer and optional separation adapters

**Files:**
- Create: `retrace_v2/adapters/paraformer.py`
- Create: `retrace_v2/adapters/separation.py`
- Test: `tests_v2/test_model_adapters.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Write adapter tests using injected fake runtimes**

```python
def test_paraformer_preserves_identity_and_window_time(tmp_path):
    runtime = FakeParaformerRuntime(text="方案", timestamps=[[0, 200], [200, 400]])
    rows = ParaformerAdapter("paraformer-zh", runtime=runtime).transcribe(
        tmp_path / "audio.wav", 1.0, 2.0
    )
    assert rows[0].model_id == "paraformer-zh"
    assert rows[0].start_sec >= 1.0


def test_separation_command_failure_is_explicit(tmp_path):
    adapter = CommandSeparator(["false"], backend_id="gss")
    with pytest.raises(ToolFailure, match="gss"):
        adapter.separate(tmp_path / "audio.wav", 0.0, 1.0)
```

- [ ] **Step 2: Run `.venv/bin/python -m pytest -q tests_v2/test_model_adapters.py`; expect missing-adapter FAIL.**

- [ ] **Step 3: Implement lazy model loading and isolated commands**

```python
class ParaformerAdapter:
    def __init__(self, model_id: str, *, runtime=None):
        self.model_id = model_id
        if runtime is None:
            from funasr import AutoModel
            runtime = AutoModel(model=model_id, disable_update=True)
        self.runtime = runtime

    def transcribe(self, audio_path, start_sec, end_sec):
        window_path = slice_audio_to_tempfile(audio_path, start_sec, end_sec)
        try:
            result = self.runtime.generate(input=str(window_path), batch_size_s=60)
            return normalize_funasr_result(result, self.model_id, start_sec, end_sec)
        finally:
            window_path.unlink(missing_ok=True)
```

Add a `v2-acoustic` optional dependency group for FunASR/ModelScope; no model is
downloaded by unit tests. The command separator captures exit code, stderr,
backend identity, output paths, and elapsed time.

- [ ] **Step 4: Run `.venv/bin/python -m pytest -q tests_v2`; expect PASS without network.**

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml retrace_v2/adapters tests_v2/test_model_adapters.py
git commit -m "feat(v2): add optional acoustic model adapters"
```

### Task 8: Three-recording candidate-funnel experiment

**Files:**
- Create on server: `/home/panyibo/results/retrace-v2/<run-id>/`
- Update: `docs/experiment-progress-2026-09-10.md`

- [ ] **Step 1: Verify the selected server read-only**

```bash
hostname
df -h /home/panyibo
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
test -d /home/panyibo/models/openmoss/MOSS-Transcribe-Diarize
test -d /home/panyibo/datasets/retrace_asr/aishell4_test20/test/wav
```

Expected: data and MOSS exist, usable GPU capacity and disk space are reported.
Do not install or copy models until these checks pass.

- [ ] **Step 2: Deploy committed V2 files without overwriting results**

Use Git fast-forward or a new timestamped source directory. Preserve
`/home/panyibo/results/retrace-moss-aishell4`. Record the exact commit in the run
manifest.

- [ ] **Step 3: Generate immutable Paraformer evidence for three development recordings**

Write evidence JSONL before scoring. Confirm every row includes a Paraformer
model identity, time bounds, source audio identity, score metadata, and no GT.

- [ ] **Step 4: Run inference, then score in a separate process**

```bash
python scripts/run_retrace_v2.py \
  --raw /home/panyibo/results/retrace-v2/dev3/raw_moss.jsonl \
  --evidence /home/panyibo/results/retrace-v2/dev3/evidence.jsonl \
  --predicted-rttm /home/panyibo/results/retrace-v2/dev3/predicted.rttm \
  --output /home/panyibo/results/retrace-v2/dev3/inference

python scripts/score_retrace_v2.py \
  --run /home/panyibo/results/retrace-v2/dev3/inference \
  --reference /home/panyibo/results/retrace-v2/dev3/reference.jsonl
```

Expected: scoring prints Raw, Final, and oracle-candidate pooled CER and S/D/I.

- [ ] **Step 5: Follow the diagnostic stop rule**

- Oracle recovery below 10%: change localization, windowing, evidence ASR, or
  separation; do not tune the gate.
- Oracle recovery at least 10%, final relative gain below 1%: improve the Agent
  policy using only development recordings.
- Large harmful edits: tighten localization, speaker consistency, and acoustic
  margins.
- At least 20 net errors removed with no large-span harm: expand to the inspected
  20-recording development set.

- [ ] **Step 6: Record the candidate funnel and timing**

Append sample IDs, Raw/Final/oracle S/D/I and CER, candidate funnel, action counts,
per-stage timing, model IDs, RTTM provenance, commit, and stop-rule decision to
`docs/experiment-progress-2026-09-10.md`.

- [ ] **Step 7: Commit only reproducible code and the report**

```bash
git add retrace_v2 scripts/run_retrace_v2.py scripts/score_retrace_v2.py tests_v2 docs/experiment-progress-2026-09-10.md
git commit -m "exp(v2): report three-recording acoustic recovery probe"
```

Never commit audio, model checkpoints, API keys, corpus references, or generated
run directories.

## Verification before expanding to 20 recordings

Run `.venv/bin/python -m pytest -q tests_v2`, followed by `git diff --check` and
`git status --short`. All V2 tests must pass, no whitespace errors may exist, and
only intentional files may be staged. Expansion is forbidden until the Stage 1
stop rule passes.
