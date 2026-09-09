# Agent 优先的实时 Memory 回溯修订实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal（目标）:** 将当前基于全局 N-gram 和写死词表提名候选的流程，改造成由 Agent 结合短期与长期 Memory 主动判断、可实时反向修订历史 ASR、并能自动回滚的系统。

**Architecture（架构）:** 保留 FastAPI、Qwen ASR、DeepSeek、音频窗口验证以及原始 Turn 不可修改的既有边界，把 `retrace.py` 拆成模型、账本、Memory、上下文判断、标定策略、证据解析和服务编排模块。实时接口先提交不可修改观察，再由按会话串行的后台协调器执行慢链路；所有提交都校验会话版本，并通过只追加事件回放当前字幕。

**Tech Stack（技术栈）:** Python 3.10+、dataclasses、FastAPI、Pydantic v2、`concurrent.futures`、JSON 持久化、pytest、TypeScript、Vite。

---

## 文件结构

- 创建 `backend/asr_agent/models.py`：领域数据类、序列化和旧会话兼容读取。
- 创建 `backend/asr_agent/storage.py`：会话级锁、原子 JSON 写入和乐观版本提交。
- 创建 `backend/asr_agent/ledger.py`：修订事件追加、回放、替代和自动回滚。
- 创建 `backend/asr_agent/memory.py`：短期 Memory 检索、长期 Memory 存储和自动巩固。
- 创建 `backend/asr_agent/context_judge.py`：Context Judge 输入输出协议、校验和无 LLM 降级策略。
- 创建 `backend/asr_agent/calibration.py`：证据概率标定、开发集门槛选择和动作策略。
- 创建 `backend/asr_agent/resolver.py`：竞争假设构建、针对性音频验证和结构化解析结果。
- 创建 `backend/asr_agent/realtime.py`：按会话串行、跨会话并行的后台分析协调器。
- 修改 `backend/asr_agent/integrations/deepseek.py`：增加结构化 Context Judge 调用。
- 修改 `backend/asr_agent/retrace.py`：缩减为 Agent-first 服务编排和兼容门面。
- 修改 `backend/asr_agent/server.py`：实时观察入队、状态查询和协调器生命周期。
- 修改 `backend/asr_agent/metrics.py`：增加修订召回、F2/F3、回滚与 Memory 污染指标。
- 修改 `frontend/src/main.ts`：消费实时分析状态、竞争假设和新版事件。
- 修改 `README.md`：描述 Agent-first 实时流程与运行方式。
- 新增对应的单元、集成和端到端测试文件。

### Task 1：抽离版本化领域模型并兼容旧会话

**Files:**
- Create: `backend/asr_agent/models.py`
- Modify: `backend/asr_agent/retrace.py:56-142`
- Create: `tests/test_retrace_models.py`

- [ ] **Step 1：编写旧 JSON 与新模型往返测试**

```python
from asr_agent.models import Session, WorkingHypothesis


def test_legacy_session_loads_with_versioned_memory_defaults():
    session = Session.from_dict({
        "session_id": "s1",
        "turns": [{"turn_id": "t1", "raw_text": "泰信", "current_text": "泰信"}],
        "verified_memory": {},
        "quarantine_memory": {},
        "revision_events": [],
    })

    assert session.version == 0
    assert session.memory_scope == "default"
    assert session.open_hypotheses == {}
    assert Session.from_dict(session.as_dict()).as_dict() == session.as_dict()


def test_working_hypothesis_keeps_competing_interpretations():
    hypothesis = WorkingHypothesis(
        hypothesis_id="h1",
        target_turn_ids=["t1"],
        current_interpretation="泰信",
        proposed_interpretation="泰康",
        alternatives=["泰信", "泰康"],
        created_version=1,
        last_evaluated_version=1,
    )
    assert hypothesis.alternatives == ["泰信", "泰康"]
    assert hypothesis.status == "active"
```

- [ ] **Step 2：运行测试并确认因模块不存在而失败**

Run: `uv run pytest tests/test_retrace_models.py -q`

Expected: FAIL，错误包含 `ModuleNotFoundError: No module named 'asr_agent.models'`。

- [ ] **Step 3：创建模型模块并迁移现有数据类**

在 `models.py` 中迁移 `EntityProfile`、`Hypothesis`、`Turn`、`RevisionEvent`、`Session`，并增加以下模型；所有 `from_dict` 对新增字段使用默认值：

```python
@dataclass
class EvidenceRef:
    turn_id: str
    kind: str
    value: str
    score: float | None = None


@dataclass
class WorkingHypothesis:
    hypothesis_id: str
    target_turn_ids: list[str]
    current_interpretation: str
    proposed_interpretation: str
    alternatives: list[str]
    supporting_evidence: list[EvidenceRef] = field(default_factory=list)
    contradicting_evidence: list[EvidenceRef] = field(default_factory=list)
    audio_windows: list[dict[str, Any]] = field(default_factory=list)
    score: float = 0.0
    status: str = "active"
    created_version: int = 0
    last_evaluated_version: int = 0


@dataclass
class MemoryBelief:
    belief_id: str
    subject: str
    predicate: str
    value: str
    aliases: list[str] = field(default_factory=list)
    confidence: float = 0.0
    status: str = "provisional"
    valid_from: str | None = None
    valid_to: str | None = None
    source_turn_ids: list[str] = field(default_factory=list)
    source_session_ids: list[str] = field(default_factory=list)
    evidence_kinds: list[str] = field(default_factory=list)
    supersedes: str | None = None


@dataclass
class Session:
    session_id: str
    version: int = 0
    memory_scope: str = "default"
    analysis_status: str = "idle"
    entities: dict[str, EntityProfile] = field(default_factory=dict)
    turns: list[Turn] = field(default_factory=list)
    working_beliefs: dict[str, MemoryBelief] = field(default_factory=dict)
    open_hypotheses: dict[str, WorkingHypothesis] = field(default_factory=dict)
    dependency_index: dict[str, list[str]] = field(default_factory=dict)
    verified_memory: dict[str, dict[str, str]] = field(default_factory=dict)
    quarantine_memory: dict[str, dict[str, str]] = field(default_factory=dict)
    revision_events: list[RevisionEvent] = field(default_factory=list)
```

给 `RevisionEvent` 增加带默认值的 `session_version: int = 0`、`supersedes_event_id: str | None = None`；在 `retrace.py` 中从 `asr_agent.models` 重新导出旧名称，保持 `from asr_agent.retrace import ReTraceService, EntityProfile` 可用。

- [ ] **Step 4：运行模型测试与现有服务测试**

Run: `uv run pytest tests/test_retrace_models.py tests/test_retrace_service.py -q`

Expected: PASS，且旧会话 JSON 无需迁移脚本即可加载。

- [ ] **Step 5：提交模型抽离**

```bash
git add backend/asr_agent/models.py backend/asr_agent/retrace.py tests/test_retrace_models.py
git commit -m "refactor: extract versioned retrace models"
```

### Task 2：实现原子会话存储与乐观版本提交

**Files:**
- Create: `backend/asr_agent/storage.py`
- Create: `tests/test_session_repository.py`
- Modify: `backend/asr_agent/retrace.py`

- [ ] **Step 1：编写原子更新、旧快照冲突和并发不丢 Turn 测试**

```python
from asr_agent.models import Session, Turn
from asr_agent.storage import SessionRepository, VersionConflict


def test_commit_rejects_stale_session_version(tmp_path):
    repo = SessionRepository(tmp_path)
    repo.create(Session("s"))
    stale = repo.load("s")
    repo.update("s", lambda session: session.turns.append(Turn("t1", "甲", "甲")))
    stale.turns.append(Turn("t2", "乙", "乙"))

    try:
        repo.commit(stale, expected_version=0)
    except VersionConflict as exc:
        assert "expected version 0, got 1" in str(exc)
    else:
        raise AssertionError("stale commit must fail")

    assert [turn.turn_id for turn in repo.load("s").turns] == ["t1"]


def test_update_increments_version_once(tmp_path):
    repo = SessionRepository(tmp_path)
    repo.create(Session("s"))
    updated = repo.update("s", lambda session: session.turns.append(Turn("t1", "甲", "甲")))
    assert updated.version == 1
    assert repo.load("s").version == 1
```

- [ ] **Step 2：运行测试并确认失败**

Run: `uv run pytest tests/test_session_repository.py -q`

Expected: FAIL，错误包含 `No module named 'asr_agent.storage'`。

- [ ] **Step 3：实现会话级锁、深拷贝读取和原子写入**

```python
class VersionConflict(RuntimeError):
    pass


class SessionRepository:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, RLock] = {}
        self._locks_guard = Lock()

    def _lock_for(self, session_id: str) -> RLock:
        with self._locks_guard:
            return self._locks.setdefault(session_id, RLock())

    def load(self, session_id: str) -> Session:
        with self._lock_for(session_id):
            path = self.root / f"{session_id}.json"
            return Session.from_dict(json.loads(path.read_text())) if path.exists() else Session(session_id)

    def create(self, session: Session) -> Session:
        with self._lock_for(session.session_id):
            self._write(session)
            return Session.from_dict(session.as_dict())

    def update(self, session_id: str, mutate: Callable[[Session], None]) -> Session:
        with self._lock_for(session_id):
            current = self.load(session_id)
            mutate(current)
            current.version += 1
            self._write(current)
            return Session.from_dict(current.as_dict())

    def commit(self, session: Session, *, expected_version: int) -> Session:
        with self._lock_for(session.session_id):
            current = self.load(session.session_id)
            if current.version != expected_version:
                raise VersionConflict(f"expected version {expected_version}, got {current.version}")
            session.version = expected_version + 1
            self._write(session)
            return Session.from_dict(session.as_dict())

    def _write(self, session: Session) -> None:
        path = self.root / f"{session.session_id}.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(session.as_dict(), ensure_ascii=False, indent=2))
        temporary.replace(path)
```

`SessionRepository` 只在锁内读写磁盘，向调用方返回反序列化后的副本。生产代码不得持有 Repository 内部对象并在锁外修改。

- [ ] **Step 4：让 `ReTraceService` 使用 Repository，但暂不改变决策逻辑**

将 `_load` 委托给 `repository.load`，将同步旧流程的最终保存改成 `repository.commit(session, expected_version=loaded_version)`；若发生 `VersionConflict`，当前同步兼容流程重新加载并重新执行该 Turn，不能覆盖新版本。

- [ ] **Step 5：运行存储和现有服务测试**

Run: `uv run pytest tests/test_session_repository.py tests/test_retrace_service.py -q`

Expected: PASS，JSON 文件仍能由现有 API 读取。

- [ ] **Step 6：提交版本化存储**

```bash
git add backend/asr_agent/storage.py backend/asr_agent/retrace.py tests/test_session_repository.py
git commit -m "feat: add optimistic session repository"
```

### Task 3：实现只追加修订账本与自动回滚

**Files:**
- Create: `backend/asr_agent/ledger.py`
- Modify: `backend/asr_agent/retrace.py`
- Create: `tests/test_revision_ledger.py`
- Modify: `tests/test_retrace_service.py`

- [ ] **Step 1：编写修订回放与回滚测试**

```python
from asr_agent.ledger import RevisionLedger
from asr_agent.models import RevisionEvent, Session, Turn
from asr_agent.storage import VersionConflict


def test_revision_and_rollback_replay_without_mutating_raw_text():
    session = Session(session_id="s", turns=[Turn("t1", "泰信方案", "泰信方案")])
    ledger = RevisionLedger()
    revise = RevisionEvent(
        event_id="e1", action="REVISE_HISTORY", target_turn_id="t1", source_turn_id="t2",
        span="泰信", before_text="泰信方案", after_text="泰康方案", replacement="泰康",
        entity_id=None, score=.82, evidence=["t2:泰康"], resolver="agent-audio",
    )
    ledger.append(session, revise, expected_version=0)
    rollback = ledger.rollback(session, event_id="e1", source_turn_id="t3", reason="later contradiction", expected_version=1)
    ledger.replay(session)

    assert session.turns[0].raw_text == "泰信方案"
    assert session.turns[0].current_text == "泰信方案"
    assert revise.active is True  # 旧事件不可修改，由 ROLLBACK 在回放时使其失效。
    assert rollback.action == "ROLLBACK"
    assert rollback.supersedes_event_id == "e1"
    assert session.version == 2
```

同时测试错误 `expected_version` 抛出 `VersionConflict`，重复 `event_id` 不会二次应用。

- [ ] **Step 2：运行测试并确认失败**

Run: `uv run pytest tests/test_revision_ledger.py -q`

Expected: FAIL，错误包含 `No module named 'asr_agent.ledger'`。

- [ ] **Step 3：实现账本最小接口**

```python
class RevisionLedger:
    def append(self, session: Session, event: RevisionEvent, *, expected_version: int) -> RevisionEvent:
        self.append_many(session, [event], expected_version=expected_version)
        return event

    def append_many(self, session: Session, events: list[RevisionEvent], *, expected_version: int) -> list[RevisionEvent]:
        if session.version != expected_version:
            raise VersionConflict(f"expected version {expected_version}, got {session.version}")
        existing = {item.event_id for item in session.revision_events}
        accepted = [event for event in events if event.event_id not in existing]
        if not accepted:
            return []
        for event in accepted:
            event.session_version = expected_version + 1
            session.revision_events.append(event)
        session.version += 1
        self.replay(session)
        return accepted

    def rollback(self, session: Session, *, event_id: str, source_turn_id: str,
                 reason: str, expected_version: int) -> RevisionEvent:
        if session.version != expected_version:
            raise VersionConflict(f"expected version {expected_version}, got {session.version}")
        target = next(item for item in session.revision_events if item.event_id == event_id and item.active)
        event = RevisionEvent(
            event_id=f"rollback-{event_id}-{expected_version + 1}", action="ROLLBACK",
            target_turn_id=target.target_turn_id, source_turn_id=source_turn_id,
            span=target.replacement or target.span, before_text=target.after_text,
            after_text=target.before_text, replacement=target.span, entity_id=target.entity_id,
            score=1.0, evidence=[], resolver="automatic-rollback", reason=reason,
            supersedes_event_id=event_id,
        )
        return self.append(session, event, expected_version=expected_version)
```

`replay` 先收集所有有效 `ROLLBACK.supersedes_event_id` 和后续事件的 `supersedes_event_id`，再从 `raw_text` 开始，仅应用未被替代的 `REVISE_CURRENT`、`REVISE_HISTORY`、旧版 `REVISE_TEXT` 和 `REVISE_ENTITY` 事件；`ROLLBACK` 本身不再次改写文本，也不修改旧事件字段。

- [ ] **Step 4：将 `ReTraceService._replay` 委托给账本并更新旧测试**

保留“没有人工撤销 API”的测试，但把“服务不能拥有任何撤销能力”的断言改为“只允许内部 `RevisionLedger.rollback`，FastAPI 不暴露人工端点”。

- [ ] **Step 5：运行账本和 API 回归测试**

Run: `uv run pytest tests/test_revision_ledger.py tests/test_retrace_service.py tests/test_retrace_api.py -q`

Expected: PASS。

- [ ] **Step 6：提交账本**

```bash
git add backend/asr_agent/ledger.py backend/asr_agent/retrace.py tests/test_revision_ledger.py tests/test_retrace_service.py
git commit -m "feat: add append-only revision ledger"
```

### Task 4：实现短期与长期 Memory

**Files:**
- Create: `backend/asr_agent/memory.py`
- Create: `tests/test_memory.py`
- Modify: `backend/asr_agent/retrace.py`

- [ ] **Step 1：编写相关检索、版本替代和跨会话持久化测试**

```python
from asr_agent.memory import LongTermMemoryRepository, MemoryConsolidator, MemoryRetriever
from asr_agent.models import MemoryBelief, Session, Turn


def test_retriever_returns_recent_open_and_relevant_long_term_memory(tmp_path):
    repo = LongTermMemoryRepository(tmp_path / "long_term")
    repo.save("acme", [MemoryBelief(
        belief_id="b1", subject="provider", predicate="name", value="泰康",
        confidence=.91, status="stable", source_session_ids=["old"],
    )])
    session = Session("s", memory_scope="acme", turns=[Turn("t1", "讨论保险", "讨论保险")])
    packet = MemoryRetriever(repo, recent_turn_limit=8, long_term_limit=24).retrieve(session, query="泰信保险")
    assert packet.recent_turns[0]["turn_id"] == "t1"
    assert packet.long_term_beliefs[0].belief_id == "b1"


def test_consolidator_supersedes_conflicting_belief_without_deleting_history(tmp_path):
    repo = LongTermMemoryRepository(tmp_path / "long_term")
    repo.save("acme", [MemoryBelief("old", "provider", "name", "泰信", confidence=.8, status="stable")])
    new = MemoryBelief(
        "new", "provider", "name", "泰康", confidence=.93, status="stable",
        source_session_ids=["s1", "s2"],
    )
    MemoryConsolidator(repo, promotion_threshold=.9).promote("acme", new)
    beliefs = {item.belief_id: item for item in repo.load("acme")}
    assert beliefs["old"].status == "superseded"
    assert beliefs["new"].supersedes == "old"


def test_long_term_failure_degrades_to_short_term_memory(tmp_path):
    repo = LongTermMemoryRepository(tmp_path / "long_term")
    repo.load = lambda _scope: (_ for _ in ()).throw(OSError("disk unavailable"))
    session = Session("s", turns=[Turn("t1", "继续讨论方案", "继续讨论方案")])
    packet = MemoryRetriever(repo).retrieve(session, query="方案")
    assert packet.recent_turns[0]["turn_id"] == "t1"
    assert packet.long_term_beliefs == []
```

- [ ] **Step 2：运行测试并确认失败**

Run: `uv run pytest tests/test_memory.py -q`

Expected: FAIL，错误包含 `No module named 'asr_agent.memory'`。

- [ ] **Step 3：实现 JSON Repository 与检索包**

```python
@dataclass
class MemoryPacket:
    recent_turns: list[dict[str, str]]
    working_beliefs: list[MemoryBelief]
    open_hypotheses: list[WorkingHypothesis]
    dependency_turn_ids: list[str]
    long_term_beliefs: list[MemoryBelief]


class MemoryRetriever:
    def __init__(self, repository: LongTermMemoryRepository, *, recent_turn_limit: int = 8,
                 long_term_limit: int = 24,
                 relevance_scorer: Callable[[str, MemoryBelief], float] | None = None) -> None:
        self.repository = repository
        self.recent_turn_limit = recent_turn_limit
        self.long_term_limit = long_term_limit
        self.relevance_scorer = relevance_scorer

    def retrieve(self, session: Session, *, query: str) -> MemoryPacket:
        recent = session.turns[-self.recent_turn_limit:]
        working_keys = {(belief.subject, belief.predicate) for belief in session.working_beliefs.values()}

        def relevance(belief: MemoryBelief) -> float:
            surfaces = [belief.value, *belief.aliases]
            exact = 1.0 if any(value.casefold() in query.casefold() for value in surfaces) else 0.0
            structural = 0.8 if (belief.subject, belief.predicate) in working_keys else 0.0
            semantic = self.relevance_scorer(query, belief) if self.relevance_scorer else 0.0
            return max(exact, structural, semantic, belief.confidence * 0.05)

        try:
            stored = self.repository.load(session.memory_scope)
        except (OSError, ValueError, json.JSONDecodeError):
            stored = []
        stable = [belief for belief in stored if belief.status == "stable"]
        relevant = sorted(stable, key=lambda belief: (-relevance(belief), -belief.confidence))[:self.long_term_limit]
        return MemoryPacket(
            recent_turns=[{"turn_id": item.turn_id, "raw_text": item.raw_text} for item in recent],
            working_beliefs=list(session.working_beliefs.values()),
            open_hypotheses=list(session.open_hypotheses.values()),
            dependency_turn_ids=list(dict.fromkeys(
                turn_id for ids in session.dependency_index.values() for turn_id in ids
            )),
            long_term_beliefs=relevant,
        )
```

Repository 使用 `scope.json`，先写同目录临时文件再 `replace`，并用 `MemoryBelief.from_dict` 读取。禁止在检索器中加入 CJK 二字滑窗或停用词集合。

- [ ] **Step 4：实现自动巩固和版本替代**

`MemoryConsolidator.promote` 只接受 `confidence >= promotion_threshold` 且至少包含两个独立 `source_session_ids` 的普通信念；`evidence_kinds` 包含 `audio_verification` 的实体别名允许一个会话来源。相同 `subject + predicate` 且值不同的有效信念被标记为 `superseded`，新信念记录 `supersedes`。

- [ ] **Step 5：运行 Memory 测试**

Run: `uv run pytest tests/test_memory.py -q`

Expected: PASS。

- [ ] **Step 6：提交 Memory 层**

```bash
git add backend/asr_agent/memory.py backend/asr_agent/retrace.py tests/test_memory.py
git commit -m "feat: add versioned short and long term memory"
```

### Task 5：定义 Context Judge 协议并接入 DeepSeek

**Files:**
- Create: `backend/asr_agent/context_judge.py`
- Modify: `backend/asr_agent/integrations/deepseek.py`
- Create: `tests/test_context_judge.py`

- [ ] **Step 1：编写结构化输出校验测试**

```python
from asr_agent.context_judge import ContextJudgment, normalize_judgment


def test_normalizer_accepts_conflict_with_targeted_focus():
    result = normalize_judgment({
        "outcome": "CONFLICT",
        "confidence": .84,
        "reason": "后文稳定使用泰康",
        "focus": [{
            "target_turn_id": "t1", "span": "泰信",
            "proposed": "泰康", "alternatives": ["泰信", "泰康"],
        }],
    }, valid_turn_ids={"t1"}, text_by_turn={"t1": "采用泰信方案"})
    assert isinstance(result, ContextJudgment)
    assert result.focus[0].span == "泰信"


def test_normalizer_rejects_unquoted_target_span():
    result = normalize_judgment({
        "outcome": "CONFLICT", "confidence": .9, "reason": "猜测",
        "focus": [{"target_turn_id": "t1", "span": "不存在", "proposed": "泰康", "alternatives": ["不存在", "泰康"]}],
    }, valid_turn_ids={"t1"}, text_by_turn={"t1": "采用泰信方案"})
    assert result.outcome == "UNCERTAIN"
    assert result.focus == []
```

- [ ] **Step 2：运行测试并确认失败**

Run: `uv run pytest tests/test_context_judge.py -q`

Expected: FAIL，错误包含 `No module named 'asr_agent.context_judge'`。

- [ ] **Step 3：实现判断协议和无 LLM 降级策略**

```python
@dataclass
class FocusProposal:
    target_turn_id: str
    span: str
    proposed: str
    alternatives: list[str]
    relation: str = "asr_conflict"


@dataclass
class ContextJudgment:
    outcome: str
    confidence: float
    reason: str
    focus: list[FocusProposal] = field(default_factory=list)


class ExplicitSignalFallbackJudge:
    def __call__(self, *, current_turn: Turn, memory: MemoryPacket) -> ContextJudgment:
        signals = current_turn.meta.get("asr_signals", {})
        confidence = signals.get("confidence", {})
        nbest = signals.get("nbest", [])
        if nbest or any(value < .65 for value in confidence.values()):
            return ContextJudgment("UNCERTAIN", .5, "explicit_asr_uncertainty")
        return ContextJudgment("CONSISTENT", .6, "no_explicit_conflict")
```

降级策略只使用 ASR 明确提供的置信度和 N-best，不自行切中文二字词。

- [ ] **Step 4：在 DeepSeek 适配器中增加 `judge_context`**

使用结构化 JSON 输入 `{current_turn, recent_turns, working_beliefs, open_hypotheses, long_term_beliefs}`，系统提示要求只输出 `CONSISTENT|NOVEL|CONFLICT|UNCERTAIN` 及最短可疑片段。调用失败返回 `UNCERTAIN`，所有输出必须经过 `normalize_judgment` 验证；目标 Turn 必须存在，`span` 必须逐字出现在目标原始文本中，`proposed` 必须非空且不同。

- [ ] **Step 5：运行 Context Judge 与 DeepSeek 单元测试**

Run: `uv run pytest tests/test_context_judge.py -q`

Expected: PASS，并确认异常 JSON、越界 Turn 和虚构片段均降级为 `UNCERTAIN`。

- [ ] **Step 6：提交 Context Judge**

```bash
git add backend/asr_agent/context_judge.py backend/asr_agent/integrations/deepseek.py tests/test_context_judge.py
git commit -m "feat: add memory-aware context judge"
```

### Task 6：实现概率标定与高召回动作策略

**Files:**
- Create: `backend/asr_agent/calibration.py`
- Create: `tests/test_calibration.py`

- [ ] **Step 1：编写概率、差距和门槛选择测试**

```python
from asr_agent.calibration import (
    CalibrationSample, DecisionPolicy, EvidenceFeatures, LinearLogitCalibrator, PolicyThresholds,
    fit_logit_calibrator,
)


def test_policy_revises_with_audio_support_and_clear_margin():
    policy = DecisionPolicy(PolicyThresholds(
        suspect=.35, relisten=.45, revise=.62, long_memory=.88, margin=.08,
    ))
    decision = policy.decide(
        target_is_current=False,
        best_probability=.74,
        second_probability=.51,
        audio_verified=True,
    )
    assert decision == "REVISE_HISTORY"


def test_policy_defers_text_only_conflict():
    policy = DecisionPolicy(PolicyThresholds(.35, .45, .62, .88, .08))
    assert policy.decide(False, .9, .1, audio_verified=False) == "DEFER"


def test_fitted_calibrator_ranks_supported_revision_above_unsupported_one():
    positive = EvidenceFeatures(.9, .9, .8, .7, .8)
    negative = EvidenceFeatures(.2, .1, .2, .2, .1)
    calibrator = fit_logit_calibrator([
        CalibrationSample(positive, 1), CalibrationSample(positive, 1),
        CalibrationSample(negative, 0), CalibrationSample(negative, 0),
    ], epochs=400, learning_rate=.1)
    assert calibrator.predict(positive) > calibrator.predict(negative)
    assert LinearLogitCalibrator.from_dict(calibrator.as_dict()).as_dict() == calibrator.as_dict()
```

- [ ] **Step 2：运行测试并确认失败**

Run: `uv run pytest tests/test_calibration.py -q`

Expected: FAIL，错误包含 `No module named 'asr_agent.calibration'`。

- [ ] **Step 3：实现标定特征、逻辑函数和动作策略**

```python
@dataclass
class EvidenceFeatures:
    agent_support: float
    audio_support: float
    short_memory_support: float
    long_memory_support: float
    independent_support: float


@dataclass
class PolicyThresholds:
    suspect: float
    relisten: float
    revise: float
    long_memory: float
    margin: float


class LinearLogitCalibrator:
    def __init__(self, bias: float, weights: dict[str, float]) -> None:
        self.bias = bias
        self.weights = weights

    def predict(self, features: EvidenceFeatures) -> float:
        z = self.bias + sum(self.weights.get(name, 0.0) * value for name, value in asdict(features).items())
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))

    def as_dict(self) -> dict[str, object]:
        return {"bias": self.bias, "weights": self.weights}

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> "LinearLogitCalibrator":
        return cls(float(raw["bias"]), {str(key): float(value) for key, value in dict(raw["weights"]).items()})


@dataclass
class CalibrationSample:
    features: EvidenceFeatures
    label: int


def fit_logit_calibrator(samples: list[CalibrationSample], *, epochs: int = 1000,
                         learning_rate: float = .05) -> LinearLogitCalibrator:
    names = list(EvidenceFeatures.__dataclass_fields__)
    bias = 0.0
    weights = {name: 0.0 for name in names}
    for _epoch in range(epochs):
        bias_gradient = 0.0
        weight_gradients = {name: 0.0 for name in names}
        model = LinearLogitCalibrator(bias, weights)
        for sample in samples:
            error = model.predict(sample.features) - float(sample.label)
            bias_gradient += error
            for name in names:
                weight_gradients[name] += error * float(getattr(sample.features, name))
        scale = 1.0 / max(1, len(samples))
        bias -= learning_rate * bias_gradient * scale
        for name in names:
            weights[name] -= learning_rate * weight_gradients[name] * scale
    return LinearLogitCalibrator(bias, weights)


class DecisionPolicy:
    def decide(self, target_is_current: bool, best_probability: float,
               second_probability: float, *, audio_verified: bool) -> str:
        if best_probability < self.thresholds.suspect:
            return "KEEP_OLD"
        if not audio_verified:
            return "DEFER"
        if best_probability >= self.thresholds.revise and best_probability - second_probability >= self.thresholds.margin:
            return "REVISE_CURRENT" if target_is_current else "REVISE_HISTORY"
        return "DEFER"
```

- [ ] **Step 4：实现开发集概率标定与门槛选择**

先用开发集的结构化证据特征和二元正确性标签调用 `fit_logit_calibrator`，把得到的 `bias/weights` JSON 与数据集版本一起保存为实验产物。再由 `select_operating_point(samples, max_overcorrection=.10)` 在候选门槛网格上计算 revision precision、recall、F2 和 over-correction；只在 `overcorrection <= .10` 的点中选择 F2 最高者。若开发集没有满足约束的点，选择 over-correction 最低的点并在报告中返回 `constraint_satisfied=False`。默认值明确命名为 `BOOTSTRAP_THRESHOLDS`，不得作为实验学习结果输出。

- [ ] **Step 5：运行标定测试**

Run: `uv run pytest tests/test_calibration.py -q`

Expected: PASS，包括相同分数下因差距不足而 `DEFER` 的测试。

- [ ] **Step 6：提交标定策略**

```bash
git add backend/asr_agent/calibration.py tests/test_calibration.py
git commit -m "feat: add calibrated high-recall revision policy"
```

### Task 7：实现针对性证据解析与音频门控

**Files:**
- Create: `backend/asr_agent/resolver.py`
- Create: `tests/test_evidence_resolver.py`
- Modify: `backend/asr_agent/integrations/audio_verifier.py`

- [ ] **Step 1：编写历史修订、共存和音频失败测试**

```python
import pytest

from asr_agent.calibration import DecisionPolicy, PolicyThresholds
from asr_agent.context_judge import FocusProposal
from asr_agent.models import Session, Turn
from asr_agent.resolver import EvidenceResolver


class FixedCalibrator:
    def __init__(self, probability: float) -> None:
        self.probability = probability

    def predict(self, _features: object) -> float:
        return self.probability


def high_recall_policy() -> DecisionPolicy:
    return DecisionPolicy(PolicyThresholds(.35, .45, .62, .88, .08))


def conflict_focus(turn_id: str, before: str, after: str) -> FocusProposal:
    return FocusProposal(turn_id, before, after, [before, after])


@pytest.fixture
def session_with_audio() -> Session:
    return Session(session_id="s", turns=[
        Turn("t1", "采用泰信方案", "采用泰信方案", meta={
            "audio_path": "/tmp/test.wav", "start_sec": 0.0, "end_sec": 1.0,
        }),
        Turn("t2", "泰康方案已经确认", "泰康方案已经确认"),
    ])


def test_resolver_revises_only_targeted_historical_span(session_with_audio):
    resolver = EvidenceResolver(
        audio_verifier=lambda **_: {"ok": True, "scores": {"泰信": .18, "泰康": .82}},
        calibrator=FixedCalibrator(.76),
        policy=high_recall_policy(),
    )
    result = resolver.resolve(session_with_audio, conflict_focus("t1", "泰信", "泰康"), source_turn_id="t2")
    assert result.action == "REVISE_HISTORY"
    assert result.span == "泰信"
    assert result.replacement == "泰康"


def test_resolver_defers_when_audio_verifier_fails(session_with_audio):
    resolver = EvidenceResolver(
        audio_verifier=lambda **_: {"ok": False, "error": "timeout"},
        calibrator=FixedCalibrator(.9),
        policy=high_recall_policy(),
    )
    assert resolver.resolve(session_with_audio, conflict_focus("t1", "泰信", "泰康"), source_turn_id="t2").action == "DEFER"
```

- [ ] **Step 2：运行测试并确认失败**

Run: `uv run pytest tests/test_evidence_resolver.py -q`

Expected: FAIL，错误包含 `No module named 'asr_agent.resolver'`。

- [ ] **Step 3：实现解析结果和闭集验证**

```python
@dataclass
class Resolution:
    action: str
    target_turn_id: str
    source_turn_id: str
    span: str
    replacement: str
    probability: float
    alternative_probability: float
    evidence: list[EvidenceRef]
    rationale: str


class EvidenceResolver:
    def resolve(self, session: Session, focus: FocusProposal, *, source_turn_id: str) -> Resolution:
        target = next(turn for turn in session.turns if turn.turn_id == focus.target_turn_id)
        if focus.span not in target.raw_text or focus.proposed in {"", focus.span}:
            return self.defer(focus, source_turn_id, "invalid_focus")
        window = self._audio_window(target)
        if window is None:
            return self.defer(focus, source_turn_id, "missing_audio_window")
        audio = self.audio_verifier(
            audio_path=window[0], start_sec=window[1], end_sec=window[2],
            candidates=list(dict.fromkeys(focus.alternatives)),
        )
        if not audio.get("ok"):
            return self.defer(focus, source_turn_id, "audio_verifier_failed")
        scores = {str(key): float(value) for key, value in audio.get("scores", {}).items()}
        best = max(scores, key=scores.get)
        if best != focus.proposed:
            return self.defer(focus, source_turn_id, "audio_did_not_support_proposal")
        probability = self.calibrator.predict(self._features(focus, scores))
        runner_up = max((value for key, value in scores.items() if key != best), default=0.0)
        action = self.policy.decide(target.turn_id == source_turn_id, probability, runner_up, audio_verified=True)
        return self._resolution(action, target, focus, source_turn_id, probability, runner_up, audio)
```

保留现有开放复听恢复能力，但把退化转写检测与 `audio_retranscriber` 调用封装为 Resolver 的独立 `resolve_open_relisten` 工具；它不参与实体候选提名。

- [ ] **Step 4：运行解析器和既有音频测试**

Run: `uv run pytest tests/test_evidence_resolver.py tests/test_audio_verifier.py tests/test_autonomous_reflection.py -q`

Expected: 新解析器测试 PASS；旧测试中依赖“音频失败仍凭文本提交”的用例按新规格改为 `DEFER` 后 PASS。

- [ ] **Step 5：提交证据解析器**

```bash
git add backend/asr_agent/resolver.py backend/asr_agent/integrations/audio_verifier.py tests/test_evidence_resolver.py tests/test_autonomous_reflection.py
git commit -m "feat: add targeted evidence resolver"
```

### Task 8：切换 `ReTraceService` 到 Agent-first 编排

**Files:**
- Modify: `backend/asr_agent/retrace.py`
- Create: `tests/test_agent_first_service.py`
- Modify: `tests/test_retrace_service.py`

- [ ] **Step 1：编写正常句不扫描、后文反修历史和自动回滚测试**

```python
from unittest.mock import Mock

from asr_agent.context_judge import ContextJudgment, FocusProposal
from asr_agent.resolver import Resolution
from asr_agent.retrace import ReTraceService


class ScriptedCallable:
    def __init__(self, values: list[object]) -> None:
        self.values = iter(values)

    def __call__(self, **_kwargs: object) -> object:
        return next(self.values)


def conflict(turn_id: str, before: str, after: str, confidence: float) -> ContextJudgment:
    return ContextJudgment(
        "CONFLICT", confidence, "later contextual conflict",
        [FocusProposal(turn_id, before, after, [before, after])],
    )


def resolution(action: str, turn_id: str, before: str, after: str, probability: float) -> Resolution:
    return Resolution(action, turn_id, "", before, after, probability, 1.0 - probability, [], "scripted")


def build_service(tmp_path, *, context_judge, resolver) -> ReTraceService:
    return ReTraceService(tmp_path, context_judge=context_judge, resolver=resolver)


def test_consistent_turn_does_not_call_candidate_tools(tmp_path):
    judge = lambda **_: ContextJudgment("CONSISTENT", .8, "coherent")
    resolver = Mock()
    resolver.resolve.side_effect = AssertionError("resolver must not run")
    service = build_service(tmp_path, context_judge=judge, resolver=resolver)
    result = service.process_turn("s", "t1", "这个方案今天可以确认")
    assert result["session"]["turns"][0]["hypotheses"] == []
    resolver.resolve.assert_not_called()


def test_later_turn_revises_history_and_third_turn_rolls_it_back(tmp_path):
    judge = ScriptedCallable([
        ContextJudgment("NOVEL", .7, "new fact"),
        conflict("t1", "泰信", "泰康", .84),
        conflict("t1", "泰康", "泰信", .88),
    ])
    resolver = Mock()
    resolver.resolve.side_effect = [
        resolution("REVISE_HISTORY", "t1", "泰信", "泰康", .76),
        resolution("ROLLBACK", "t1", "泰康", "泰信", .81),
    ]
    service = build_service(tmp_path, context_judge=judge, resolver=resolver)
    service.process_turn("s", "t1", "采用泰信方案")
    service.process_turn("s", "t2", "泰康方案已经确认")
    result = service.process_turn("s", "t3", "泰信和泰康是两家不同公司")
    assert result["session"]["turns"][0]["raw_text"] == "采用泰信方案"
    assert result["session"]["turns"][0]["current_text"] == "采用泰信方案"
    assert [event["action"] for event in result["session"]["revision_events"]] == ["REVISE_HISTORY", "ROLLBACK"]
```

- [ ] **Step 2：运行测试并确认当前规则流程失败**

Run: `uv run pytest tests/test_agent_first_service.py -q`

Expected: FAIL，因为 `ReTraceService` 还没有 `context_judge`、`resolver` 和版本化分析流程。

- [ ] **Step 3：实现观察与分析两个阶段**

```python
def observe_turn(self, session_id: str, turn_id: str, text: str, *, memory_scope: str = "default",
                 confidence: dict[str, float] | None = None,
                 text_candidates: dict[str, list[str]] | None = None,
                 entity_candidate_ids: dict[str, list[str]] | None = None,
                 nbest: list[str] | None = None, risk: str = "medium",
                 source: str = "text", meta: dict[str, Any] | None = None) -> dict[str, Any]:
    def append_observation(session: Session) -> None:
        if any(turn.turn_id == turn_id for turn in session.turns):
            raise ValueError(f"duplicate turn_id: {turn_id}")
        session.memory_scope = memory_scope
        session.turns.append(Turn(turn_id, text, text, source=source, meta={
            **(meta or {}),
            "asr_signals": {
                "confidence": confidence or {},
                "text_candidates": text_candidates or {},
                "entity_candidate_ids": entity_candidate_ids or {},
                "nbest": nbest or [],
                "risk": risk,
            },
        }))
        session.analysis_status = "queued"

    session = self.repository.update(session_id, append_observation)
    return {"session": session.as_dict(), "revisions": [], "analysis_status": "queued"}


def analyze_turn(self, session_id: str, turn_id: str, observed_version: int | None = None) -> dict[str, Any]:
    for _attempt in range(3):
        session = self.repository.load(session_id)
        if observed_version is not None and session.version < observed_version:
            raise VersionConflict(f"observation version {observed_version} is not persisted")
        expected_version = session.version
        trigger = next(turn for turn in session.turns if turn.turn_id == turn_id)
        memory = self.memory_retriever.retrieve(session, query=trigger.raw_text)
        judgment = self.context_judge(current_turn=trigger, memory=memory)
        events = self._apply_judgment(session, trigger, judgment, expected_version=expected_version)
        session.analysis_status = "idle" if not session.open_hypotheses else "deferred"
        try:
            committed = self.repository.commit(session, expected_version=expected_version)
        except VersionConflict:
            continue  # 新 Turn 已到达，必须基于最新 Memory 重新判断。
        return {
            "session": committed.as_dict(),
            "revisions": [asdict(item) for item in events],
            "analysis_status": committed.analysis_status,
        }
    queued = self.repository.update(session_id, lambda current: setattr(current, "analysis_status", "queued"))
    return {"session": queued.as_dict(), "revisions": [], "analysis_status": "queued"}


def process_turn(self, session_id: str, turn_id: str, text: str, *,
                 confidence: dict[str, float] | None = None,
                 text_candidates: dict[str, list[str]] | None = None,
                 entity_candidate_ids: dict[str, list[str]] | None = None,
                 use_llm: bool = False, source: str = "text",
                 meta: dict[str, Any] | None = None, risk: str = "medium",
                 nbest: list[str] | None = None, memory_scope: str = "default") -> dict[str, Any]:
    del use_llm  # Agent 是否可用由注入的 Context Judge 决定，保留参数只为 API 兼容。
    self.observe_turn(
        session_id, turn_id, text, memory_scope=memory_scope, confidence=confidence,
        text_candidates=text_candidates, entity_candidate_ids=entity_candidate_ids,
        nbest=nbest, risk=risk, source=source, meta=meta,
    )
    return self.analyze_turn(session_id, turn_id)
```

`_apply_judgment` 对 `CONSISTENT` 和 `NOVEL` 不调用 Resolver；对 `CONFLICT`/`UNCERTAIN` 创建或更新 `WorkingHypothesis`，逐个解析 focus，并通过 `RevisionLedger.append_many(..., expected_version=expected_version)` 在同一状态版本中提交事件。每个新 Turn 还会重评 `dependency_index` 指向的开放假设。

- [ ] **Step 4：运行服务测试**

Run: `uv run pytest tests/test_agent_first_service.py tests/test_retrace_service.py -q`

Expected: PASS；正常中文句不会生成二字词假设，后文可以修订并自动回滚历史 Turn。

- [ ] **Step 5：提交 Agent-first 服务编排**

```bash
git add backend/asr_agent/retrace.py tests/test_agent_first_service.py tests/test_retrace_service.py
git commit -m "refactor: make context agent own revision nomination"
```

### Task 9：实现实时后台协调器与 FastAPI 接口

**Files:**
- Create: `backend/asr_agent/realtime.py`
- Modify: `backend/asr_agent/server.py`
- Create: `tests/test_realtime_coordinator.py`
- Modify: `tests/test_retrace_api.py`

- [ ] **Step 1：编写快速返回、同会话串行和过期结果重评测试**

```python
class RecordingService:
    def __init__(self, _root) -> None:
        self.calls: list[tuple[str, str, int | None]] = []

    def analyze_turn(self, session_id: str, turn_id: str, observed_version: int | None = None) -> dict[str, object]:
        self.calls.append((session_id, turn_id, observed_version))
        return {"analysis_status": "idle"}


class RecordingCoordinator:
    def __init__(self) -> None:
        self.submitted: list[tuple[str, str, int]] = []

    def submit(self, session_id: str, turn_id: str, observed_version: int) -> None:
        self.submitted.append((session_id, turn_id, observed_version))

    def close(self) -> None:
        return None


def test_coordinator_serializes_same_session_and_drains(tmp_path):
    service = RecordingService(tmp_path)
    coordinator = RealtimeCoordinator(service, max_workers=2)
    coordinator.submit("s", "t1", 1)
    coordinator.submit("s", "t2", 2)
    coordinator.drain("s", timeout=2)
    assert service.calls == [("s", "t1", 1), ("s", "t2", 2)]
    coordinator.close()


def test_turn_api_returns_observation_before_slow_analysis(tmp_path):
    coordinator = RecordingCoordinator()
    app = create_app(tmp_path, coordinator=coordinator)
    with TestClient(app) as client:
        response = client.post("/api/sessions/s/turns", json={"turn_id": "t1", "text": "采用泰信方案"})
        assert response.status_code == 202
        assert response.json()["analysis_status"] == "queued"
        assert response.json()["session"]["turns"][0]["raw_text"] == "采用泰信方案"
        assert coordinator.submitted == [("s", "t1", 1)]
```

- [ ] **Step 2：运行测试并确认失败**

Run: `uv run pytest tests/test_realtime_coordinator.py tests/test_retrace_api.py -q`

Expected: FAIL，因为尚无协调器，接口仍同步调用 `process_turn` 并返回 200。

- [ ] **Step 3：实现按会话串行协调器**

```python
class RealtimeCoordinator:
    def __init__(self, service: ReTraceService, max_workers: int = 4) -> None:
        self.service = service
        self.pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="retrace")
        self.tails: dict[str, Future[Any]] = {}
        self.lock = Lock()

    def submit(self, session_id: str, turn_id: str, observed_version: int) -> Future[Any]:
        with self.lock:
            previous = self.tails.get(session_id)

            def run() -> dict[str, Any]:
                if previous is not None:
                    previous.result()
                return self.service.analyze_turn(session_id, turn_id, observed_version)

            future = self.pool.submit(run)
            self.tails[session_id] = future
            return future

    def drain(self, session_id: str, timeout: float | None = None) -> None:
        future = self.tails.get(session_id)
        if future is not None:
            future.result(timeout=timeout)

    def close(self) -> None:
        self.pool.shutdown(wait=True, cancel_futures=False)
```

同一会话的任务严格按 Turn 顺序执行，不同会话可并行。`analyze_turn` 每次重新加载最新会话，因此不会提交基于旧快照的结果。

- [ ] **Step 4：修改 FastAPI 生命周期与 Turn 接口**

`create_app` 增加可注入的 `context_judge`、`resolver` 和 `coordinator`；应用关闭时调用 `coordinator.close()`。`POST /api/sessions/{id}/turns` 调用 `observe_turn`，再把响应中的 `session.version` 传给 `coordinator.submit(session_id, turn_id, observed_version)`，返回 HTTP 202。`GET /api/sessions/{id}` 返回 `analysis_status`。音频批处理 `_build_session_from_asr` 继续使用同步 `process_turn`，保证上传完成后返回完整会话。

- [ ] **Step 5：运行实时与 API 测试**

Run: `uv run pytest tests/test_realtime_coordinator.py tests/test_retrace_api.py -q`

Expected: PASS，并且测试退出时没有仍在运行的线程。

- [ ] **Step 6：提交实时调度**

```bash
git add backend/asr_agent/realtime.py backend/asr_agent/server.py tests/test_realtime_coordinator.py tests/test_retrace_api.py
git commit -m "feat: add realtime asynchronous revision pipeline"
```

### Task 10：移除主流程 N-gram 规则并更新前端状态

**Files:**
- Modify: `backend/asr_agent/retrace.py`
- Modify: `backend/asr_agent/uncertainty.py`
- Modify: `frontend/src/main.ts`
- Modify: `tests/test_autonomous_reflection.py`
- Modify: `tests/test_retrace_frontend.py`
- Create: `tests/test_no_global_ngram_nomination.py`

- [ ] **Step 1：编写规则退出主流程的结构测试**

```python
from pathlib import Path


def test_retrace_core_has_no_global_ngram_nomination():
    source = Path("backend/asr_agent/retrace.py").read_text(encoding="utf-8")
    assert "_COMMON_BIGRAMS" not in source
    assert "_GENERIC_ENTITY_BLOCK" not in source
    assert "_sliding_cjk_grams" not in source


def test_uncertainty_uses_only_explicit_asr_signals():
    source = Path("backend/asr_agent/uncertainty.py").read_text(encoding="utf-8")
    assert "memory_values" not in source
    assert "difflib.get_close_matches(value, [text]" not in source
```

- [ ] **Step 2：运行测试并确认失败**

Run: `uv run pytest tests/test_no_global_ngram_nomination.py -q`

Expected: FAIL，因为旧常量和滑窗函数仍存在。

- [ ] **Step 3：删除旧候选提名与决定路径**

从 `retrace.py` 删除 `_COMMON_BIGRAMS`、`_GENERIC_ENTITY_BLOCK`、`_BRAND_SURFACE_ALIASES`、`_content_tokens`、`_entity_tokens`、`_similar_conflict`、`_quality_tokens`、`_sliding_cjk_grams`、`_deterministic_proposals`、`_sparse_llm_triggers` 和旧 `_reflect` 调用链。`uncertainty.py` 仅保留显式 confidence 与 N-best 差异信号，作为 Agent 输入特征，不直接创建修订假设。

把仍需保留的音频窗口、退化转写开放复听和证据引用校验移动到 `resolver.py`，确保 `retrace.py` 只负责服务编排。

- [ ] **Step 4：更新前端类型与实时状态展示**

```typescript
type WorkingHypothesis = {
  hypothesis_id: string;
  target_turn_ids: string[];
  current_interpretation: string;
  proposed_interpretation: string;
  alternatives: string[];
  score: number;
  status: 'active' | 'resolved' | 'superseded';
};

type RevisionEvent = {
  event_id: string;
  action: string;
  target_turn_id: string;
  source_turn_id: string;
  before_text: string;
  after_text: string;
  evidence: string[];
  score: number;
  active: boolean;
  resolver?: string;
  rationale?: string;
  span?: string;
  replacement?: string;
  reverted_event_id?: string;
  supersedes_event_id?: string;
  reason?: string;
};

type Session = {
  session_id: string;
  version: number;
  analysis_status: 'idle' | 'queued' | 'analyzing' | 'deferred';
  turns: Turn[];
  open_hypotheses: Record<string, WorkingHypothesis>;
  revision_events: RevisionEvent[];
  quarantine_memory: Record<string, unknown>;
};

function activeEvents(current: Session | null): RevisionEvent[] {
  const events = current?.revision_events ?? [];
  const superseded = new Set(
    events.map((event) => event.supersedes_event_id).filter((id): id is string => Boolean(id)),
  );
  return events.filter((event) => event.active && !superseded.has(event.event_id));
}
```

将方法轨迹文案改为“观察 → Memory 检索 → 上下文判断 → 针对性复听 → 事件回放”；显示 `queued/analyzing/deferred` 状态，并把 `ROLLBACK` 事件标为自动回滚。保留现有字幕、证据和差异展示，不进行整体 UI 重设计。

- [ ] **Step 5：运行后端回归与前端构建**

Run: `uv run pytest tests/test_no_global_ngram_nomination.py tests/test_autonomous_reflection.py tests/test_retrace_frontend.py -q`

Expected: PASS。

Run: `cd frontend && npm run build`

Expected: TypeScript 检查与 Vite 构建成功，退出码为 0。

- [ ] **Step 6：提交旧规则移除和前端兼容**

```bash
git add backend/asr_agent/retrace.py backend/asr_agent/uncertainty.py backend/asr_agent/resolver.py frontend/src/main.ts tests/test_no_global_ngram_nomination.py tests/test_autonomous_reflection.py tests/test_retrace_frontend.py
git commit -m "refactor: retire global ngram revision rules"
```

### Task 11：完成长期 Memory 自动巩固与跨会话回溯

**Files:**
- Modify: `backend/asr_agent/memory.py`
- Modify: `backend/asr_agent/retrace.py`
- Create: `tests/test_long_term_memory_flow.py`

- [ ] **Step 1：编写完全自动的跨会话巩固测试**

```python
from asr_agent.context_judge import ContextJudgment, FocusProposal
from asr_agent.retrace import ReTraceService


class CrossSessionScriptedJudge:
    def __call__(self, *, current_turn, memory):
        if current_turn.raw_text == "泰康方案确认":
            return ContextJudgment(
                "CONFLICT", .9, "later stable naming",
                [FocusProposal("t1", "泰信", "泰康", ["泰信", "泰康"])],
            )
        if current_turn.raw_text == "继续讨论泰康保险":
            return ContextJudgment("CONSISTENT", .85, "matches long-term memory")
        return ContextJudgment("NOVEL", .7, "new session fact")


def memory_enabled_service(tmp_path):
    judge = CrossSessionScriptedJudge()
    verifier = lambda **kwargs: {
        "ok": True,
        "scores": {candidate: (.85 if candidate == "泰康" else .15) for candidate in kwargs["candidates"]},
    }
    return ReTraceService(tmp_path, context_judge=judge, audio_verifier=verifier)


def test_verified_revision_promotes_memory_and_supports_next_session(tmp_path):
    service = memory_enabled_service(tmp_path)
    service.process_turn("s1", "t1", "采用泰信方案", memory_scope="acme")
    service.process_turn("s1", "t2", "泰康方案确认", memory_scope="acme")
    service.process_turn("s2", "t1", "继续讨论泰康保险", memory_scope="acme")
    result = service.get_session("s2")

    stored = service.long_term_repository.load("acme")
    assert any(item.value == "泰康" and item.status == "stable" for item in stored)
    assert result["memory_scope"] == "acme"


def test_single_unverified_asr_cannot_poison_long_term_memory(tmp_path):
    service = memory_enabled_service(tmp_path)
    service.process_turn("s1", "t1", "合作方是泰信", memory_scope="acme")
    assert service.long_term_repository.load("acme") == []
```

- [ ] **Step 2：运行测试并确认失败**

Run: `uv run pytest tests/test_long_term_memory_flow.py -q`

Expected: FAIL，因为 Service 尚未在事件提交后调用巩固器。

- [ ] **Step 3：在事件提交后自动更新 Memory**

`REVISE_CURRENT`/`REVISE_HISTORY` 成功后，从结构化假设生成 `MemoryBelief`，将目标 Turn、证据 Turn、会话 ID 和音频验证记录为来源。达到 `T_long_memory` 时调用 `MemoryConsolidator.promote`；未达到时只写入 `session.working_beliefs`。`ROLLBACK` 会把由被回滚事件产生的长期信念标记为 `superseded`，并重新评估依赖该信念的开放假设。

- [ ] **Step 4：运行跨会话与 Memory 测试**

Run: `uv run pytest tests/test_long_term_memory_flow.py tests/test_memory.py tests/test_agent_first_service.py -q`

Expected: PASS；整个流程没有人工确认入口。

- [ ] **Step 5：提交自动巩固**

```bash
git add backend/asr_agent/memory.py backend/asr_agent/retrace.py tests/test_long_term_memory_flow.py
git commit -m "feat: consolidate verified beliefs across sessions"
```

### Task 12：扩展评测指标与连续会话端到端测试

**Files:**
- Modify: `backend/asr_agent/metrics.py`
- Modify: `tests/test_retrace_metrics.py`
- Create: `tests/fixtures/realtime_memory_cases.json`
- Create: `tests/test_realtime_memory_e2e.py`
- Modify: `README.md`

- [ ] **Step 1：编写高召回指标测试**

```python
def test_metrics_report_recall_f2_rollback_and_memory_contamination():
    report = evaluate_revisions(
        events=[
            {"event_id": "e1", "action": "REVISE_HISTORY", "target_turn_id": "t1", "source_turn_id": "t3", "after_text": "泰康方案", "active": True},
            {"event_id": "e2", "action": "ROLLBACK", "target_turn_id": "t2", "source_turn_id": "t4", "supersedes_event_id": "bad", "active": True},
        ],
        reference_by_turn={"t1": "泰康方案", "t2": "原文"},
        turn_order=["t1", "t2", "t3", "t4"],
        ambiguous_turn_ids={"t1", "t2"},
        promoted_beliefs=[{"value": "泰康", "correct": True}, {"value": "泰信", "correct": False}],
    )
    assert report["revision_recall"] == .5
    assert report["rollback_success_rate"] == 1.0
    assert report["memory_contamination_rate"] == .5
    assert report["revision_f2"] > 0
```

- [ ] **Step 2：运行指标测试并确认签名不兼容**

Run: `uv run pytest tests/test_retrace_metrics.py -q`

Expected: FAIL，因为现有 `evaluate_revisions` 没有 `ambiguous_turn_ids`、`promoted_beliefs`、F2/F3 和回滚指标。

- [ ] **Step 3：扩展指标实现并保留旧调用兼容**

新增 `ambiguous_turn_ids: set[str] | None = None`、`promoted_beliefs: list[dict[str, Any]] | None = None`，继续接受旧 `ambiguous_turn_count`。先从 `supersedes_event_id` 计算失效事件集合；有效修订动作统一包含 `REVISE_CURRENT`、`REVISE_HISTORY`、旧版 `REVISE_TEXT` 和 `REVISE_ENTITY`。随后按目标 Turn 去重计算正确修订、漏修、precision、recall、F2/F3；按 `ROLLBACK.supersedes_event_id` 计算自动回滚；按 `correct=False` 的已提升信念占比计算长期 Memory 污染率。

- [ ] **Step 4：添加连续会话 fixture 与端到端测试**

`realtime_memory_cases.json` 至少包含四个完整序列：后文修订历史、相似名称共存、现实事实变化、误修后自动回滚。每个序列明确给出 Turn 顺序、Agent 判断、音频闭集分数、参考字幕和预期事件。测试使用注入式 Judge 与 Verifier 重放 fixture，不访问网络，并断言最终字幕、事件顺序、Memory 状态和指标。

- [ ] **Step 5：更新 README**

把旧的“先标记可疑片段再反思”改为 Agent-first 流程，说明：短期/长期 Memory、实时快慢链路、自动回溯与回滚、音频门控、202 实时接口和轮询 `GET /api/sessions/{id}`。明确 `_COMMON_BIGRAMS` 不再处于主决策链。

- [ ] **Step 6：运行端到端测试和指标测试**

Run: `uv run pytest tests/test_retrace_metrics.py tests/test_realtime_memory_e2e.py -q`

Expected: PASS，四类连续会话均得到预期事件与最终字幕。

- [ ] **Step 7：提交评测和文档**

```bash
git add backend/asr_agent/metrics.py tests/test_retrace_metrics.py tests/fixtures/realtime_memory_cases.json tests/test_realtime_memory_e2e.py README.md
git commit -m "test: evaluate realtime memory revision flows"
```

### Task 13：全量验证与迁移检查

**Files:**
- Modify only if verification exposes an in-scope defect.

- [ ] **Step 1：运行 Python 全量测试**

Run: `uv run pytest -q`

Expected: 所有测试 PASS；不得通过删除旧行为测试来规避兼容问题，只有与已确认新规格直接冲突的断言可以改写。

- [ ] **Step 2：运行前端生产构建**

Run: `cd frontend && npm run build`

Expected: `tsc --noEmit` 和 `vite build` 均成功，退出码为 0。

- [ ] **Step 3：检查规则、占位符和格式**

Run: `rg -n "_COMMON_BIGRAMS|_GENERIC_ENTITY_BLOCK|_sliding_cjk_grams|TBD|TODO|FIXME" backend tests README.md`

Expected: 主流程中没有旧 N-gram 常量和方法，也没有实现占位符；测试 fixture 中若以字符串提及旧名称，仅用于结构断言。

Run: `git diff --check`

Expected: 无输出，退出码为 0。

- [ ] **Step 4：执行无网络实时冒烟测试**

Run: `uv run pytest tests/test_realtime_memory_e2e.py::test_later_evidence_revises_and_rolls_back_history -q`

Expected: PASS，并且原始 Turn、`REVISE_HISTORY`、`ROLLBACK` 和最终字幕全部符合 fixture。

- [ ] **Step 5：记录最终验证结果并提交必要的收尾修改**

如果前四步无需修改，不创建空提交；如果发现并修复了范围内问题，则运行受影响测试后提交：

```bash
git add backend/asr_agent frontend/src tests README.md
git commit -m "fix: complete realtime memory revision migration"
```
