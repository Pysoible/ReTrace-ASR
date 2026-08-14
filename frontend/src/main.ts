import './styles.css';

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
  session_version?: number;
  reason?: string;
};

type MemoryBelief = {
  belief_id: string;
  subject: string;
  predicate: string;
  value: string;
  aliases: string[];
  confidence: number;
  status: string;
  valid_from?: string;
  valid_to?: string;
  source_turn_ids: string[];
  source_session_ids: string[];
  evidence_kinds: string[];
  created_version: number;
  updated_version: number;
  supersedes?: string;
};

type WorkingHypothesis = {
  hypothesis_id: string;
  target_turn_ids: string[];
  current_interpretation: string;
  proposed_interpretation: string;
  alternatives: string[];
  relationship: 'MUTUALLY_EXCLUSIVE' | 'COEXIST' | 'TEMPORAL_CHANGE';
  score: number;
  status: string;
  created_version: number;
  last_evaluated_version: number;
};

type Turn = {
  turn_id: string;
  raw_text: string;
  current_text: string;
  source?: string;
  meta?: {
    start_sec?: number;
    end_sec?: number;
    context_judgment?: { outcome?: string; confidence?: number; rationale?: string };
  };
};

type LatestAnalysis = {
  turn_id: string;
  outcome: string;
  confidence: number;
  rationale: string;
  observed_version?: number;
  analyzed_version?: number;
  revalidated: boolean;
};

type Observability = {
  latest_analysis: LatestAnalysis | null;
  short_term: {
    recent_turns: Turn[];
    dependent_turns: Turn[];
    working_beliefs: MemoryBelief[];
    open_hypotheses: WorkingHypothesis[];
  };
  long_term: { beliefs: MemoryBelief[] };
};

type Session = {
  session_id: string;
  version: number;
  analysis_status: string;
  turns: Turn[];
  revision_events: RevisionEvent[];
  working_beliefs: Record<string, MemoryBelief>;
  open_hypotheses: Record<string, WorkingHypothesis>;
  observability?: Observability;
};

const root = document.querySelector<HTMLDivElement>('#app')!;
let session: Session | null = null;
let selectedTurnId = '';
let selectedEventId = '';
let memoryView: 'short' | 'long' = 'short';
let status = '等待新的 ASR observation。';
let busy = false;
let liveEventSource: EventSource | null = null;
let liveStreamSession = '';
// Panel collapse state: keyed by panel name, expanded by default.
let collapsedPanels: Record<string, boolean> = {};

const decisionActions = ['KEEP_OLD', 'ACCEPT_NEW', 'COEXIST', 'DEFER', 'REVISE_CURRENT', 'REVISE_HISTORY', 'ROLLBACK'];
const relationshipLabels: Record<string, string> = {
  MUTUALLY_EXCLUSIVE: '互斥解释',
  COEXIST: '信息共存',
  TEMPORAL_CHANGE: '时序变化',
};

function escapeHtml(value: unknown): string {
  return String(value ?? '').replace(/[&<>'"]/g, (char) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#039;', '"': '&quot;',
  }[char]!));
}

function stripTime(text: string): string {
  return (text || '').replace(/^\[[0-9.]+-[0-9.]+\]\s*/, '');
}

function percent(value: number | undefined): string {
  return `${Math.round((value ?? 0) * 100)}%`;
}

function allEvents(): RevisionEvent[] {
  return session?.revision_events ?? [];
}

function transcriptEvents(): RevisionEvent[] {
  const events = allEvents();
  const superseded = new Set(events.filter((event) => event.active).map((event) => event.supersedes_event_id).filter(Boolean));
  const actions = new Set(['REVISE_CURRENT', 'REVISE_HISTORY', 'REVISE_TEXT', 'REVISE_ENTITY', 'ROLLBACK']);
  return events.filter((event) => event.active && !superseded.has(event.event_id) && actions.has(event.action));
}

function eventForTurn(turnId: string): RevisionEvent | undefined {
  return transcriptEvents().find((event) => event.target_turn_id === turnId);
}

function selectedEvent(): RevisionEvent | undefined {
  if (selectedEventId) return allEvents().find((event) => event.event_id === selectedEventId);
  if (selectedTurnId) return [...allEvents()].reverse().find((event) => event.target_turn_id === selectedTurnId || event.source_turn_id === selectedTurnId);
  return allEvents().at(-1);
}

function renderTurnBody(turn: Turn, event?: RevisionEvent): string {
  const raw = stripTime(turn.raw_text);
  const current = stripTime(turn.current_text);
  if (!event || raw === current) return `<p class="turn-text">${escapeHtml(turn.current_text)}</p>`;
  const rawDisplay = raw.trim() ? escapeHtml(raw) : '<i class="empty-asr">empty ASR</i>';
  return `<div class="diff-stack">
    <div class="diff-row"><span class="diff-label">RAW</span><div><mark class="incorrect">${rawDisplay}</mark></div></div>
    <div class="diff-row current"><span class="diff-label">CURRENT</span><div><mark class="corrected">${escapeHtml(current)}</mark></div></div>
  </div>`;
}

function renderTurns(): string {
  if (!session?.turns.length) {
    return `<div class="transcript-paper empty-state"><b>等待 ASR 输入</b><p>首遍识别会作为 immutable raw observation 保留在这里。</p></div>`;
  }
  return `<div class="transcript-paper">${session.turns.map((turn, index) => {
    const event = eventForTurn(turn.turn_id);
    const selected = selectedTurnId === turn.turn_id;
    const time = turn.meta?.start_sec === undefined ? '' : `${Number(turn.meta.start_sec).toFixed(1)}-${Number(turn.meta.end_sec).toFixed(1)}s`;
    return `<article class="subtitle-line ${event ? 'revised' : ''} ${selected ? 'selected' : ''}" data-turn="${escapeHtml(turn.turn_id)}" tabindex="0">
      <div class="subtitle-meta">
        <span>TURN ${String(index + 1).padStart(2, '0')}</span>
        <span>${escapeHtml(time || turn.turn_id)}</span>
        <span class="turn-state">${event ? escapeHtml(event.action) : 'OBSERVED'}</span>
      </div>
      ${renderTurnBody(turn, event)}
      ${event ? `<button class="revision-link evidence-note" data-event="${escapeHtml(event.event_id)}"><span>Evidence</span>查看修订依据</button>` : ''}
    </article>`;
  }).join('')}</div>`;
}

function stageState(name: string): { state: string; detail: string } {
  const observation = Boolean(session?.turns.length);
  const trace = session?.observability;
  const analysis = trace?.latest_analysis;
  const events = allEvents();
  if (name === 'OBSERVE') return { state: observation ? 'done' : 'waiting', detail: observation ? 'raw 与时间戳已冻结' : '等待首个 turn' };
  if (name === 'MEMORY RETRIEVE') {
    const count = (trace?.short_term.recent_turns.length ?? 0) + (trace?.long_term.beliefs.length ?? 0);
    return { state: analysis ? 'done' : observation ? 'active' : 'waiting', detail: analysis ? `${count} 条上下文进入判断` : '等待分析' };
  }
  if (name === 'CONTEXT JUDGE') return { state: analysis ? 'done' : observation ? 'active' : 'waiting', detail: analysis ? `${analysis.outcome} · ${percent(analysis.confidence)}` : '尚无结构化判断' };
  if (name === 'TARGETED RELISTEN') {
    const audio = events.some((event) => event.resolver?.includes('audio') || event.evidence.some((item) => item.toLowerCase().includes('audio')));
    return { state: audio ? 'done' : analysis ? 'skipped' : 'waiting', detail: audio ? '冲突焦点已做音频验证' : analysis ? '本轮无需或无可用音频' : '由冲突触发' };
  }
  return { state: events.length ? 'done' : observation ? 'ready' : 'waiting', detail: events.length ? `${events.length} 个事件可重放` : '账本保持原始投影' };
}

function renderMethodTrace(): string {
  const stages = [
    ['OBSERVE', '保留不可变首遍识别'],
    ['MEMORY RETRIEVE', '检索短期与长期记忆'],
    ['CONTEXT JUDGE', '判断新旧信息关系'],
    ['TARGETED RELISTEN', '仅验证冲突音频窗口'],
    ['EVENT REPLAY', '从 raw 与账本重建字幕'],
  ];
  return `<section class="method-trace-card">
    <div class="section-label"><span>METHOD TRACE</span><small>live protocol</small></div>
    <ol class="method-trace">${stages.map(([name, fallback], index) => {
      const stage = stageState(name);
      return `<li class="stage-${stage.state}"><span class="stage-index">${index + 1}</span><div><b>${name}</b><span>${escapeHtml(stage.detail || fallback)}</span></div><i>${stage.state}</i></li>`;
    }).join('')}</ol>
  </section>`;
}

function renderAnalysis(): string {
  const analysis = session?.observability?.latest_analysis;
  if (!analysis) return `<div class="analysis-empty"><b>尚无 Agent 判断</b><p>输入到达后，这里显示 outcome、置信度与版本一致性。</p></div>`;
  return `<section class="analysis-summary">
    <div class="analysis-top"><span class="outcome outcome-${escapeHtml(analysis.outcome)}">${escapeHtml(analysis.outcome)}</span><b>${percent(analysis.confidence)}</b></div>
    <p>${escapeHtml(analysis.rationale || 'Agent 未返回补充说明。')}</p>
    <div class="version-row">
      <span>Observed v${escapeHtml(analysis.observed_version ?? '-')}</span>
      <span>Analyzed v${escapeHtml(analysis.analyzed_version ?? '-')}</span>
      <span class="${analysis.revalidated ? 'version-alert' : ''}">${analysis.revalidated ? 'revalidated on newer state' : 'version aligned'}</span>
    </div>
  </section>`;
}

function renderDecision(event: RevisionEvent | undefined): string {
  if (!event) return `<div class="decision method-rail analysis-empty"><b>账本尚为空</b><p>KEEP_OLD、ACCEPT_NEW、COEXIST、DEFER、修订与 ROLLBACK 都会自动记录。</p></div>`;
  const before = stripTime(event.before_text);
  const after = stripTime(event.after_text || event.replacement || event.before_text);
  return `<div class="decision method-rail">
    <div class="decision-heading"><span class="action action-${escapeHtml(event.action)}">${escapeHtml(event.action)}</span><b>${percent(event.score)}</b></div>
    ${before !== after ? `<div class="belief-compare"><span>${escapeHtml(before || 'empty')}</span><i>→</i><strong>${escapeHtml(after || 'empty')}</strong></div>` : ''}
    <p>${escapeHtml(event.rationale || event.reason || '该动作已写入 append-only audit。')}</p>
    <dl>
      <dt>Target / Source</dt><dd>${escapeHtml(event.target_turn_id)} ← ${escapeHtml(event.source_turn_id)}</dd>
      <dt>Resolver</dt><dd>${escapeHtml(event.resolver || 'context-judge')}</dd>
      <dt>Evidence</dt><dd>${event.evidence.length ? event.evidence.map(escapeHtml).join('<br>') : '仅上下文判断，无额外音频证据'}</dd>
      <dt>Ledger version</dt><dd>v${escapeHtml(event.session_version ?? '-')} · ${event.active ? 'active' : 'inactive'}</dd>
      ${event.supersedes_event_id ? `<dt>Supersedes</dt><dd>${escapeHtml(event.supersedes_event_id)}</dd>` : ''}
      ${event.reverted_event_id ? `<dt>Rollback target</dt><dd>${escapeHtml(event.reverted_event_id)}</dd>` : ''}
    </dl>
  </div>`;
}

function renderLedger(): string {
  const events = [...allEvents()].reverse();
  if (!events.length) return `<div class="ledger-empty">首个判断完成后，自动决策会出现在这里。</div>`;
  return `<div class="ledger-list">${events.map((event) => `<button class="ledger-event ${selectedEventId === event.event_id ? 'selected' : ''}" data-ledger-event="${escapeHtml(event.event_id)}">
    <span class="ledger-dot action-${escapeHtml(event.action)}"></span>
    <span><b>${escapeHtml(event.action)}</b><small>${escapeHtml(event.target_turn_id)} · v${escapeHtml(event.session_version ?? '-')}</small></span>
    <i>${percent(event.score)}</i>
  </button>`).join('')}</div>`;
}

function renderBelief(belief: MemoryBelief): string {
  return `<article class="belief-item">
    <div class="belief-head"><span>${escapeHtml(belief.status)}</span><b>${percent(belief.confidence)}</b></div>
    <h4>${escapeHtml(belief.subject)} <i>${escapeHtml(belief.predicate)}</i></h4>
    <p>${escapeHtml(belief.value)}</p>
    <dl>
      <dt>Source turns</dt><dd>${belief.source_turn_ids.map(escapeHtml).join(', ') || '-'}</dd>
      <dt>Evidence</dt><dd>${belief.evidence_kinds.map(escapeHtml).join(', ') || 'context'}</dd>
      <dt>Version</dt><dd>v${belief.created_version} → v${belief.updated_version}</dd>
      ${belief.supersedes ? `<dt>Supersedes</dt><dd>${escapeHtml(belief.supersedes)}</dd>` : ''}
    </dl>
  </article>`;
}

function renderHypothesis(item: WorkingHypothesis): string {
  return `<article class="candidate-card">
    <div class="belief-head"><span>${escapeHtml(item.status)}</span><b>${percent(item.score)}</b></div>
    <span class="relationship">${escapeHtml(item.relationship)} · ${escapeHtml(relationshipLabels[item.relationship] || item.relationship)}</span>
    <div class="candidate-pair"><span>${escapeHtml(item.current_interpretation)}</span><i>↔</i><strong>${escapeHtml(item.proposed_interpretation)}</strong></div>
    <div class="evidence-packet"><span>targets ${item.target_turn_ids.map(escapeHtml).join(', ')}</span><span>v${item.created_version} → v${item.last_evaluated_version}</span></div>
  </article>`;
}

function renderMemory(): string {
  const trace = session?.observability;
  const short = trace?.short_term;
  const long = trace?.long_term.beliefs ?? [];
  const tabs = `<div class="memory-tabs" role="tablist"><button class="${memoryView === 'short' ? 'active' : ''}" data-memory="short">SHORT-TERM</button><button class="${memoryView === 'long' ? 'active' : ''}" data-memory="long">LONG-TERM</button></div>`;
  if (memoryView === 'long') {
    return `${tabs}<div class="memory-content"><div class="memory-section-title"><b>STABLE BELIEFS</b><span>${long.length}</span></div>${long.length ? long.map(renderBelief).join('') : '<div class="memory-empty"><b>没有相关长期记忆</b><p>只有稳定且与当前 turn 相关的信念会进入判断。</p></div>'}</div>`;
  }
  const recent = short?.recent_turns ?? [];
  const dependent = short?.dependent_turns ?? [];
  // Fall back to the raw session map when the observability array is empty,
  // so WORKING BELIEFS is never misleadingly shown as zero.
  const beliefs = (short?.working_beliefs?.length ? short.working_beliefs : Object.values(session?.working_beliefs ?? {}));
  const hypotheses = (short?.open_hypotheses?.length ? short.open_hypotheses : Object.values(session?.open_hypotheses ?? {}));
  return `${tabs}<div class="memory-content">
    <div class="memory-section-title"><b>RECENT CONTEXT</b><span>${recent.length}</span></div>
    ${recent.length ? `<div class="context-list">${recent.map((turn) => `<button data-context-turn="${escapeHtml(turn.turn_id)}"><span>${escapeHtml(turn.turn_id)}</span><p>${escapeHtml(stripTime(turn.current_text))}</p></button>`).join('')}</div>` : '<p class="compact-empty">尚无近期 turn。</p>'}
    ${dependent.length ? `<div class="memory-section-title"><b>DEPENDENCY CONTEXT</b><span>${dependent.length}</span></div><div class="context-list">${dependent.map((turn) => `<button data-context-turn="${escapeHtml(turn.turn_id)}"><span>${escapeHtml(turn.turn_id)}</span><p>${escapeHtml(stripTime(turn.current_text))}</p></button>`).join('')}</div>` : ''}
    <div class="memory-section-title"><b>WORKING BELIEFS</b><span>${beliefs.length}</span></div>
    ${beliefs.length ? beliefs.map(renderBelief).join('') : '<p class="compact-empty">尚未形成工作信念。</p>'}
    <div class="memory-section-title"><b>COMPETING BELIEFS</b><span>${hypotheses.length}</span></div>
    ${hypotheses.length ? hypotheses.map(renderHypothesis).join('') : '<p class="compact-empty">当前没有竞争解释。</p>'}
  </div>`;
}

function render(): void {
  const turns = session?.turns.length ?? 0;
  const revisions = transcriptEvents().length;
  const activeHypotheses = Object.values(session?.open_hypotheses ?? {}).filter((item) => item.status === 'active').length;
  root.innerHTML = `<main class="agent-workbench">
    <header class="app-header">
      <div class="brand"><span class="brand-mark">R</span><div><h1>ReTrace-ASR</h1><p>Realtime temporal memory agent</p></div></div>
      <div class="runtime-metrics">
        <div><span>SESSION</span><b>${escapeHtml(session?.session_id || 'not started')}</b></div>
        <div><span>VERSION</span><b>v${session?.version ?? 0}</b></div>
        <div><span>AGENT</span><b class="agent-state state-${escapeHtml(session?.analysis_status || 'idle')}">${escapeHtml(session?.analysis_status || 'idle')}</b></div>
      </div>
    </header>

    <section class="protocol-summary">
      <div><span>Turns</span><b>${turns}</b></div><div><span>Revisions</span><b>${revisions}</b></div><div><span>Open hypotheses</span><b>${activeHypotheses}</b></div>
      <p>新信息可以支持、共存或推翻旧解释；所有判断自动完成，raw 始终不可变。</p>
    </section>

    <div class="workspace-grid">
      <section class="workspace-panel transcript-panel ${collapsedPanels.transcript ? 'collapsed' : ''}">
        <div class="panel-head">
          <div><span class="eyebrow">LIVE TRANSCRIPT</span><h2>时序字幕</h2></div>
          <span class="panel-head-actions"><span>raw → current</span><button class="panel-toggle" data-toggle="transcript" title="${collapsedPanels.transcript ? '展开' : '收起'}">${collapsedPanels.transcript ? '▸' : '▾'}</button></span>
        </div>
        <div class="panel-body">${renderTurns()}</div>
      </section>

      <section class="workspace-panel agent-panel ${collapsedPanels.agent ? 'collapsed' : ''}">
        <div class="panel-head">
          <div><span class="eyebrow">AGENT NOTE</span><h2>实时推理轨迹</h2></div>
          <span class="panel-head-actions"><span>append-only audit</span><button class="panel-toggle" data-toggle="agent" title="${collapsedPanels.agent ? '展开' : '收起'}">${collapsedPanels.agent ? '▸' : '▾'}</button></span>
        </div>
        <div class="panel-body">${renderAnalysis()}
          ${renderMethodTrace()}
          <div class="subsection-head"><b>DECISION DETAIL</b><span>${decisionActions.join(' · ')}</span></div>
          ${renderDecision(selectedEvent())}
          <div class="subsection-head"><b>EVENT LEDGER</b><span>${allEvents().length} events</span></div>
          ${renderLedger()}
        </div>
      </section>

      <aside class="workspace-panel memory-panel ${collapsedPanels.memory ? 'collapsed' : ''}">
        <div class="panel-head">
          <div><span class="eyebrow">MEMORY INSPECTOR</span><h2>双时间尺度记忆</h2></div>
          <span class="panel-head-actions"><span>read only</span><button class="panel-toggle" data-toggle="memory" title="${collapsedPanels.memory ? '展开' : '收起'}">${collapsedPanels.memory ? '▸' : '▾'}</button></span>
        </div>
        <div class="panel-body">${renderMemory()}</div>
      </aside>
    </div>

    <section class="observation">
      <div class="composer-copy"><span class="eyebrow">QWEN-OMNI AUDIO INPUT</span><h2>音频输入</h2><p>上传音频后实时转写并自动分析，每个 turn 完成后即时显示。</p></div>
      <div class="input-stack">
        <label><span>Session ID</span><input id="session" value="${escapeHtml(session?.session_id || 'demo')}" aria-label="Session ID"></label>
        <label class="file-field"><span>Audio file</span><input id="audio" type="file" accept="audio/*" aria-label="Audio file"></label>
        <button id="submit-audio" class="secondary-command" ${busy ? 'disabled' : ''}>Transcribe audio</button>
        <p class="status-line"><span class="status-dot ${busy ? 'busy' : ''}"></span>${escapeHtml(status)}</p>
      </div>
    </section>
  </main>`;

  document.querySelectorAll<HTMLElement>('[data-turn]').forEach((item) => {
    const select = () => { selectedTurnId = item.dataset.turn || ''; selectedEventId = ''; render(); };
    item.addEventListener('click', select);
    item.addEventListener('keydown', (event) => { if (event.key === 'Enter' || event.key === ' ') select(); });
  });
  document.querySelectorAll<HTMLButtonElement>('[data-event], [data-ledger-event]').forEach((button) => button.addEventListener('click', (event) => {
    event.stopPropagation();
    selectedEventId = button.dataset.event || button.dataset.ledgerEvent || '';
    const selected = selectedEvent();
    selectedTurnId = selected?.target_turn_id || selectedTurnId;
    render();
  }));
  document.querySelectorAll<HTMLButtonElement>('[data-toggle]').forEach((button) => button.addEventListener('click', (event) => {
    event.stopPropagation();
    const name = button.dataset.toggle || '';
    collapsedPanels[name] = !collapsedPanels[name];
    render();
  }));
  document.querySelectorAll<HTMLButtonElement>('[data-memory]').forEach((button) => button.addEventListener('click', () => {
    memoryView = button.dataset.memory === 'long' ? 'long' : 'short';
    render();
  }));
  document.querySelectorAll<HTMLButtonElement>('[data-context-turn]').forEach((button) => button.addEventListener('click', () => {
    selectedTurnId = button.dataset.contextTurn || '';
    selectedEventId = '';
    render();
  }));
  document.querySelector<HTMLButtonElement>('#submit-audio')?.addEventListener('click', submitAudio);
}

async function refreshSession(sessionId: string): Promise<Session> {
  const response = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}`);
  if (!response.ok) throw new Error(await response.text());
  return (await response.json()).session as Session;
}

function closeLiveStream(): void {
  liveEventSource?.close();
  liveEventSource = null;
  liveStreamSession = '';
}

async function submitAudio(): Promise<void> {
  const sessionId = document.querySelector<HTMLInputElement>('#session')!.value.trim() || 'auto';
  const file = document.querySelector<HTMLInputElement>('#audio')!.files?.[0];
  if (!file) return;
  closeLiveStream();
  busy = true;
  status = '音频上传中…';
  render();
  try {
    const form = new FormData();
    form.append('file', file);
    const response = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}/audio/upload`, { method: 'POST', body: form });
    if (!response.ok) throw new Error(await response.text());
    const body = await response.json();
    const boundSession = (body.session_id as string) || sessionId;
    liveStreamSession = boundSession;
    status = '音频已上传 · 正在实时转写与分析，每个 turn 完成后即时显示…';
    busy = false;
    render();

    const es = new EventSource(`/api/sessions/${encodeURIComponent(boundSession)}/events`);
    liveEventSource = es;
    es.onmessage = async (event) => {
      let payload: any;
      try {
        payload = JSON.parse(event.data);
      } catch {
        return;
      }
      if (payload.type === 'session') {
        status = '正在切分并转写音频…';
        render();
        return;
      }
      if (payload.type === 'asr_done') {
        status = `音频转写完成（${payload.chunk_count ?? ''} 段），正在逐句分析上下文…`;
        render();
        return;
      }
      if (payload.type === 'turn') {
        try {
          session = payload.session ?? (await refreshSession(boundSession));
        } catch {
          /* keep last session */
        }
        selectedTurnId = payload.turn_id || session?.turns.at(-1)?.turn_id || '';
        selectedEventId = allEvents().at(-1)?.event_id || '';
        status = `已处理 ${session?.turns.length ?? payload.index + 1} 个 turn · 持续分析中…`;
        render();
        return;
      }
      if (payload.type === 'done') {
        try {
          session = payload.session ?? (await refreshSession(boundSession));
        } catch {
          /* keep last session */
        }
        closeLiveStream();
        selectedTurnId = session?.turns.at(-1)?.turn_id || '';
        selectedEventId = allEvents().at(-1)?.event_id || '';
        status = `完成：${session?.turns.length ?? 0} turns · ${transcriptEvents().length} revisions。`;
        busy = false;
        render();
        return;
      }
      if (payload.type === 'error') {
        status = `处理出错：${String(payload.message ?? '未知错误')}`;
        closeLiveStream();
        busy = false;
        render();
      }
    };
    es.onerror = () => {
      // EventSource auto-reconnects; only surface a message once the stream ends.
      if (!liveStreamSession) return;
    };
  } catch (error) {
    closeLiveStream();
    status = `音频处理失败：${error instanceof Error ? error.message : String(error)}`;
    busy = false;
    render();
  }
}

render();
