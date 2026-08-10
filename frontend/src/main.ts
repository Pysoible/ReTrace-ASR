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
  reason?: string;
};
type WorkingHypothesis = {
  hypothesis_id: string;
  target_turn_ids: string[];
  current_interpretation: string;
  proposed_interpretation: string;
  alternatives: string[];
  score: number;
  status: string;
};
type Turn = {
  turn_id: string;
  raw_text: string;
  current_text: string;
  source?: string;
  meta?: { start_sec?: number; end_sec?: number; degenerate_relisten?: Record<string, unknown> };
};
type Session = {
  session_id: string;
  version: number;
  analysis_status: string;
  turns: Turn[];
  revision_events: RevisionEvent[];
  open_hypotheses: Record<string, WorkingHypothesis>;
};
type Impact = {
  revised_turns: number;
  rollbacks: number;
  deferred: number;
  recovered_chars: number;
  highlight?: string;
};

const root = document.querySelector<HTMLDivElement>('#app')!;
let session: Session | null = null;
let selected: RevisionEvent | null = null;
let status = '等待新的 ASR observation。';
let busy = false;
let impactNote = '';

function escapeHtml(value: string): string {
  return value.replace(/[&<>'"]/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#039;', '"': '&quot;' }[char]!));
}

function stripTime(text: string): string {
  return (text || '').replace(/^\[[0-9.]+-[0-9.]+\]\s*/, '');
}

function activeEvents(): RevisionEvent[] {
  const events = session?.revision_events ?? [];
  const superseded = new Set(events.filter((event) => event.active).map((event) => event.supersedes_event_id).filter(Boolean));
  const transcriptActions = new Set(['REVISE_CURRENT', 'REVISE_HISTORY', 'REVISE_TEXT', 'REVISE_ENTITY', 'ROLLBACK']);
  return events.filter((event) => event.active && !superseded.has(event.event_id) && transcriptActions.has(event.action));
}

function activeEvent(turnId: string): RevisionEvent | undefined {
  return activeEvents().find((event) => event.target_turn_id === turnId);
}

function computeImpact(current: Session | null): Impact {
  const events = (current?.revision_events ?? []).filter((event) => event.active);
  let recovered = 0;
  for (const event of events) {
    const before = stripTime(event.before_text || '');
    const after = stripTime(event.after_text || event.replacement || '');
    recovered += Math.max(0, after.length - before.length);
  }
  const rollbacks = events.filter((event) => event.action === 'ROLLBACK').length;
  const deferred = Object.values(current?.open_hypotheses ?? {}).filter((item) => item.status === 'active').length;
  const joined = (current?.turns ?? []).map((turn) => stripTime(turn.current_text)).join('');
  const highlight = joined.includes('骁龙') ? '骁龙 recovered' : joined.includes('麒麟') ? '麒麟 recovered' : undefined;
  return {
    revised_turns: new Set(events.map((event) => event.target_turn_id)).size,
    rollbacks,
    deferred,
    recovered_chars: recovered,
    highlight,
  };
}

function resolverLabel(resolver?: string): string {
  if (!resolver) return 'REVISION';
  if (resolver.includes('rollback')) return 'ROLLBACK';
  if (resolver.includes('agent-context-audio')) return 'CONTEXT + AUDIO';
  return resolver.toUpperCase();
}

function renderTurnBody(turn: Turn, event?: RevisionEvent): string {
  const raw = stripTime(turn.raw_text);
  const cur = stripTime(turn.current_text);
  if (!event) return `<p class="turn-text">${escapeHtml(turn.current_text)}</p>`;

  const isOpen = raw !== cur;
  if (isOpen) {
    const rawShow = raw.trim() ? escapeHtml(raw) : '<i class="empty-asr">∅ empty / collapsed ASR</i>';
    return `<div class="diff-stack">
      <div class="diff-row raw"><span class="diff-label">ASR</span><div class="diff-body"><mark class="incorrect">${rawShow}</mark></div></div>
      <div class="diff-row cur reveal"><span class="diff-label">RETRACE</span><div class="diff-body"><mark class="corrected">${escapeHtml(cur)}</mark></div></div>
    </div>`;
  }

  const span = event.span || '';
  const replacement = event.replacement || '';
  if (span && replacement && cur.includes(replacement)) {
    const highlighted = escapeHtml(cur).replace(
      escapeHtml(replacement),
      `<mark class="corrected">${escapeHtml(replacement)}</mark>`,
    );
    return `<p class="turn-text">${highlighted}</p>
      <p class="inline-fix"><mark class="incorrect">${escapeHtml(span)}</mark> → <mark class="corrected">${escapeHtml(replacement)}</mark></p>`;
  }
  return `<div class="diff-stack">
    <div class="diff-row raw"><span class="diff-label">BEFORE</span><div class="diff-body"><mark class="incorrect">${escapeHtml(stripTime(event.before_text))}</mark></div></div>
    <div class="diff-row cur reveal"><span class="diff-label">AFTER</span><div class="diff-body"><mark class="corrected">${escapeHtml(stripTime(event.after_text))}</mark></div></div>
  </div>`;
}

function renderTurns(): string {
  const current = session;
  if (!current?.turns.length) {
    return `<div class="transcript-paper">
      <p class="empty">还没有会话。</p>
    </div>`;
  }
  return `<div class="transcript-paper">${current.turns
    .map((turn, index) => {
      const event = activeEvent(turn.turn_id);
      const time =
        turn.meta?.start_sec === undefined
          ? ''
          : ` · ${Number(turn.meta.start_sec).toFixed(1)}–${Number(turn.meta.end_sec).toFixed(1)}s`;
      const badge = event ? resolverLabel(event.resolver) : 'OBSERVED';
      return `<article class="subtitle-line ${event ? 'revised' : ''}" data-turn="${escapeHtml(turn.turn_id)}">
        <div class="subtitle-meta">
          <span>TURN ${String(index + 1).padStart(2, '0')}</span>
          <span>${escapeHtml(turn.turn_id)}${time}</span>
          <span class="${event ? 'badge-revised' : ''}">${escapeHtml(badge)}</span>
        </div>
        ${renderTurnBody(turn, event)}
        ${
          event
            ? `<button class="revision-link evidence-note" data-event="${event.event_id}">
                <span>Why</span> ${escapeHtml(event.rationale || 'inspect dual-evidence gate')}
              </button>`
            : ''
        }
      </article>`;
    })
    .join('')}</div>`;
}

function renderImpact(): string {
  const impact = computeImpact(session);
  if (!session?.turns.length) {
    return `<div class="impact-strip idle">
      <div><b>—</b><span>revised turns</span></div>
      <div><b>—</b><span>rollbacks</span></div>
      <div><b>—</b><span>chars recovered</span></div>
      <div class="impact-note"><span>DEMO</span><p>Load R0015 to feel the lift</p></div>
    </div>`;
  }
  return `<div class="impact-strip ${impact.revised_turns ? 'hot' : ''}">
    <div><b>${impact.revised_turns}</b><span>revised turns</span></div>
    <div><b>${impact.rollbacks}</b><span>rollbacks</span></div>
    <div><b>${impact.deferred}</b><span>deferred</span></div>
    <div><b>+${impact.recovered_chars}</b><span>chars recovered</span></div>
    <div class="impact-note">
      <span>LIFT</span>
      <p>${escapeHtml(impactNote || impact.highlight || (impact.revised_turns ? 'Transcript improved after re-listen' : 'No revisions yet'))}</p>
    </div>
  </div>`;
}

function renderDecision(): string {
  const pending = Object.values(session?.open_hypotheses ?? {}).find((item) => item.status === 'active');
  if (!selected && pending) return renderHypothesis(pending);
  if (!selected) {
    return `<p class="empty">点一条 <b>REVISED</b> 字幕查看证据链。</p>`;
  }
  return `<div class="decision method-rail">
    <span class="status">${escapeHtml(resolverLabel(selected.resolver))}</span>
    <h3><span class="was">${escapeHtml(stripTime(selected.before_text) || '∅')}</span> <i>→</i> <span class="now">${escapeHtml(stripTime(selected.after_text || selected.replacement || ''))}</span></h3>
    <p>${escapeHtml(selected.rationale || 'Dual-evidence revision committed.')}</p>
    <dl>
      <dt>Resolver</dt><dd>${escapeHtml(selected.resolver || '—')}</dd>
      <dt>Target → Source</dt><dd>${escapeHtml(selected.target_turn_id)} ← ${escapeHtml(selected.source_turn_id)}</dd>
      <dt>Evidence</dt><dd>${selected.evidence.map(escapeHtml).join('<br>') || '—'}</dd>
      <dt>Confidence</dt><dd>${selected.score}</dd>
      <dt>Event state</dt><dd>${selected.active ? 'active' : 'inactive'}</dd>
    </dl>
  </div>`;
}

function renderMethodTrace(): string {
  return `<section class="method-trace-card">
    <span class="eyebrow">METHOD TRACE</span>
    <ol class="method-trace">
      <li><b>① OBSERVE</b><span>保留首遍 ASR 与时间戳，永不覆盖 raw。</span></li>
      <li><b>② MEMORY RETRIEVE</b><span>检索近期对话、未决假设与长期稳定事实。</span></li>
      <li><b>③ CONTEXT JUDGE</b><span>区分一致、新信息、冲突与不确定。</span></li>
      <li><b>④ TARGETED RELISTEN</b><span>只对冲突焦点做 closed-set 历史音频验证。</span></li>
      <li><b>⑤ EVENT REPLAY</b><span>从 raw 与只追加事件重建当前字幕。</span></li>
    </ol>
  </section>`;
}

function renderHypothesis(hypothesis: WorkingHypothesis): string {
  const choices = hypothesis.alternatives.map((candidate) => `<li><span>${escapeHtml(candidate)}</span></li>`).join('');
  return `<div class="decision method-rail candidate-card">
    <span class="status action-${escapeHtml(hypothesis.status)}">${escapeHtml(hypothesis.status.toUpperCase())}</span>
    <h3>${escapeHtml(hypothesis.current_interpretation)} <i>↔</i> ${escapeHtml(hypothesis.proposed_interpretation)}</h3>
    <ul class="candidate-list">${choices}</ul>
    <dl class="evidence-packet">
      <dt>Target turns</dt><dd>${hypothesis.target_turn_ids.map(escapeHtml).join(', ')}</dd>
      <dt>Evidence score</dt><dd>${Math.round(hypothesis.score * 100)}%</dd>
    </dl>
  </div>`;
}

function render(): void {
  const turns = session?.turns.length ?? 0;
  const events = activeEvents().length;
  const impact = computeImpact(session);
  root.innerHTML = `<main class="research-notebook">
    <header>
      <div>
        <span class="eyebrow">RETRACE-ASR · LIVING TRANSCRIPT</span>
        <h1>Hear it twice.</h1>
        <p>最新上下文可以强化新信息，也可以推翻旧解释；raw 始终保留，修订由上下文与音频共同决定。</p>
      </div>
      <div class="metrics">
        <div><b>${turns}</b><span>turns</span></div>
        <div><b class="${events ? 'pulse' : ''}">${events}</b><span>revisions</span></div>
        <div><b>${escapeHtml(session?.analysis_status || 'idle')}</b><span>agent</span></div>
      </div>
    </header>
    ${renderImpact()}
    <div class="notebook-grid">
      <section class="subtitle-stage">
        <div class="stage-head">
          <div><span class="eyebrow">LIVE TRANSCRIPT</span><h2>Conversation record</h2></div>
          <span>amber = revised · raw stays immutable</span>
        </div>
        ${renderTurns()}
      </section>
      <aside class="explainer">
        <div class="panel-head">
          <div><span class="eyebrow">AGENT NOTE</span><h2>Why this changed</h2></div>
          <span>append-only audit</span>
        </div>
        ${renderMethodTrace()}
        ${renderDecision()}
      </aside>
    </div>
    <section class="observation">
      <div>
        <span class="eyebrow">QWEN-OMNI AUDIO INPUT</span>
        <p>上传音频建立实时会话，Agent 会持续更新分析状态与修订账本。</p>
      </div>
      <div class="input-stack">
        <input id="session" value="${escapeHtml(session?.session_id || 'demo')}" aria-label="Session ID">
        <input id="audio" type="file" accept="audio/*" aria-label="Audio file">
        <button id="submit-audio" ${busy ? 'disabled' : ''}>Transcribe audio</button>
      </div>
      <textarea id="text" placeholder="Or append a text ASR observation（可粘贴：遥。遥遥遥遥。 并配合 audio meta 走服务端重听）" aria-label="ASR observation"></textarea>
      <button id="submit-text" ${busy ? 'disabled' : ''}>Append observation</button>
      <p class="status-line">${escapeHtml(status)}</p>
    </section>
  </main>`;

  document.querySelectorAll<HTMLButtonElement>('[data-event]').forEach((button) =>
    button.addEventListener('click', () => {
      selected = session?.revision_events.find((event) => event.event_id === button.dataset.event) ?? null;
      render();
      document.querySelector('.decision')?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }),
  );
  document.querySelector<HTMLButtonElement>('#submit-text')?.addEventListener('click', submitText);
  document.querySelector<HTMLButtonElement>('#submit-audio')?.addEventListener('click', submitAudio);
}

async function submitText(): Promise<void> {
  const sessionId = document.querySelector<HTMLInputElement>('#session')!.value.trim() || 'demo';
  const text = document.querySelector<HTMLTextAreaElement>('#text')!.value.trim();
  if (!text) return;
  busy = true;
  status = 'Appending observation…';
  render();
  const result = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}/turns`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ turn_id: `t${Date.now()}`, text }),
  });
  if (!result.ok) {
    status = `Request failed: ${await result.text()}`;
    busy = false;
    render();
    return;
  }
  const body = await result.json();
  session = body.session;
  status = 'Observation queued · analyzing context…';
  session = await waitForAnalysis(sessionId);
  const events = activeEvents();
  selected = events[events.length - 1] ?? null;
  impactNote = selected ? `Live revision via ${selected.resolver || 'agent'}` : '';
  status = session.analysis_status === 'deferred' ? 'Analysis deferred pending stronger evidence.' : 'Context analysis complete.';
  busy = false;
  render();
}

async function waitForAnalysis(sessionId: string): Promise<Session> {
  for (let attempt = 0; attempt < 30; attempt += 1) {
    await new Promise((resolve) => window.setTimeout(resolve, 100));
    const response = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}`);
    if (!response.ok) throw new Error(await response.text());
    const body = await response.json();
    const current = body.session as Session;
    if (current.analysis_status !== 'queued' && current.analysis_status !== 'analyzing') return current;
  }
  throw new Error('analysis timeout');
}

async function submitAudio(): Promise<void> {
  const sessionId = document.querySelector<HTMLInputElement>('#session')!.value.trim() || 'auto';
  const file = document.querySelector<HTMLInputElement>('#audio')!.files?.[0];
  if (!file) return;
  busy = true;
  status = 'Transcribing and analyzing context (may take a few minutes)…';
  impactNote = '';
  render();
  const form = new FormData();
  form.append('file', file);
  const result = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}/audio/upload`, { method: 'POST', body: form });
  if (!result.ok) {
    status = `Audio failed: ${await result.text()}`;
    busy = false;
    render();
    return;
  }
  const body = await result.json();
  session = body.session;
  const events = activeEvents();
  selected = events[0] ?? null;
  const impact = computeImpact(session);
  impactNote = impact.revised_turns
    ? `${impact.revised_turns} turns revised · +${impact.recovered_chars} chars`
    : 'ASR complete · no revisions this time';
  status = `${body.turn_count} turns · ${events.length} revisions.`;
  busy = false;
  render();
}

render();
