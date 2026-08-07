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
  reason?: string;
};
type Candidate = { text: string; score: number };
type Hypothesis = {
  span: string;
  candidates: Candidate[];
  decision: string;
  decision_rationale: string[];
  evidence_packet?: {
    asr_uncertainty?: { confidence?: number; nbest?: string[] };
    suspicious_span?: { reasons?: string[] } | null;
    later_raw_evidence?: string[];
    audio_verification?: { ok?: boolean; scores?: Record<string, number> };
    audio_retranscription?: { ok?: boolean; text?: string };
  };
};
type Turn = {
  turn_id: string;
  raw_text: string;
  current_text: string;
  hypotheses?: Hypothesis[];
  source?: string;
  meta?: { start_sec?: number; end_sec?: number; degenerate_relisten?: Record<string, unknown> };
};
type Session = {
  session_id: string;
  turns: Turn[];
  revision_events: RevisionEvent[];
  quarantine_memory: Record<string, unknown>;
};
type Impact = {
  revised_turns: number;
  open_relisten: number;
  semantic_gate: number;
  recovered_chars: number;
  highlight?: string;
};

const root = document.querySelector<HTMLDivElement>('#app')!;
let session: Session | null = null;
let selected: RevisionEvent | null = null;
let status = '上传音频，或一键回放 AliMeeting 修订样例，直观看到 ReTrace 的提升。';
let busy = false;
let impactNote = '';

function escapeHtml(value: string): string {
  return value.replace(/[&<>'"]/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#039;', '"': '&quot;' }[char]!));
}

function stripTime(text: string): string {
  return (text || '').replace(/^\[[0-9.]+-[0-9.]+\]\s*/, '');
}

function activeEvents(): RevisionEvent[] {
  return (session?.revision_events ?? []).filter((event) => event.active);
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
  const open = events.filter((event) => (event.resolver || '').includes('open-relisten')).length;
  const semantic = events.filter((event) => (event.resolver || '').includes('semantic')).length;
  const joined = (current?.turns ?? []).map((turn) => stripTime(turn.current_text)).join('');
  const highlight = joined.includes('骁龙') ? '骁龙 recovered' : joined.includes('麒麟') ? '麒麟 recovered' : undefined;
  return {
    revised_turns: new Set(events.map((event) => event.target_turn_id)).size,
    open_relisten: open,
    semantic_gate: semantic,
    recovered_chars: recovered,
    highlight,
  };
}

function resolverLabel(resolver?: string): string {
  if (!resolver) return 'REVISION';
  if (resolver.includes('open-relisten')) return 'OPEN RELISTEN';
  if (resolver.includes('semantic')) return 'AUDIO GATE';
  return resolver.toUpperCase();
}

function renderTurnBody(turn: Turn, event?: RevisionEvent): string {
  const raw = stripTime(turn.raw_text);
  const cur = stripTime(turn.current_text);
  if (!event) return `<p class="turn-text">${escapeHtml(turn.current_text)}</p>`;

  const isOpen = (event.resolver || '').includes('open-relisten') || raw !== cur;
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
      <p class="empty">还没有会话。点击右侧 <b>Replay R0015</b> 立刻看「遥遥遥遥 → 骁龙」的修订效果，或上传自己的音频。</p>
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
      <div><b>—</b><span>open re-listen</span></div>
      <div><b>—</b><span>chars recovered</span></div>
      <div class="impact-note"><span>DEMO</span><p>Load R0015 to feel the lift</p></div>
    </div>`;
  }
  return `<div class="impact-strip ${impact.revised_turns ? 'hot' : ''}">
    <div><b>${impact.revised_turns}</b><span>revised turns</span></div>
    <div><b>${impact.open_relisten}</b><span>open re-listen</span></div>
    <div><b>${impact.semantic_gate}</b><span>audio gate</span></div>
    <div><b>+${impact.recovered_chars}</b><span>chars recovered</span></div>
    <div class="impact-note">
      <span>LIFT</span>
      <p>${escapeHtml(impactNote || impact.highlight || (impact.revised_turns ? 'Transcript improved after re-listen' : 'No revisions yet'))}</p>
    </div>
  </div>`;
}

function renderDecision(): string {
  const pending = session?.turns
    .flatMap((turn) => (turn.hypotheses ?? []).map((hypothesis) => ({ turn, hypothesis })))
    .find((item) => item.hypothesis.decision !== 'COMMIT');
  if (!selected && pending) return renderHypothesis(pending.turn, pending.hypothesis);
  if (!selected) {
    return `<p class="empty">点一条 <b>REVISED</b> 字幕，查看开放重听 / 音频门控的证据链。</p>`;
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
      <li><b>② DETECT COLLAPSE</b><span>识别「遥遥遥遥 / 嗯嗯嗯」等退化转写。</span></li>
      <li><b>③ OPEN RELISTEN</b><span>对历史音频窗做开放重听并写回。</span></li>
      <li><b>④ SEMANTIC + AUDIO GATE</b><span>近形/回声冲突再走 closed-set 音频验证。</span></li>
      <li><b>⑤ AUDIT</b><span>每次修订可追溯、可回放。</span></li>
    </ol>
  </section>`;
}

function renderHypothesis(turn: Turn, hypothesis: Hypothesis): string {
  const asr = hypothesis.evidence_packet?.asr_uncertainty;
  const reasons =
    hypothesis.evidence_packet?.suspicious_span?.reasons?.join(' · ') || hypothesis.decision_rationale.join(' · ');
  const choices = hypothesis.candidates
    .map((candidate) => `<li><span>${escapeHtml(candidate.text)}</span><small>${Math.round(candidate.score * 100)}%</small></li>`)
    .join('');
  const audio = hypothesis.evidence_packet?.audio_verification;
  const audioScores = audio?.scores
    ? Object.entries(audio.scores)
        .map(([candidate, score]) => `${escapeHtml(candidate)}: ${Math.round(score * 100)}%`)
        .join('<br>')
    : 'Pending targeted re-listen';
  return `<div class="decision method-rail candidate-card">
    <span class="status action-${escapeHtml(hypothesis.decision)}">${escapeHtml(hypothesis.decision)}</span>
    <h3>Unresolved: ${escapeHtml(hypothesis.span)}</h3>
    <p>${escapeHtml(reasons || 'Awaiting independent evidence.')}</p>
    <ul class="candidate-list">${choices}</ul>
    <dl class="evidence-packet">
      <dt>ASR confidence</dt><dd>${asr?.confidence ?? '—'}</dd>
      <dt>Later raw evidence</dt><dd>${(hypothesis.evidence_packet?.later_raw_evidence ?? []).map(escapeHtml).join('<br>') || '—'}</dd>
      <dt>Audio verification</dt><dd>${audioScores}</dd>
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
        <p>首遍 ASR 可以崩坏；ReTrace 用开放重听与双证据门控把「遥遥遥遥」救回「骁龙」——修订看得见、证据点得开。</p>
      </div>
      <div class="metrics">
        <div><b>${turns}</b><span>turns</span></div>
        <div><b class="${events ? 'pulse' : ''}">${events}</b><span>revisions</span></div>
        <div><b>${impact.recovered_chars ? `+${impact.recovered_chars}` : '0'}</b><span>chars up</span></div>
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
        <div class="demo-actions">
          <button id="demo-r0015" class="demo-btn" ${busy ? 'disabled' : ''}>Replay R0015 lift</button>
          <p class="demo-caption">一键加载 AliMeeting 样例：3 处开放重听，含 遥→骁龙。</p>
        </div>
      </aside>
    </div>
    <section class="observation">
      <div>
        <span class="eyebrow">QWEN-OMNI AUDIO INPUT</span>
        <p>上传新音频跑完整链路；Agent 会自动重听退化 turn 并写回修订。</p>
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
  document.querySelector<HTMLButtonElement>('#demo-r0015')?.addEventListener('click', loadDemo);
}

async function loadDemo(): Promise<void> {
  busy = true;
  status = 'Loading AliMeeting R0015 revision replay…';
  render();
  try {
    const result = await fetch('/api/demo/r0015');
    if (!result.ok) {
      status = `Demo failed: ${await result.text()}`;
      busy = false;
      render();
      return;
    }
    const body = await result.json();
    session = body.session;
    const events = activeEvents();
    selected = events.find((event) => stripTime(event.before_text).includes('遥')) ?? events[0] ?? null;
    const impact = computeImpact(session);
    impactNote = body.impact_note || `CER 0.561 → 0.453 · ${impact.revised_turns} revised turns · 骁龙 recovered`;
    status = `Demo ready · ${events.length} revisions on ${session?.turns.length ?? 0} turns.`;
  } catch (error) {
    status = `Demo failed: ${error instanceof Error ? error.message : String(error)}`;
  }
  busy = false;
  render();
  const revised = document.querySelector('.subtitle-line.revised');
  revised?.scrollIntoView({ behavior: 'smooth', block: 'center' });
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
    body: JSON.stringify({ turn_id: `t${Date.now()}`, text, use_llm: true }),
  });
  if (!result.ok) {
    status = `Request failed: ${await result.text()}`;
    busy = false;
    render();
    return;
  }
  const body = await result.json();
  session = body.session;
  selected = body.revisions?.[0] ?? null;
  impactNote = selected ? `Live revision via ${selected.resolver || 'agent'}` : '';
  status = selected ? 'Observation appended · revision committed.' : 'Immutable observation appended.';
  busy = false;
  render();
}

async function submitAudio(): Promise<void> {
  const sessionId = document.querySelector<HTMLInputElement>('#session')!.value.trim() || 'auto';
  const file = document.querySelector<HTMLInputElement>('#audio')!.files?.[0];
  if (!file) return;
  busy = true;
  status = 'Transcribing + ReTrace revising (may take a few minutes)…';
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
