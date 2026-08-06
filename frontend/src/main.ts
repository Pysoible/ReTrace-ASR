import './styles.css';

type RevisionEvent = { event_id: string; action: string; target_turn_id: string; source_turn_id: string; before_text: string; after_text: string; evidence: string[]; score: number; active: boolean; reverted_event_id?: string; reason?: string };
type Turn = { turn_id: string; raw_text: string; current_text: string; source?: string; meta?: { start_sec?: number; end_sec?: number } };
type Session = { session_id: string; turns: Turn[]; revision_events: RevisionEvent[]; quarantine_memory: Record<string, unknown> };

const root = document.querySelector<HTMLDivElement>('#app')!;
let session: Session | null = null;
let selected: RevisionEvent | null = null;
let status = 'Ready for an immutable ASR observation.';

function escapeHtml(value: string): string { return value.replace(/[&<>'"]/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#039;', '"': '&quot;' }[char]!)); }

function activeEvent(turnId: string): RevisionEvent | undefined {
  return session?.revision_events.find((event) => event.target_turn_id === turnId && event.active && event.action !== 'UNDO_REVISION');
}

function renderTurns(): string {
  const current = session;
  if (!current?.turns.length) return '<p class="empty">上传音频或追加一条 ASR 观测后，字幕会按 Turn 依次出现。</p>';
  return current.turns.map((turn, index) => {
    const event = activeEvent(turn.turn_id);
    const time = turn.meta?.start_sec === undefined ? '' : ` · ${turn.meta.start_sec?.toFixed(1)}–${turn.meta.end_sec?.toFixed(1)}s`;
    const text = event ? `<mark class="incorrect">${escapeHtml(event.before_text)}</mark> → <mark class="corrected">${escapeHtml(event.after_text)}</mark>` : escapeHtml(turn.current_text);
    return `<article class="subtitle-line ${event ? 'revised' : ''}"><span class="subtitle-meta">TURN ${index + 1}/${current.turns.length} · ${escapeHtml(turn.turn_id)}${time}${event ? ' · REVISED' : ' · ASR'}</span><p>${text}</p>${event ? `<button class="revision-link" data-event="${event.event_id}">↳ evidence from Turn ${escapeHtml(event.source_turn_id)}</button>` : ''}</article>`;
  }).join('');
}

function renderDecision(): string {
  if (!selected) return '<p class="empty">选择一条已修订字幕，查看其后续证据、置信分数和可逆审计事件。</p>';
  const undo = selected.active && selected.action !== 'UNDO_REVISION'
    ? `<label class="undo-field">撤销理由 <input id="undo-reason" placeholder="可选：人工复核结论"></label><button id="undo" class="btn-danger">Undo revision</button>`
    : `<p class="empty">此事件已撤销${selected.reason ? `：${escapeHtml(selected.reason)}` : ''}。</p>`;
  return `<div class="decision"><span class="status">${escapeHtml(selected.action)}</span><h3>${escapeHtml(selected.before_text)} → ${escapeHtml(selected.after_text)}</h3><p>由 <b>${escapeHtml(selected.source_turn_id)}</b> 的后续证据影响 <b>${escapeHtml(selected.target_turn_id)}</b>。</p><dl><dt>Evidence</dt><dd>${selected.evidence.map(escapeHtml).join(' · ') || '—'}</dd><dt>Score</dt><dd>${selected.score}</dd><dt>State</dt><dd>${selected.active ? 'active' : 'reverted'}</dd></dl>${undo}</div>`;
}

function render(): void {
  const turns = session?.turns.length ?? 0;
  const events = session?.revision_events.length ?? 0;
  root.innerHTML = `<main><header><div><span class="eyebrow">RETRACE-ASR · EVIDENCE AGENT</span><h1>Retrospective Revision Lab</h1><p>Later conversational evidence can revise an earlier subtitle without erasing its raw ASR observation.</p></div><div class="metrics"><div><b>${turns}</b><span>turns</span></div><div><b>${events}</b><span>events</span></div><div><b>${Object.keys(session?.quarantine_memory ?? {}).length}</b><span>deferred</span></div></div></header><section class="subtitle-stage"><div class="stage-head"><span class="eyebrow">LIVE SUBTITLES</span><span>immutable observations · evidence-linked revisions · reversible audit</span></div>${renderTurns()}</section><section class="workspace"><aside class="explainer"><div class="panel-head"><div><span class="eyebrow">DECISION EXPLAINER</span><h2>Why this subtitle changed</h2></div><span>append-only audit</span></div>${renderDecision()}</aside></section><section class="observation"><div><span class="eyebrow">QWEN-OMNI AUDIO INPUT</span><p>One audio file becomes a session; silence-aware chunks become ordered Turns.</p></div><input id="session" value="${escapeHtml(session?.session_id || 'demo')}" aria-label="Session ID"><input id="audio" type="file" accept="audio/*" aria-label="Audio file"><button id="submit-audio">Transcribe audio</button><textarea id="text" placeholder="Or append a text ASR observation" aria-label="ASR observation"></textarea><button id="submit-text">Append observation</button><p class="status-line">${escapeHtml(status)}</p></section></main>`;
  document.querySelectorAll<HTMLButtonElement>('[data-event]').forEach((button) => button.addEventListener('click', () => { selected = session?.revision_events.find((event) => event.event_id === button.dataset.event) ?? null; render(); }));
  document.querySelector<HTMLButtonElement>('#submit-text')?.addEventListener('click', submitText);
  document.querySelector<HTMLButtonElement>('#submit-audio')?.addEventListener('click', submitAudio);
  document.querySelector<HTMLButtonElement>('#undo')?.addEventListener('click', undoSelected);
}

async function submitText(): Promise<void> {
  const sessionId = document.querySelector<HTMLInputElement>('#session')!.value.trim() || 'demo';
  const text = document.querySelector<HTMLTextAreaElement>('#text')!.value.trim();
  if (!text) return;
  const result = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}/turns`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ turn_id: `t${Date.now()}`, text }) });
  if (!result.ok) { status = `Request failed: ${await result.text()}`; render(); return; }
  const body = await result.json(); session = body.session; selected = body.revisions?.[0] ?? null; status = 'Immutable observation appended.'; render();
}

async function submitAudio(): Promise<void> {
  const sessionId = document.querySelector<HTMLInputElement>('#session')!.value.trim() || 'auto';
  const file = document.querySelector<HTMLInputElement>('#audio')!.files?.[0];
  if (!file) return;
  const form = new FormData(); form.append('file', file);
  status = 'Transcribing Qwen-Omni audio…'; render();
  const result = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}/audio/upload`, { method: 'POST', body: form });
  if (!result.ok) { status = `Audio failed: ${await result.text()}`; render(); return; }
  const body = await result.json(); session = body.session; selected = body.revisions?.[0] ?? null; status = `${body.turn_count} immutable turns appended.`; render();
}

async function undoSelected(): Promise<void> {
  if (!session || !selected) return;
  const reason = document.querySelector<HTMLInputElement>('#undo-reason')?.value || '';
  const result = await fetch(`/api/sessions/${encodeURIComponent(session.session_id)}/revisions/${encodeURIComponent(selected.event_id)}/undo`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ reason }) });
  if (!result.ok) { status = `Undo failed: ${await result.text()}`; render(); return; }
  const body = await result.json(); session = body.session; selected = body.event; status = 'Undo event appended; raw ASR remains intact.'; render();
}

render();
