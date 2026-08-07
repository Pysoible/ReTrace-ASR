import './styles.css';

type RevisionEvent = { event_id: string; action: string; target_turn_id: string; source_turn_id: string; before_text: string; after_text: string; evidence: string[]; score: number; active: boolean; reverted_event_id?: string; reason?: string };
type Candidate = { text: string; score: number };
type Hypothesis = { span: string; candidates: Candidate[]; decision: string; decision_rationale: string[]; evidence_packet?: { asr_uncertainty?: { confidence?: number; nbest?: string[] }; suspicious_span?: { reasons?: string[] } | null; later_raw_evidence?: string[]; audio_verification?: { ok?: boolean; scores?: Record<string, number> } } };
type Turn = { turn_id: string; raw_text: string; current_text: string; hypotheses?: Hypothesis[]; source?: string; meta?: { start_sec?: number; end_sec?: number } };
type Session = { session_id: string; turns: Turn[]; revision_events: RevisionEvent[]; quarantine_memory: Record<string, unknown> };

const root = document.querySelector<HTMLDivElement>('#app')!;
let session: Session | null = null;
let selected: RevisionEvent | null = null;
let status = 'Ready for an immutable ASR observation.';

function escapeHtml(value: string): string { return value.replace(/[&<>'"]/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#039;', '"': '&quot;' }[char]!)); }

function activeEvent(turnId: string): RevisionEvent | undefined {
  return session?.revision_events.find((event) => event.target_turn_id === turnId && event.active);
}

function renderTurns(): string {
  const current = session;
  if (!current?.turns.length) return '<div class="transcript-paper"><p class="empty">上传音频或追加一条 ASR 观测后，字幕会按 Turn 依次出现。</p></div>';
  return `<div class="transcript-paper">${current.turns.map((turn, index) => {
    const event = activeEvent(turn.turn_id);
    const time = turn.meta?.start_sec === undefined ? '' : ` · ${turn.meta.start_sec?.toFixed(1)}–${turn.meta.end_sec?.toFixed(1)}s`;
    const text = event ? `<mark class="incorrect">${escapeHtml(event.before_text)}</mark> → <mark class="corrected">${escapeHtml(event.after_text)}</mark>` : escapeHtml(turn.current_text);
    return `<article class="subtitle-line ${event ? 'revised' : ''}"><div class="subtitle-meta"><span>TURN ${String(index + 1).padStart(2, '0')}</span><span>${escapeHtml(turn.turn_id)}${time}</span><span>${event ? 'REVISED' : 'OBSERVED'}</span></div><p>${text}</p>${event ? `<button class="revision-link evidence-note" data-event="${event.event_id}"><span>Evidence note</span> Turn ${escapeHtml(event.source_turn_id)} → inspect rationale</button>` : ''}</article>`;
  }).join('')}</div>`;
}

function renderDecision(): string {
  const pending = session?.turns.flatMap((turn) => (turn.hypotheses ?? []).map((hypothesis) => ({ turn, hypothesis }))).find((item) => item.hypothesis.decision !== 'COMMIT');
  if (!selected && pending) return renderHypothesis(pending.turn, pending.hypothesis);
  if (!selected) return '<p class="empty">选择一条已修订字幕，查看其后续证据、置信分数和可逆审计事件。</p>';
  return `<div class="decision method-rail"><span class="status">${escapeHtml(selected.action)}</span><h3>${escapeHtml(selected.before_text)} <i>→</i> ${escapeHtml(selected.after_text)}</h3><p>后续 Turn <b>${escapeHtml(selected.source_turn_id)}</b> 提供了足以重新解释 <b>${escapeHtml(selected.target_turn_id)}</b> 的证据。</p><dl><dt>Evidence</dt><dd>${selected.evidence.map(escapeHtml).join('<br>') || '—'}</dd><dt>Confidence</dt><dd>${selected.score}</dd><dt>Event state</dt><dd>${selected.active ? 'active' : 'inactive'}</dd></dl></div>`;
}

function renderMethodTrace(): string {
  return `<section class="method-trace-card"><span class="eyebrow">METHOD TRACE</span><ol class="method-trace"><li><b>① OBSERVE</b><span>Preserve the first-pass ASR turn and timestamp.</span></li><li><b>② SEMANTIC TRIGGER</b><span>Verify a later raw conversational quote.</span></li><li><b>③ SELECTIVE RELISTEN</b><span>Re-check only the historical audio interval.</span></li><li><b>④ DUAL-EVIDENCE GATE</b><span>Require semantic and audio support for one closed candidate.</span></li><li><b>⑤ AUTONOMOUS DECISION</b><span>REVISE only on agreement; otherwise WAIT.</span></li></ol></section>`;
}

function renderHypothesis(turn: Turn, hypothesis: Hypothesis): string {
  const asr = hypothesis.evidence_packet?.asr_uncertainty;
  const reasons = hypothesis.evidence_packet?.suspicious_span?.reasons?.join(' · ') || hypothesis.decision_rationale.join(' · ');
  const choices = hypothesis.candidates.map((candidate) => `<li><span>${escapeHtml(candidate.text)}</span><small>${Math.round(candidate.score * 100)}%</small></li>`).join('');
  const audio = hypothesis.evidence_packet?.audio_verification;
  const audioScores = audio?.scores ? Object.entries(audio.scores).map(([candidate, score]) => `${escapeHtml(candidate)}: ${Math.round(score * 100)}%`).join('<br>') : 'Pending targeted re-listen';
  return `<div class="decision method-rail candidate-card"><span class="status action-${escapeHtml(hypothesis.decision)}">${escapeHtml(hypothesis.decision)}</span><h3>Unresolved: ${escapeHtml(hypothesis.span)}</h3><p>${escapeHtml(reasons || 'Awaiting independent evidence.')}</p><ul class="candidate-list">${choices}</ul><dl class="evidence-packet"><dt>ASR confidence</dt><dd>${asr?.confidence ?? '—'}</dd><dt>N-best</dt><dd>${(asr?.nbest ?? []).map(escapeHtml).join('<br>') || '—'}</dd><dt>Later raw evidence</dt><dd>${(hypothesis.evidence_packet?.later_raw_evidence ?? []).map(escapeHtml).join('<br>') || '—'}</dd><dt>Audio verification</dt><dd>${audioScores}</dd></dl></div>`;
}

function render(): void {
  const turns = session?.turns.length ?? 0;
  const events = session?.revision_events.length ?? 0;
  root.innerHTML = `<main class="research-notebook"><header><div><span class="eyebrow">RETRACE-ASR · RESEARCH NOTEBOOK</span><h1>Living Transcript</h1><p>Later conversational evidence can revise an earlier subtitle without erasing its raw ASR observation.</p></div><div class="metrics"><div><b>${turns}</b><span>turns</span></div><div><b>${events}</b><span>revisions</span></div><div><b>${Object.keys(session?.quarantine_memory ?? {}).length}</b><span>open questions</span></div></div></header><div class="notebook-grid"><section class="subtitle-stage"><div class="stage-head"><div><span class="eyebrow">LIVE TRANSCRIPT</span><h2>Conversation record</h2></div><span>raw observations remain intact</span></div>${renderTurns()}</section><aside class="explainer"><div class="panel-head"><div><span class="eyebrow">AGENT NOTE</span><h2>Why this changed</h2></div><span>append-only audit</span></div>${renderMethodTrace()}${renderDecision()}</aside></div><section class="observation"><div><span class="eyebrow">QWEN-OMNI AUDIO INPUT</span><p>Append a new observation; the Agent then reconsiders earlier language against later evidence.</p></div><div class="input-stack"><input id="session" value="${escapeHtml(session?.session_id || 'demo')}" aria-label="Session ID"><input id="audio" type="file" accept="audio/*" aria-label="Audio file"><button id="submit-audio">Transcribe audio</button></div><textarea id="text" placeholder="Or append a text ASR observation" aria-label="ASR observation"></textarea><button id="submit-text">Append observation</button><p class="status-line">${escapeHtml(status)}</p></section></main>`;
  document.querySelectorAll<HTMLButtonElement>('[data-event]').forEach((button) => button.addEventListener('click', () => { selected = session?.revision_events.find((event) => event.event_id === button.dataset.event) ?? null; render(); }));
  document.querySelector<HTMLButtonElement>('#submit-text')?.addEventListener('click', submitText);
  document.querySelector<HTMLButtonElement>('#submit-audio')?.addEventListener('click', submitAudio);
}

async function submitText(): Promise<void> {
  const sessionId = document.querySelector<HTMLInputElement>('#session')!.value.trim() || 'demo';
  const text = document.querySelector<HTMLTextAreaElement>('#text')!.value.trim();
  if (!text) return;
  const result = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}/turns`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ turn_id: `t${Date.now()}`, text, use_llm: true }) });
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

render();
