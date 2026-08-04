import './styles.css';

type Revision = { action: string; target_turn_id: string; source_turn_id: string; before_text: string; after_text: string; entity_id: string; score: number; evidence: string[] };
type Session = { session_id: string; turns: Array<{ turn_id: string; raw_text: string; current_text: string; hypotheses: Array<{ span: string; action: string; entity_id?: string }> }>; verified_memory: Record<string, { name: string }>; quarantine_memory: Record<string, { name: string }> };

let session: Session | null = null;
let revisions: Revision[] = [];
let selected: Revision | null = null;

const root = document.querySelector<HTMLDivElement>('#app')!;

function render(): void {
  root.innerHTML = `<main><header><div><span class="eyebrow">RETRACE-ASR · EXPERIMENTAL SESSION</span><h1>Retrospective Revision Lab</h1><p>Evidence-grounded revision of conversational ASR hypotheses.</p></div><div class="metrics"><div><b>${session?.turns.length ?? 0}</b><span>turns</span></div><div><b>${revisions.length}</b><span>revisions</span></div><div><b>${Object.keys(session?.quarantine_memory ?? {}).length}</b><span>deferred</span></div></div></header><section class="observation"><div><span class="eyebrow">ASR OBSERVATION / INPUT</span><p>Submit a Qwen-Omni hypothesis to append an immutable observation.</p></div><input id="session" value="demo" aria-label="Session ID"><input id="turn" placeholder="Turn ID" aria-label="Turn ID"><textarea id="text" placeholder="Paste Qwen-Omni ASR observation" aria-label="ASR observation"></textarea><button id="submit">Run revision</button></section><section class="workspace"><div class="timeline"><div class="panel-head"><div><span class="eyebrow">01 / TRANSCRIPT</span><h2>Conversation timeline</h2></div><span>evidence unfolds by turn</span></div>${renderTurns()}</div><aside class="explainer"><div class="panel-head"><div><span class="eyebrow">02 / EXPLANATION</span><h2>Decision explainer</h2></div><span>append-only audit</span></div>${renderDecision()}</aside></section></main>`;
  document.querySelector('#submit')?.addEventListener('click', submit);
  document.querySelectorAll<HTMLButtonElement>('[data-revision]').forEach((button) => button.addEventListener('click', () => { selected = revisions[Number(button.dataset.revision)]; render(); }));
}

function renderTurns(): string {
  if (!session?.turns.length) return `<div class="trace"><span class="trace-label">EXAMPLE REVISION TRACE</span><article class="turn provisional"><span>TURN 01 · DEFER</span><p>“图博士让我交报告。”</p><small>candidate: 图博士 / 涂博士</small></article><div class="trace-link">later evidence</div><article class="turn evidence"><span>TURN 04 · EVIDENCE</span><p>“实验室负责人下午汇报。”</p></article><div class="trace-result">↳ revision target: TURN 01 · entity E1</div></div>`;
  return session.turns.map((turn) => { const revisionIndex = revisions.findIndex((item) => item.target_turn_id === turn.turn_id); return `<article class="turn"><span>TURN ${turn.turn_id}</span><p>${turn.current_text}</p>${turn.raw_text !== turn.current_text ? `<small>raw ASR: ${turn.raw_text}</small>` : ''}${turn.hypotheses.map((item) => `<em>${item.span} · ${item.action}</em>`).join('')}${revisionIndex >= 0 ? `<button data-revision="${revisionIndex}">↳ revised by later evidence</button>` : ''}</article>`; }).join('');
}

function renderDecision(): string {
  if (!selected) return '<p class="empty">Select a revision to inspect candidate evidence, score, and immutable audit record.</p>';
  return `<div class="decision"><span class="status">${selected.action}</span><h3>${selected.before_text} → ${selected.after_text}</h3><p>Evidence from <b>${selected.source_turn_id}</b> revised <b>${selected.target_turn_id}</b>.</p><dl><dt>Entity</dt><dd>${selected.entity_id}</dd><dt>Score</dt><dd>${selected.score}</dd><dt>Evidence</dt><dd>${selected.evidence.join(' · ')}</dd></dl><small>Raw ASR is preserved; this card is an append-only audit event.</small></div>`;
}

async function submit(): Promise<void> {
  const sessionId = (document.querySelector<HTMLInputElement>('#session')!).value || 'demo';
  const turnId = (document.querySelector<HTMLInputElement>('#turn')!).value || `t${(session?.turns.length ?? 0) + 1}`;
  const text = (document.querySelector<HTMLTextAreaElement>('#text')!).value.trim();
  if (!text) return;
  const response = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}/turns`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ turn_id: turnId, text }) });
  const result = await response.json();
  session = result.session; revisions = result.revisions; selected = revisions[0] ?? null; render();
}

render();
