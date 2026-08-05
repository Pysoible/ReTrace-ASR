import './styles.css';

type Revision = { action: string; target_turn_id: string; source_turn_id: string; before_text: string; after_text: string; entity_id: string; score: number; evidence: string[]; resolver?: string };
type Session = { session_id: string; turns: Array<{ turn_id: string; raw_text: string; current_text: string; source?: string; meta?: { start_sec?: number; end_sec?: number; chunk_index?: number }; hypotheses: Array<{ span: string; action: string; entity_id?: string }> }>; verified_memory: Record<string, { name: string }>; quarantine_memory: Record<string, { name: string }> };
type IntegrationStatus = { deepseek: { ready: boolean; model?: string; transport?: string; error?: string }; qwen_asr: { enabled: boolean; loaded: boolean; model_path: string; model_exists: boolean } };
type AsrResult = { ok?: boolean; pass1_text?: string; final_text?: string; applied_terms?: string[]; elapsed_sec?: number; error?: string; chunked?: boolean; chunk_count?: number; duration_sec?: number };
type EvolveStep = {
  iteration: number;
  candidates?: string[];
  applied_terms?: string[];
  ingest?: { n_added?: number; n_rejected?: number; n_dup?: number; added?: string[]; dict_size?: number };
  text_changed?: boolean;
  stop_reason?: string | null;
  final_text?: string;
};
type EvolveResult = { iterations?: number; max_iters?: number; evolved?: boolean; history?: EvolveStep[]; final_text?: string };

let session: Session | null = null;
let revisions: Revision[] = [];
let selected: Revision | null = null;
let integrations: IntegrationStatus | null = null;
let lastAsr: AsrResult | null = null;
let lastEvolve: EvolveResult | null = null;
let busy = false;
let statusMessage = '';

let form = {
  sessionId: 'auto',
  turnId: '',
  text: '',
  audioPath: '',
  useLlm: true,
  correctLlm: false,
  twoPass: true,
  evolve: false,
  maxIters: 3,
};

let selectedFile: File | null = null;
let previewUrl: string | null = null;

const root = document.querySelector<HTMLDivElement>('#app')!;

function render(): void {
  const ds = integrations?.deepseek;
  const asr = integrations?.qwen_asr;
  const asrLabel = !asr ? 'unknown' : !asr.enabled ? 'disabled' : asr.loaded ? 'loaded' : 'enabled';
  root.innerHTML = `<main>
  <header class="topbar">
    <div class="brand">
      <span class="eyebrow">RETRACE-ASR</span>
      <h1>Retrospective Revision Lab</h1>
      <p>Audio → session · chunks → turns · evidence-grounded revise</p>
    </div>
    <div class="topbar-aside">
      <div class="metrics">
        <div><b>${session?.turns.length ?? 0}</b><span>turns</span></div>
        <div><b>${revisions.length}</b><span>revisions</span></div>
        <div><b>${Object.keys(session?.quarantine_memory ?? {}).length}</b><span>deferred</span></div>
      </div>
      <div class="status-chips">
        <span class="chip ${ds?.ready ? 'ok' : 'off'}">DeepSeek ${ds?.ready ? 'ready' : 'offline'}</span>
        <span class="chip ${asr?.loaded ? 'ok' : asr?.enabled ? 'warn' : 'off'}">Qwen ${escapeHtml(asrLabel)}</span>
      </div>
    </div>
  </header>

  ${statusMessage ? `<p class="status-banner">${escapeHtml(statusMessage)}</p>` : ''}

  <div class="studio">
    <aside class="controls">
      <section class="panel audio-panel">
        <div class="panel-intro">
          <span class="eyebrow">AUDIO · PRIMARY</span>
          <p>One long audio → one session. Auto-split ≤15s; each chunk is a turn.</p>
        </div>
        <div id="dropzone" class="dropzone ${selectedFile ? 'has-file' : ''}" tabindex="0" role="button" aria-label="Audio upload dropzone">
          <input id="audio-file" class="file-input" type="file" accept="audio/*,.wav,.mp3,.flac,.m4a,.ogg,.aac" ${busy ? 'disabled' : ''}>
          <div class="dropzone-body">
            <strong>${selectedFile ? escapeHtml(selectedFile.name) : 'Drop audio or click to choose'}</strong>
            <span>${selectedFile ? `${(selectedFile.size / 1024).toFixed(1)} KB · ${selectedFile.type || 'audio'}` : 'wav / mp3 / flac / m4a'}</span>
          </div>
        </div>
        ${previewUrl ? `<audio id="audio-preview" class="audio-preview" controls src="${previewUrl}"></audio>` : '<div class="audio-preview empty-preview">No local file selected</div>'}
        <label class="path-label" for="audio-path">Or server path</label>
        <input id="audio-path" value="${escapeAttr(form.audioPath)}" placeholder="/path/to/sample.wav" aria-label="Audio path" ${busy ? 'disabled' : ''}>
        <div class="field-row">
          <input id="session" value="${escapeAttr(form.sessionId)}" aria-label="Session ID" placeholder="session: auto">
          <input id="max_iters" type="number" min="1" max="8" value="${form.maxIters}" aria-label="Max evolve iterations" ${busy || !form.evolve ? 'disabled' : ''} title="max evolve iters">
        </div>
        <div class="checks">
          <label><input id="two_pass" type="checkbox" ${form.twoPass ? 'checked' : ''}> two-pass</label>
          <label><input id="evolve" type="checkbox" ${form.evolve ? 'checked' : ''}> self-evolve</label>
          <label><input id="use_llm" type="checkbox" ${form.useLlm ? 'checked' : ''}> DeepSeek revise</label>
        </div>
        <button id="submit-audio" class="btn-primary" ${busy ? 'disabled' : ''}>${busy ? (form.evolve ? 'Evolving…' : 'Transcribing…') : (form.evolve ? 'Evolve + transcribe' : 'Transcribe audio')}</button>
        ${renderEvolveResult()}
        ${renderAsrResult()}
      </section>

      <section class="panel text-panel">
        <div class="panel-intro">
          <span class="eyebrow">TEXT · SECONDARY</span>
          <p>Paste an ASR observation into the current session.</p>
        </div>
        <div class="field-row">
          <input id="turn" value="${escapeAttr(form.turnId)}" placeholder="Turn ID (optional)" aria-label="Turn ID">
          <label class="check-inline"><input id="correct_llm" type="checkbox" ${form.correctLlm ? 'checked' : ''}> correct first</label>
        </div>
        <textarea id="text" placeholder="Paste ASR text…" aria-label="ASR observation">${escapeHtml(form.text)}</textarea>
        <button id="submit-text" ${busy ? 'disabled' : ''}>${busy ? 'Running…' : 'Run text revision'}</button>
      </section>
    </aside>

    <section class="stage">
      <div class="stage-split">
        <section class="panel subtitle-stage">
          <div class="stage-head">
            <span class="eyebrow">LIVE SUBTITLES · ${escapeHtml(session?.session_id || form.sessionId || 'auto')}</span>
            <span>chunk = turn</span>
          </div>
          <div class="subtitle-scroll">
            ${renderTurns()}
          </div>
        </section>

        <aside class="panel explainer">
          <div class="panel-head">
            <div>
              <span class="eyebrow">DECISION EXPLAINER</span>
              <h2>Why this changed</h2>
            </div>
            <span>audit</span>
          </div>
          ${renderDecision()}
        </aside>
      </div>
    </section>
  </div>
</main>`;

  bindForm();
  document.querySelector('#submit-text')?.addEventListener('click', () => void submitText());
  document.querySelector('#submit-audio')?.addEventListener('click', () => void submitAudio());
  document.querySelectorAll<HTMLButtonElement>('[data-revision]').forEach((button) => {
    button.addEventListener('click', () => {
      const index = Number(button.dataset.revision);
      selected = revisions[index] ?? (!session?.turns.length ? demoRevision() : null);
      render();
    });
  });
  bindDropzone();
}

function bindForm(): void {
  const sessionEl = document.querySelector<HTMLInputElement>('#session');
  const turnEl = document.querySelector<HTMLInputElement>('#turn');
  const textEl = document.querySelector<HTMLTextAreaElement>('#text');
  const pathEl = document.querySelector<HTMLInputElement>('#audio-path');
  const useLlm = document.querySelector<HTMLInputElement>('#use_llm');
  const correctLlm = document.querySelector<HTMLInputElement>('#correct_llm');
  const twoPass = document.querySelector<HTMLInputElement>('#two_pass');
  const evolve = document.querySelector<HTMLInputElement>('#evolve');
  const maxIters = document.querySelector<HTMLInputElement>('#max_iters');
  sessionEl?.addEventListener('input', () => { form.sessionId = sessionEl.value; });
  turnEl?.addEventListener('input', () => { form.turnId = turnEl.value; });
  textEl?.addEventListener('input', () => { form.text = textEl.value; });
  pathEl?.addEventListener('input', () => { form.audioPath = pathEl.value; });
  useLlm?.addEventListener('change', () => { form.useLlm = useLlm.checked; });
  correctLlm?.addEventListener('change', () => { form.correctLlm = correctLlm.checked; });
  twoPass?.addEventListener('change', () => { form.twoPass = twoPass.checked; });
  evolve?.addEventListener('change', () => { form.evolve = evolve.checked; render(); });
  maxIters?.addEventListener('change', () => {
    const n = Number(maxIters.value);
    form.maxIters = Number.isFinite(n) ? Math.max(1, Math.min(8, Math.round(n))) : 3;
  });
}

function bindDropzone(): void {
  const zone = document.querySelector<HTMLDivElement>('#dropzone');
  const fileInput = document.querySelector<HTMLInputElement>('#audio-file');
  if (!zone || !fileInput) return;

  const openPicker = () => fileInput.click();
  zone.addEventListener('click', (event) => {
    if (event.target === fileInput) return;
    openPicker();
  });
  zone.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      openPicker();
    }
  });
  fileInput.addEventListener('change', () => {
    const file = fileInput.files?.[0] || null;
    setSelectedFile(file);
  });

  ;['dragenter', 'dragover'].forEach((type) => {
    zone.addEventListener(type, (event) => {
      event.preventDefault();
      zone.classList.add('dragover');
    });
  });
  ;['dragleave', 'drop'].forEach((type) => {
    zone.addEventListener(type, (event) => {
      event.preventDefault();
      zone.classList.remove('dragover');
    });
  });
  zone.addEventListener('drop', (event) => {
    const file = event.dataTransfer?.files?.[0] || null;
    if (file) setSelectedFile(file);
  });
}

function setSelectedFile(file: File | null): void {
  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
    previewUrl = null;
  }
  selectedFile = file;
  if (file) previewUrl = URL.createObjectURL(file);
  render();
}

function renderTurns(): string {
  const demo = !session?.turns.length;
  const turns = demo
    ? [
        { turn_id: '01', raw_text: '图博士让我交报告。', current_text: '涂博士让我交报告。', hypotheses: [] as Array<{ span: string; action: string }> },
        { turn_id: '04', raw_text: '实验室负责人下午汇报。', current_text: '实验室负责人下午汇报。', hypotheses: [] as Array<{ span: string; action: string }> },
      ]
    : session!.turns;
  return turns.map((turn, index) => {
    const revisionIndex = demo && turn.turn_id === '01' ? 0 : revisions.findIndex((item) => item.target_turn_id === turn.turn_id);
    const revision = revisionIndex >= 0
      ? (demo ? { source_turn_id: '04', before_text: turn.raw_text, after_text: turn.current_text } : revisions[revisionIndex])
      : null;
    const meta = (turn as { meta?: { start_sec?: number; end_sec?: number } }).meta;
    const timeLabel = meta?.start_sec != null && meta?.end_sec != null
      ? ` · ${meta.start_sec.toFixed(1)}-${meta.end_sec.toFixed(1)}s`
      : '';
    return `<article class="subtitle-line ${revision ? 'revised' : ''}">
      <span class="subtitle-meta">TURN ${index + 1}/${turns.length} · ${escapeHtml(turn.turn_id)}${timeLabel} ${revision ? '· REVISED' : '· ASR'}${(turn as { source?: string }).source ? ` · ${(turn as { source?: string }).source}` : ''}</span>
      <p>${revision
        ? `<mark class="incorrect">${escapeHtml(revision.before_text)}</mark> → <mark class="corrected">${escapeHtml(revision.after_text)}</mark>`
        : `“${escapeHtml(turn.current_text)}”`}</p>
      ${revision ? `<button class="revision-link" data-revision="${revisionIndex}">↳ evidence from Turn ${revision.source_turn_id}</button>` : ''}
    </article>`;
  }).join('');
}

function demoRevision(): Revision {
  return {
    action: 'REVISE',
    target_turn_id: '01',
    source_turn_id: '04',
    before_text: '图博士让我交报告。',
    after_text: '涂博士让我交报告。',
    entity_id: 'person:tu',
    score: 0.91,
    evidence: ['同音纠错', '后文实验室负责人指称'],
    resolver: 'retrospective',
  };
}

function activeRevision(): Revision | null {
  if (selected) return selected;
  if (!session?.turns.length && revisions.length === 0) return demoRevision();
  return null;
}

function renderDecision(): string {
  const item = activeRevision();
  if (!item) return '<p class="empty">点字幕里的 revision 链接，查看证据与审计记录。</p>';
  return `<div class="decision">
    <span class="status">${escapeHtml(item.action)}</span>
    <h3>${escapeHtml(item.before_text)} → ${escapeHtml(item.after_text)}</h3>
    <p>Evidence from <b>${escapeHtml(item.source_turn_id)}</b> revised <b>${escapeHtml(item.target_turn_id)}</b>${item.resolver ? ` via <b>${escapeHtml(item.resolver)}</b>` : ''}.</p>
    <dl>
      <dt>Entity</dt><dd>${escapeHtml(item.entity_id)}</dd>
      <dt>Score</dt><dd>${item.score}</dd>
      <dt>Evidence</dt><dd>${escapeHtml(item.evidence.join(' · '))}</dd>
    </dl>
    <small>Raw ASR preserved · append-only audit</small>
  </div>`;
}

function renderEvolveResult(): string {
  if (!lastEvolve) return '';
  const steps = lastEvolve.history || [];
  const summary = `${lastEvolve.iterations ?? steps.length}/${lastEvolve.max_iters ?? '?'} iters · ${lastEvolve.evolved ? 'dictionary updated' : 'no new terms'}`;
  const rows = steps.map((step) => {
    const added = step.ingest?.added?.length ? step.ingest.added.join('、') : '—';
    const cands = step.candidates?.length ? step.candidates.slice(0, 8).join('、') : '—';
    return `<li>
      <strong>iter ${step.iteration}</strong>
      · +${step.ingest?.n_added ?? 0} terms · dict ${step.ingest?.dict_size ?? '—'}
      ${step.stop_reason ? ` · stop:${escapeHtml(step.stop_reason)}` : ''}
      <br><span>candidates: ${escapeHtml(cands)}</span>
      <br><span>added: ${escapeHtml(added)}</span>
    </li>`;
  }).join('');
  return `<div class="asr-result evolve-result">
    <span class="eyebrow">SELF-EVOLVE</span>
    <p>${escapeHtml(summary)}</p>
    <ol>${rows}</ol>
  </div>`;
}

function renderAsrResult(): string {
  if (!lastAsr) return '';
  if (lastAsr.ok === false || lastAsr.error) {
    return `<div class="asr-result error"><span class="eyebrow">ASR RESULT</span><p>${escapeHtml(lastAsr.error || 'failed')}</p></div>`;
  }
  const applied = lastAsr.applied_terms?.length ? lastAsr.applied_terms.join('、') : '—';
  const chunkLine = lastAsr.chunked
    ? `${lastAsr.chunk_count ?? '?'} chunks · ${lastAsr.duration_sec ?? '—'}s`
    : `1 clip · ${lastAsr.duration_sec ?? '—'}s`;
  return `<div class="asr-result">
    <span class="eyebrow">ASR RESULT</span>
    <dl>
      <dt>Chunks</dt><dd>${escapeHtml(chunkLine)}</dd>
      <dt>Pass1</dt><dd>${escapeHtml(lastAsr.pass1_text || '')}</dd>
      <dt>Final</dt><dd>${escapeHtml(lastAsr.final_text || '')}</dd>
      <dt>Applied</dt><dd>${escapeHtml(applied)}</dd>
      <dt>Elapsed</dt><dd>${lastAsr.elapsed_sec ?? '—'}s</dd>
    </dl>
  </div>`;
}

function nextTurnId(): string {
  if (form.turnId.trim()) return form.turnId.trim();
  // Always unique: page refresh used to regenerate t1 and hit duplicate turn_id.
  const used = new Set((session?.turns || []).map((turn) => turn.turn_id));
  let n = (session?.turns.length ?? 0) + 1;
  let candidate = `t${n}`;
  while (used.has(candidate)) {
    n += 1;
    candidate = `t${n}`;
  }
  return `${candidate}-${Date.now().toString(36)}`;
}

async function submitText(): Promise<void> {
  const text = form.text.trim();
  if (!text) {
    statusMessage = 'Please enter ASR text, or use the audio panel.';
    render();
    return;
  }
  busy = true;
  statusMessage = 'Running text revision…';
  render();
  try {
    const response = await fetch(`/api/sessions/${encodeURIComponent(form.sessionId || 'demo')}/turns`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        turn_id: nextTurnId(),
        text,
        use_llm: form.useLlm,
        correct_with_llm: form.correctLlm,
      }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(formatError(result));
    applyResult(result);
    form.turnId = '';
    form.text = '';
    statusMessage = 'Text turn accepted.';
  } catch (error) {
    statusMessage = error instanceof Error ? error.message : String(error);
  } finally {
    busy = false;
    render();
  }
}

async function submitAudio(): Promise<void> {
  const sessionId = form.sessionId.trim() || 'auto';
  if (!selectedFile && !form.audioPath.trim()) {
    statusMessage = 'Choose an audio file or enter a server path.';
    render();
    return;
  }
  busy = true;
  lastEvolve = null;
  statusMessage = form.evolve
    ? `Self-evolving up to ${form.maxIters} iters (ASR → terms → dict → re-ASR)…`
    : selectedFile
      ? 'Uploading… one audio will become one session (chunks → turns).'
      : 'Transcribing… one audio will become one session (chunks → turns).';
  render();
  try {
    let response: Response;
    if (selectedFile) {
      const body = new FormData();
      body.append('file', selectedFile);
      body.append('turn_id', '');
      body.append('two_pass', String(form.twoPass));
      body.append('use_llm', String(form.useLlm));
      body.append('correct_with_llm', String(form.correctLlm));
      body.append('evolve', String(form.evolve));
      body.append('max_iters', String(form.maxIters));
      response = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}/audio/upload`, {
        method: 'POST',
        body,
      });
    } else {
      response = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}/audio`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          turn_id: 'unused',
          audio: form.audioPath.trim(),
          two_pass: form.twoPass,
          use_llm: form.useLlm,
          correct_with_llm: form.correctLlm,
          evolve: form.evolve,
          max_iters: form.maxIters,
        }),
      });
    }
    const result = await response.json();
    if (!response.ok) throw new Error(formatError(result));
    applyResult(result);
    lastAsr = result.asr || null;
    lastEvolve = result.evolve || null;
    if (result.session_id) form.sessionId = result.session_id;
    form.turnId = '';
    const evolveNote = lastEvolve
      ? ` · evolve ${lastEvolve.iterations}/${lastEvolve.max_iters}${lastEvolve.evolved ? ' · dict updated' : ''}`
      : '';
    statusMessage = `Session ${result.session_id || form.sessionId} · ${result.turn_count ?? session?.turns.length ?? 0} turns${evolveNote}.`;
  } catch (error) {
    statusMessage = error instanceof Error ? error.message : String(error);
  } finally {
    busy = false;
    render();
  }
}

function applyResult(result: { session: Session; revisions: Revision[] }): void {
  session = result.session;
  revisions = result.revisions || [];
  selected = revisions[0] ?? null;
}

function formatError(result: { detail?: unknown }): string {
  if (typeof result.detail === 'string') return result.detail;
  return JSON.stringify(result.detail || result);
}

function escapeHtml(value: string): string {
  return value
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;');
}

function escapeAttr(value: string): string {
  return escapeHtml(value).replaceAll("'", '&#39;');
}

async function loadStatus(): Promise<void> {
  try {
    const response = await fetch('/api/integrations/status');
    integrations = await response.json();
  } catch {
    integrations = null;
  }
}

async function loadSession(): Promise<void> {
  const id = (form.sessionId || '').trim();
  if (!id || id === 'auto') return;
  try {
    const response = await fetch(`/api/sessions/${encodeURIComponent(id)}`);
    if (!response.ok) return;
    const result = await response.json();
    if (result.session) {
      session = result.session;
      revisions = [];
      selected = null;
    }
  } catch {
    // ignore — empty local session is fine
  }
}

Promise.all([loadStatus(), loadSession()]).then(render);
