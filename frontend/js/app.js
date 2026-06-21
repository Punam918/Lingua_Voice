/* ═══════════════════════════════════════════════════════════════════
   LinguaVoice – Frontend Logic
   ═══════════════════════════════════════════════════════════════════ */

const API = '';   // same origin; set to 'http://localhost:8000' for local dev

// ── Language metadata ────────────────────────────────────────────────
const LANG_META = {
  english: { flag:'🇬🇧', name:'English',  color:'#6366f1', rgb:'99,102,241',  lt:'#a5b4fc', ph:'Hello, how are you today?' },
  german:  { flag:'🇩🇪', name:'Deutsch',  color:'#ef4444', rgb:'239,68,68',   lt:'#fca5a5', ph:'Guten Tag, wie geht es Ihnen?' },
  spanish: { flag:'🇪🇸', name:'Español',  color:'#f59e0b', rgb:'245,158,11',  lt:'#fde68a', ph:'Hola, ¿cómo estás hoy?' },
  french:  { flag:'🇫🇷', name:'Français', color:'#3b82f6', rgb:'59,130,246',  lt:'#93c5fd', ph:'Bonjour, comment allez-vous?' },
  nepali:  { flag:'🇳🇵', name:'नेपाली',   color:'#dc2626', rgb:'220,38,38',   lt:'#fca5a5', ph:'नमस्ते, तपाईलाई कस्तो छ?' },
};

// ── State ────────────────────────────────────────────────────────────
let selectedLang   = 'english';
let mediaRecorder  = null;
let audioChunks    = [];
let isRecording    = false;
let timerSecs      = 0;
let timerInterval  = null;
let waveAnimId     = null;
let idleWaveAnimId = null;
let audioCtx       = null;
let analyser       = null;
let micStream      = null;
let lastAudioBlob  = null;
let sessionHistory = [];

// ── DOM refs ─────────────────────────────────────────────────────────
const langCards      = document.querySelectorAll('.lang-card');
const navbar         = document.getElementById('navbar');
const navFlag        = document.getElementById('navFlag');
const navLangName    = document.getElementById('navLangName');
const statusPip      = document.getElementById('statusPip');
const statusLabel    = document.getElementById('statusLabel');
const recordBtn      = document.getElementById('recordBtn');
const micIcon        = document.getElementById('micIcon');
const stopIcon       = document.getElementById('stopIcon');
const micRings       = document.getElementById('micRings');
const recInfo        = document.getElementById('recInfo');
const timerVal       = document.getElementById('timerVal');
const recProgress    = document.getElementById('recProgress');
const uploadBtn      = document.getElementById('uploadBtn');
const audioFileInput = document.getElementById('audioFileInput');
const autoDetect     = document.getElementById('autoDetect');
const waveCanvas     = document.getElementById('waveCanvas');
const waveLabel      = document.getElementById('waveLabel');
const sttEmpty       = document.getElementById('sttEmpty');
const segList        = document.getElementById('segmentsList');
const transcriptFull = document.getElementById('transcriptFull');
const sttActions     = document.getElementById('sttActions');
const copyBtn        = document.getElementById('copyBtn');
const sendToTtsBtn   = document.getElementById('sendToTtsBtn');
const clearSttBtn    = document.getElementById('clearSttBtn');
const sttMeta        = document.getElementById('sttMeta');
const confWrap       = document.getElementById('confWrap');
const confFill       = document.getElementById('confFill');
const confPct        = document.getElementById('confPct');
const sttStatus      = document.getElementById('sttStatus');
const ttsInput       = document.getElementById('ttsInput');
const charCount      = document.getElementById('charCount');
const generateBtn    = document.getElementById('generateBtn');
const btnSpinner     = document.getElementById('btnSpinner');
const btnText        = generateBtn.querySelector('.btn-text');
const btnIcon        = generateBtn.querySelector('.btn-icon');
const ttsStatus      = document.getElementById('ttsStatus');
const playerCard     = document.getElementById('playerCard');
const playerCanvas   = document.getElementById('playerCanvas');
const playPauseBtn   = document.getElementById('playPauseBtn');
const playI          = playPauseBtn.querySelector('.play-i');
const pauseI         = playPauseBtn.querySelector('.pause-i');
const seekBar        = document.getElementById('seekBar');
const seekFill       = document.getElementById('seekFill');
const currTime       = document.getElementById('currTime');
const durTime        = document.getElementById('durTime');
const downloadBtn    = document.getElementById('downloadBtn');
const audioElem      = document.getElementById('audioElem');
const rateSlider     = document.getElementById('rateSlider');
const pitchSlider    = document.getElementById('pitchSlider');
const rateDisplay    = document.getElementById('rateDisplay');
const pitchDisplay   = document.getElementById('pitchDisplay');
const exampleChip    = document.querySelector('.ex-chip');
const toastWrap      = document.getElementById('toastWrap');
const historySection = document.getElementById('historySection');
const historyGrid    = document.getElementById('historyGrid');
const historyClearBtn= document.getElementById('historyClearBtn');
const bgCanvas       = document.getElementById('bgCanvas');
const particlesDiv   = document.getElementById('particles');

const wCtx  = waveCanvas  ? waveCanvas.getContext('2d')  : null;
const bgCtx = bgCanvas    ? bgCanvas.getContext('2d')    : null;

// ════════════════════════════════════════════════════════════════════
// ANIMATED BACKGROUND CANVAS – drifting aurora orbs + dot grid
// ════════════════════════════════════════════════════════════════════
const ORB_DEFS = [
  { x:0.12, y:0.18, r:0.38, hue:260, speed:0.00030, phase:0.0  },
  { x:0.80, y:0.25, r:0.30, hue:200, speed:0.00042, phase:1.7  },
  { x:0.50, y:0.78, r:0.42, hue:285, speed:0.00022, phase:3.2  },
  { x:0.88, y:0.72, r:0.26, hue:170, speed:0.00055, phase:2.1  },
  { x:0.30, y:0.60, r:0.22, hue:320, speed:0.00035, phase:4.5  },
];
let bgTick = 0;

function resizeBg() {
  if (!bgCanvas) return;
  bgCanvas.width  = window.innerWidth;
  bgCanvas.height = window.innerHeight;
}

function drawBg() {
  if (!bgCtx) return;
  bgTick++;
  const W = bgCanvas.width, H = bgCanvas.height;
  bgCtx.clearRect(0, 0, W, H);

  // Aurora orbs
  for (const o of ORB_DEFS) {
    const t  = bgTick * o.speed + o.phase;
    const cx = (o.x + Math.sin(t)       * 0.14) * W;
    const cy = (o.y + Math.cos(t * 0.7) * 0.11) * H;
    const r  = o.r * Math.min(W, H);

    const g = bgCtx.createRadialGradient(cx, cy, 0, cx, cy, r);
    g.addColorStop(0,   `hsla(${o.hue},80%,58%,0.10)`);
    g.addColorStop(0.5, `hsla(${o.hue},70%,45%,0.04)`);
    g.addColorStop(1,   `hsla(${o.hue},60%,38%,0.00)`);

    bgCtx.save();
    bgCtx.translate(cx, cy);
    bgCtx.rotate(t * 0.15);
    bgCtx.scale(1, 0.65);
    bgCtx.translate(-cx, -cy);
    bgCtx.fillStyle = g;
    bgCtx.beginPath();
    bgCtx.arc(cx, cy, r, 0, Math.PI * 2);
    bgCtx.fill();
    bgCtx.restore();
  }

  // Subtle dot grid
  const spacing = 64;
  for (let x = spacing / 2; x < W; x += spacing) {
    for (let y = spacing / 2; y < H; y += spacing) {
      const alpha = (Math.sin(bgTick * 0.018 + (x + y) * 0.009) * 0.5 + 0.5) * 0.035;
      bgCtx.fillStyle = `rgba(255,255,255,${alpha.toFixed(3)})`;
      bgCtx.beginPath();
      bgCtx.arc(x, y, 1.5, 0, Math.PI * 2);
      bgCtx.fill();
    }
  }

  requestAnimationFrame(drawBg);
}

// ════════════════════════════════════════════════════════════════════
// FLOATING PARTICLES
// ════════════════════════════════════════════════════════════════════
function spawnParticles() {
  if (!particlesDiv) return;
  for (let i = 0; i < 22; i++) {
    const el   = document.createElement('div');
    el.className = 'particle';
    const size = Math.random() * 4 + 2;
    Object.assign(el.style, {
      width:            `${size}px`,
      height:           `${size}px`,
      left:             `${Math.random() * 100}%`,
      bottom:           '-20px',
      animationDuration:`${Math.random() * 16 + 10}s`,
      animationDelay:   `${Math.random() * 18}s`,
    });
    particlesDiv.appendChild(el);
  }
}

// ════════════════════════════════════════════════════════════════════
// INIT
// ════════════════════════════════════════════════════════════════════
(async function init() {
  resizeBg();
  drawBg();
  spawnParticles();

  if (wCtx)         { resizeCanvas(waveCanvas);   animateIdleWave(); }
  if (playerCanvas) { resizeCanvas(playerCanvas); drawPlayerBar(0); }

  checkHealth();
  updateExample();

  window.addEventListener('resize', () => {
    resizeBg();
    if (wCtx)         { resizeCanvas(waveCanvas);   if (!isRecording) drawIdleWave(); }
    if (playerCanvas)   resizeCanvas(playerCanvas);
  });

  window.addEventListener('scroll', () => {
    if (!navbar) return;
    navbar.style.boxShadow = window.scrollY > 30
      ? '0 4px 40px rgba(0,0,0,0.55)'
      : '';
  });
})();

function resizeCanvas(canvas) {
  if (!canvas?.parentElement) return;
  const rect    = canvas.parentElement.getBoundingClientRect();
  const dpr     = devicePixelRatio || 1;
  canvas.width  = rect.width  * dpr;
  canvas.height = rect.height * dpr;
}

// ── Health check ──────────────────────────────────────────────────────
async function checkHealth() {
  try {
    const r = await fetch(`${API}/health`, { signal: AbortSignal.timeout(6000) });
    if (r.ok) {
      const d = await r.json();
      statusPip.className     = 'status-pip online';
      statusLabel.textContent = `Online · Whisper ${d.model || 'base'}`;
    } else throw new Error();
  } catch {
    statusPip.className     = 'status-pip error';
    statusLabel.textContent = 'API unreachable';
  }
}

// ── Language selection ────────────────────────────────────────────────
langCards.forEach(btn => {
  btn.addEventListener('click', () => {
    langCards.forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    selectedLang = btn.dataset.lang;
    applyAccentColor(selectedLang);
    updateNavLang(selectedLang);
    updateExample();
  });
});

function applyAccentColor(lang) {
  const m = LANG_META[lang];
  if (!m) return;
  const root = document.documentElement;
  root.style.setProperty('--accent',     m.color);
  root.style.setProperty('--accent-rgb', m.rgb);
  root.style.setProperty('--accent-lt',  m.lt);
  root.style.setProperty('--accent-glow',`rgba(${m.rgb},0.25)`);
  root.style.setProperty('--shadow-btn', `0 8px 32px rgba(${m.rgb},0.4)`);
}

function updateNavLang(lang) {
  const m = LANG_META[lang];
  if (!m) return;
  navFlag.textContent     = m.flag;
  navLangName.textContent = m.name;
  // tiny bounce via re-triggering animation
  navFlag.style.animation = 'none';
  requestAnimationFrame(() => { navFlag.style.animation = ''; });
}

function updateExample() {
  const ph = LANG_META[selectedLang]?.ph || '';
  if (exampleChip) {
    exampleChip.textContent     = ph;
    exampleChip.dataset.example = ph;
  }
  if (ttsInput) ttsInput.placeholder = ph;
}

exampleChip?.addEventListener('click', () => {
  ttsInput.value        = exampleChip.dataset.example || '';
  charCount.textContent = ttsInput.value.length;
  showToast('Example loaded', 'success');
});

// ── STT: Recording ────────────────────────────────────────────────────
recordBtn.addEventListener('click', () => {
  isRecording ? stopRecording() : startRecording();
});

async function startRecording() {
  try {
    micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch {
    showToast('Microphone access denied', 'error');
    return;
  }

  audioChunks   = [];
  const mime    = getSupportedMime();
  mediaRecorder = new MediaRecorder(micStream, { mimeType: mime });
  mediaRecorder.ondataavailable = e => { if (e.data.size > 0) audioChunks.push(e.data); };
  mediaRecorder.onstop = async () => {
    const blob = new Blob(audioChunks, { type: mime });
    micStream.getTracks().forEach(t => t.stop());
    stopVisualiser();
    await runSTT(blob, mime);
  };

  mediaRecorder.start(100);
  isRecording = true;

  recordBtn.classList.add('recording');
  micIcon.classList.add('hidden');
  stopIcon.classList.remove('hidden');
  micRings.classList.add('pulsing');
  recInfo.style.display = 'flex';
  if (waveLabel) waveLabel.textContent = 'Recording…';
  timerSecs             = 0;
  timerVal.textContent  = '0:00';
  if (recProgress) recProgress.style.strokeDashoffset = 226;

  timerInterval = setInterval(() => {
    timerSecs++;
    timerVal.textContent = fmtTime(timerSecs);
    if (recProgress) {
      const pct = Math.min(timerSecs / 180, 1);
      recProgress.style.strokeDashoffset = 226 - (226 * pct);
    }
    if (timerSecs >= 180) stopRecording();
  }, 1000);

  startVisualiser(micStream);
  setSTTStatus('loading', 'Recording…');
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state !== 'inactive') mediaRecorder.stop();
  isRecording = false;
  clearInterval(timerInterval);
  recordBtn.classList.remove('recording');
  micIcon.classList.remove('hidden');
  stopIcon.classList.add('hidden');
  micRings.classList.remove('pulsing');
  recInfo.style.display = 'none';
  if (recProgress) recProgress.style.strokeDashoffset = 226;
  if (waveLabel) waveLabel.textContent = 'Processing…';
  setSTTStatus('loading', 'Processing audio…');
}

function getSupportedMime() {
  for (const m of ['audio/webm;codecs=opus','audio/webm','audio/ogg;codecs=opus','audio/ogg','audio/mp4'])
    if (MediaRecorder.isTypeSupported(m)) return m;
  return 'audio/webm';
}

// ── STT: File upload ──────────────────────────────────────────────────
uploadBtn.addEventListener('click', () => audioFileInput.click());
audioFileInput.addEventListener('change', async () => {
  const file = audioFileInput.files[0];
  if (!file) return;
  setSTTStatus('loading', `Loading "${file.name}"…`);
  if (waveLabel) waveLabel.textContent = 'Uploading…';
  await runSTT(file, file.type);
  audioFileInput.value = '';
});

// ── STT: API call ─────────────────────────────────────────────────────
async function runSTT(blob, mimeType) {
  const ext  = mimeType.includes('ogg') ? '.ogg' : mimeType.includes('mp4') ? '.m4a' : '.webm';
  const file = new File([blob], `rec${ext}`, { type: mimeType });
  const form = new FormData();
  form.append('audio',       file);
  form.append('language',    selectedLang);
  form.append('auto_detect', autoDetect.checked ? 'true' : 'false');

  setSTTStatus('loading', 'Transcribing with Whisper…');

  try {
    const res = await fetch(`${API}/api/transcribe`, { method: 'POST', body: form });
    if (!res.ok) {
      const e = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(e.detail || 'Server error');
    }
    const data = await res.json();
    renderTranscript(data);
    const det = data.detected_language ? ` · detected: ${data.detected_language}` : '';
    setSTTStatus('success', `Done in ${(data.elapsed_sec ?? 0).toFixed(2)}s${det}`);
    if (waveLabel) waveLabel.textContent = 'Ready to record';
    addHistory('stt', selectedLang, data.text);
  } catch (err) {
    setSTTStatus('error', `Error: ${err.message}`);
    if (waveLabel) waveLabel.textContent = 'Ready to record';
    showToast(err.message, 'error');
  }
}

function renderTranscript(data) {
  sttEmpty.style.display       = 'none';
  transcriptFull.style.display = 'none';
  segList.style.display        = 'none';

  if (data.segments && data.segments.length > 1) {
    segList.innerHTML = '';
    data.segments.forEach((s, i) => {
      const item = document.createElement('div');
      item.className          = 'segment-item';
      item.style.animationDelay = `${i * 0.04}s`;
      item.innerHTML = `
        <span class="segment-time">${fmtTime(Math.round(s.start))} – ${fmtTime(Math.round(s.end))}</span>
        <span class="segment-text">${escHtml(s.text)}</span>`;
      segList.appendChild(item);
    });
    segList.style.display    = 'flex';
    transcriptFull.textContent = data.text; // keep for copy/send
  } else {
    transcriptFull.textContent   = data.text;
    transcriptFull.style.display = 'block';
  }

  sttActions.style.display = 'flex';
  const words = data.text.trim().split(/\s+/).filter(Boolean).length;
  sttMeta.textContent = `${words} word${words !== 1 ? 's' : ''}`;

  // Confidence bar
  if (data.confidence != null) {
    const pct = Math.round(data.confidence * 100);
    confFill.style.width   = `${pct}%`;
    confPct.textContent    = `${pct}%`;
    confWrap.style.display = 'flex';
  } else {
    confWrap.style.display = 'none';
  }
}

// ── STT: Actions ──────────────────────────────────────────────────────
copyBtn.addEventListener('click', () => {
  navigator.clipboard.writeText(transcriptFull.textContent.trim())
    .then(() => showToast('Copied!', 'success'));
});

sendToTtsBtn.addEventListener('click', () => {
  ttsInput.value          = transcriptFull.textContent;
  charCount.textContent   = ttsInput.value.length;
  ttsInput.closest('.card').scrollIntoView({ behavior: 'smooth', block: 'start' });
  showToast('Text sent to TTS', 'success');
});

clearSttBtn.addEventListener('click', () => {
  transcriptFull.textContent   = '';
  transcriptFull.style.display = 'none';
  segList.innerHTML            = '';
  segList.style.display        = 'none';
  sttEmpty.style.display       = 'flex';
  sttActions.style.display     = 'none';
  confWrap.style.display       = 'none';
  setSTTStatus('', '');
  if (waveLabel) waveLabel.textContent = 'Ready to record';
});

// ── TTS ───────────────────────────────────────────────────────────────
ttsInput.addEventListener('input', () => { charCount.textContent = ttsInput.value.length; });

rateSlider.addEventListener('input', () => {
  const v = parseInt(rateSlider.value);
  rateDisplay.textContent = v === 0 ? 'Normal' : (v > 0 ? `+${v}%` : `${v}%`);
});

pitchSlider.addEventListener('input', () => {
  const v = parseInt(pitchSlider.value);
  pitchDisplay.textContent = v === 0 ? 'Normal' : (v > 0 ? `+${v}Hz` : `${v}Hz`);
});

generateBtn.addEventListener('click', async () => {
  const text = ttsInput.value.trim();
  if (!text) { showToast('Please enter some text first', 'error'); return; }

  const gender = document.querySelector('input[name="gender"]:checked').value;
  const rate   = fmtRate(parseInt(rateSlider.value));
  const pitch  = fmtPitch(parseInt(pitchSlider.value));

  const form = new FormData();
  form.append('text', text);
  form.append('language', selectedLang);
  form.append('gender',   gender);
  form.append('rate',     rate);
  form.append('pitch',    pitch);

  generateBtn.disabled = true;
  btnIcon.classList.add('hidden');
  btnSpinner.classList.remove('hidden');
  btnText.textContent = 'Generating…';
  setTTSStatus('loading', 'Synthesizing speech…');
  playerCard.style.display = 'none';

  try {
    const res = await fetch(`${API}/api/synthesize`, { method: 'POST', body: form });
    if (!res.ok) {
      const e = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(e.detail || 'Server error');
    }
    const audioBlob = await res.blob();
    lastAudioBlob   = audioBlob;
    audioElem.src   = URL.createObjectURL(audioBlob);
    playerCard.style.display = 'block';
    audioElem.play();
    setTTSStatus('success', 'Ready – playing');
    addHistory('tts', selectedLang, text);
  } catch (err) {
    setTTSStatus('error', `Error: ${err.message}`);
    showToast(err.message, 'error');
  } finally {
    generateBtn.disabled = false;
    btnSpinner.classList.add('hidden');
    btnIcon.classList.remove('hidden');
    btnText.textContent = 'Generate Speech';
  }
});

// ── Audio player controls ─────────────────────────────────────────────
playPauseBtn.addEventListener('click', () => {
  audioElem.paused ? audioElem.play() : audioElem.pause();
});

audioElem.addEventListener('play',  () => {
  playI.classList.add('hidden');
  pauseI.classList.remove('hidden');
});
audioElem.addEventListener('pause', () => {
  playI.classList.remove('hidden');
  pauseI.classList.add('hidden');
});
audioElem.addEventListener('ended', () => {
  playI.classList.remove('hidden');
  pauseI.classList.add('hidden');
  seekBar.value        = 0;
  seekFill.style.width = '0%';
});

audioElem.addEventListener('loadedmetadata', () => {
  durTime.textContent = fmtTime(Math.round(audioElem.duration));
  resizeCanvas(playerCanvas);
  drawPlayerBar(0);
});

audioElem.addEventListener('timeupdate', () => {
  if (!audioElem.duration) return;
  const pct            = (audioElem.currentTime / audioElem.duration) * 100;
  seekBar.value        = pct;
  seekFill.style.width = `${pct}%`;
  currTime.textContent = fmtTime(Math.round(audioElem.currentTime));
  drawPlayerBar(pct / 100);
});

seekBar.addEventListener('input', () => {
  if (!audioElem.duration) return;
  audioElem.currentTime = (seekBar.value / 100) * audioElem.duration;
});

downloadBtn.addEventListener('click', () => {
  if (!lastAudioBlob) return;
  const a      = document.createElement('a');
  a.href       = URL.createObjectURL(lastAudioBlob);
  a.download   = `lingua_${selectedLang}_${Date.now()}.mp3`;
  a.click();
});

function drawPlayerBar(progress = 0) {
  if (!playerCanvas) return;
  const ctx = playerCanvas.getContext('2d');
  const W = playerCanvas.width, H = playerCanvas.height;
  ctx.clearRect(0, 0, W, H);
  const m        = LANG_META[selectedLang];
  const barCount = Math.floor(W / 4);
  for (let i = 0; i < barCount; i++) {
    const h      = (Math.sin(i * 0.31) * 0.38 + Math.sin(i * 0.073) * 0.38 + 0.24) * H * 0.72;
    const x      = i * 4;
    const filled = (i / barCount) <= progress;
    if (filled) {
      const g = ctx.createLinearGradient(0, 0, W, 0);
      g.addColorStop(0, m.color);
      g.addColorStop(1, '#06b6d4');
      ctx.fillStyle = g;
    } else {
      ctx.fillStyle = 'rgba(255,255,255,0.09)';
    }
    ctx.beginPath();
    if (ctx.roundRect) {
      ctx.roundRect(x, (H - h) / 2, 2.5, h, 1.5);
    } else {
      ctx.rect(x, (H - h) / 2, 2.5, h);
    }
    ctx.fill();
  }
}

// ── Waveform visualiser ───────────────────────────────────────────────
function drawIdleWave() {
  if (!wCtx || !waveCanvas) return;
  const W = waveCanvas.width, H = waveCanvas.height;
  wCtx.clearRect(0, 0, W, H);
  const now = Date.now() * 0.001;

  // Primary wave
  const g1 = wCtx.createLinearGradient(0, 0, W, 0);
  g1.addColorStop(0,   'rgba(99,102,241,0.15)');
  g1.addColorStop(0.5, 'rgba(6,182,212,0.25)');
  g1.addColorStop(1,   'rgba(99,102,241,0.15)');
  wCtx.strokeStyle = g1;
  wCtx.lineWidth   = 2;
  wCtx.beginPath();
  for (let x = 0; x < W; x++) {
    const y = H / 2 + Math.sin(x * 0.014 + now) * 7 + Math.sin(x * 0.007 + now * 0.5) * 4;
    x === 0 ? wCtx.moveTo(x, y) : wCtx.lineTo(x, y);
  }
  wCtx.stroke();

  // Secondary softer wave
  wCtx.strokeStyle = 'rgba(255,255,255,0.05)';
  wCtx.lineWidth   = 1;
  wCtx.beginPath();
  for (let x = 0; x < W; x++) {
    const y = H / 2 + Math.sin(x * 0.021 + now * 0.7 + 1.2) * 4;
    x === 0 ? wCtx.moveTo(x, y) : wCtx.lineTo(x, y);
  }
  wCtx.stroke();
}

function animateIdleWave() {
  drawIdleWave();
  idleWaveAnimId = requestAnimationFrame(animateIdleWave);
}

function startVisualiser(stream) {
  cancelAnimationFrame(idleWaveAnimId);
  idleWaveAnimId = null;

  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  analyser = audioCtx.createAnalyser();
  analyser.fftSize = 512;
  audioCtx.createMediaStreamSource(stream).connect(analyser);
  const data = new Uint8Array(analyser.frequencyBinCount);

  const draw = () => {
    waveAnimId = requestAnimationFrame(draw);
    analyser.getByteTimeDomainData(data);
    const W = waveCanvas.width, H = waveCanvas.height;
    wCtx.clearRect(0, 0, W, H);

    const m  = LANG_META[selectedLang];
    const g  = wCtx.createLinearGradient(0, 0, W, 0);
    g.addColorStop(0,   m.color);
    g.addColorStop(0.5, '#06b6d4');
    g.addColorStop(1,   m.color);
    wCtx.strokeStyle = g;
    wCtx.lineWidth   = 2.5;
    wCtx.shadowBlur  = 14;
    wCtx.shadowColor = m.color;
    wCtx.beginPath();
    const step = W / data.length;
    data.forEach((v, i) => {
      const y = ((v / 128) - 1) * (H / 2.2) + H / 2;
      i === 0 ? wCtx.moveTo(0, y) : wCtx.lineTo(i * step, y);
    });
    wCtx.lineTo(W, H / 2);
    wCtx.stroke();
    wCtx.shadowBlur = 0;
  };
  draw();
}

function stopVisualiser() {
  cancelAnimationFrame(waveAnimId);
  waveAnimId = null;
  if (audioCtx) { audioCtx.close(); audioCtx = null; }
  animateIdleWave();
}

// ════════════════════════════════════════════════════════════════════
// SESSION HISTORY
// ════════════════════════════════════════════════════════════════════
function addHistory(type, lang, text) {
  if (!text?.trim()) return;
  const m = LANG_META[lang];
  sessionHistory.unshift({
    type, lang, text: text.trim(),
    flag: m?.flag || '',
    time: new Date().toLocaleTimeString([], { hour:'2-digit', minute:'2-digit' }),
  });
  if (sessionHistory.length > 24) sessionHistory.pop();
  renderHistory();
  historySection.style.display = 'block';
}

function renderHistory() {
  historyGrid.innerHTML = '';
  sessionHistory.forEach((e, idx) => {
    const el = document.createElement('div');
    el.className = 'history-item';
    el.style.animationDelay = `${idx * 0.03}s`;
    el.innerHTML = `
      <div class="history-item-header">
        <span class="history-badge ${e.type}">${e.type === 'stt' ? '🎙 STT' : '🔊 TTS'}</span>
        <span class="history-lang">${e.flag} ${e.lang}</span>
      </div>
      <div class="history-text">${escHtml(e.text)}</div>
      <div class="history-time">${e.time}</div>
      <div class="history-item-actions">
        <button class="history-act-btn" onclick="histCopy(${idx})">Copy</button>
        ${e.type === 'stt' ? `<button class="history-act-btn" onclick="histToTTS(${idx})">→ TTS</button>` : ''}
      </div>`;
    historyGrid.appendChild(el);
  });
}

window.histCopy  = idx => navigator.clipboard.writeText(sessionHistory[idx].text).then(() => showToast('Copied!', 'success'));
window.histToTTS = idx => {
  ttsInput.value        = sessionHistory[idx].text;
  charCount.textContent = ttsInput.value.length;
  ttsInput.closest('.card').scrollIntoView({ behavior:'smooth', block:'start' });
  showToast('Loaded in TTS', 'success');
};

historyClearBtn.addEventListener('click', () => {
  sessionHistory              = [];
  historyGrid.innerHTML       = '';
  historySection.style.display = 'none';
});

// ── Status helpers ────────────────────────────────────────────────────
function setSTTStatus(type, msg) {
  sttStatus.className   = `inline-status ${type}`;
  sttStatus.textContent = msg;
}
function setTTSStatus(type, msg) {
  ttsStatus.className   = `inline-status ${type}`;
  ttsStatus.textContent = msg;
}

// ── Toast ─────────────────────────────────────────────────────────────
function showToast(msg, type = '', duration = 3200) {
  const el         = document.createElement('div');
  el.className     = `toast-item ${type}`;
  el.textContent   = msg;
  toastWrap.appendChild(el);
  requestAnimationFrame(() => requestAnimationFrame(() => el.classList.add('show')));
  setTimeout(() => {
    el.classList.remove('show');
    setTimeout(() => el.remove(), 300);
  }, duration);
}

// ── Utilities ─────────────────────────────────────────────────────────
function fmtTime(s) {
  const m = Math.floor(s / 60);
  return `${m}:${String(s % 60).padStart(2, '0')}`;
}
function fmtRate(v)  { return v >= 0 ? `+${v}%` : `${v}%`; }
function fmtPitch(v) { return v >= 0 ? `+${v}Hz` : `${v}Hz`; }
function escHtml(s)  {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}
