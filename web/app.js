// 저지연 음성 채팅 프론트 — Gradio 없이 브라우저가 백엔드(/ws/voice-chat)에 직접 연결한다.
// VAD 설정·encodeWAV·디바운스 로직은 render/app.py의 HANDSFREE_INIT_JS를 기반으로
// WebSocket 직결 구조에 맞게 포팅했다(base64 인코딩 + hidden textbox 폴링 단계 제거).

const statusEl = document.getElementById('status');
const chatEl = document.getElementById('chat');
const nicknameInput = document.getElementById('nickname');
const startScreen = document.getElementById('start-screen');
const chatScreen = document.getElementById('chat-screen');
const startBtn = document.getElementById('start-btn');
const micBtn = document.getElementById('mic-btn');
const micLabel = document.getElementById('mic-label');
const textInput = document.getElementById('text-input');
const sendBtn = document.getElementById('send-btn');
const typingIndicator = document.getElementById('typing-indicator');
const voiceModeBtn = document.getElementById('voice-mode-btn');
const voiceCloseBtn = document.getElementById('voice-close-btn');
const voiceOverlay = document.getElementById('voice-overlay');

// 탭(대화/프로필/관리자)
const tabButtons = document.querySelectorAll('.tab-btn');
const tabPanels = document.querySelectorAll('.tab-panel');

// 프로필 탭
const profileNameInput = document.getElementById('profile-name');
const profileAgeInput = document.getElementById('profile-age');
const profileConsentInput = document.getElementById('profile-consent');
const profileConditionsInput = document.getElementById('profile-conditions');
const profileEmergencyInput = document.getElementById('profile-emergency');
const profileNotesInput = document.getElementById('profile-notes');
const profileSaveBtn = document.getElementById('profile-save-btn');
const profileStatusEl = document.getElementById('profile-status');

// 관리자 탭
const adminPwInput = document.getElementById('admin-pw');
const adminLoginBtn = document.getElementById('admin-login-btn');
const adminStatusEl = document.getElementById('admin-status');
const adminPanel = document.getElementById('admin-panel');
const adminViewBtn = document.getElementById('admin-view-btn');
const adminCsvBtn = document.getElementById('admin-csv-btn');
const adminTableWrap = document.getElementById('admin-table-wrap');
let adminPassword = ''; // 로그인 성공 시 보관해 매 요청마다 재전송(세션/쿠키 없음)

let ws = null;
let nickname = '';
let vad = null;
let vadLoaded = false;
let active = false;
let ignoreVAD = false;
let lastSpeechEndAt = 0;
let longSpeechTimer = null;
let endRequested = false;

// 문장 단위로 도착하는 binary WAV를 순서대로 재생하는 큐.
const audioQueue = [];
let isPlaying = false;

// 같은 AI 턴의 문장들을 풍선 하나에 이어붙이기 위한 참조.
// 사용자 발화(새 턴)가 들어오면 null로 리셋해 다음 AI 문장은 새 풍선부터 시작한다.
let currentAiBubble = null;

// ── 음성 오브 실시간 진폭 — ChatGPT/Gemini 음성모드 느낌의 원형 비주얼.
// VAD 내부 스트림을 건드리지 않고, 시각화 전용으로 별도 getUserMedia 스트림을
// 하나 더 열어 분석한다(VAD 로직과 완전히 분리해 회귀 위험을 낮춤).
let audioCtx = null;
let micStream = null; // VAD가 getStream으로 얻은 마이크 스트림 — 오브 분석도 이걸 재사용한다.
let micAnalyser = null;
let micAnalyserData = null;
let playbackAnalyser = null;
let playbackAnalyserData = null;
let orbRAF = null;

async function ensureAudioContext() {
  if (!audioCtx) {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    audioCtx = new Ctx();
  }
  if (audioCtx.state === 'suspended') {
    try { await audioCtx.resume(); } catch (_) {}
  }
  return audioCtx;
}

// getUserMedia를 VAD와 오브 시각화가 각자 따로 호출하면(특히 모바일 Safari에서)
// 동시 마이크 스트림 두 개가 충돌해 음성 인식이 불안정해지는 문제가 있었다.
// 그래서 VAD 쪽 옵션(getStream)에서 받아온 스트림을 여기서 그대로 재사용한다.
async function setupMicAnalyser() {
  if (micAnalyser || !micStream) return;
  try {
    const ctx = await ensureAudioContext();
    const source = ctx.createMediaStreamSource(micStream);
    micAnalyser = ctx.createAnalyser();
    micAnalyser.fftSize = 256;
    micAnalyserData = new Uint8Array(micAnalyser.frequencyBinCount);
    source.connect(micAnalyser);
  } catch (e) {
    console.warn('[orb] 마이크 진폭 분석 설정 실패(오브는 애니메이션만 동작)', e);
  }
}

function attachPlaybackAnalyser(audioEl) {
  if (!audioCtx) return;
  try {
    const source = audioCtx.createMediaElementSource(audioEl);
    playbackAnalyser = audioCtx.createAnalyser();
    playbackAnalyser.fftSize = 256;
    playbackAnalyserData = new Uint8Array(playbackAnalyser.frequencyBinCount);
    source.connect(playbackAnalyser);
    playbackAnalyser.connect(audioCtx.destination); // 분석 후 실제 출력으로 이어줘야 소리가 남
  } catch (e) {
    console.warn('[orb] 재생 진폭 분석 연결 실패(소리는 정상 재생됨)', e);
  }
}

function getRmsLevel(analyser, data) {
  analyser.getByteTimeDomainData(data);
  let sumSquares = 0;
  for (let i = 0; i < data.length; i++) {
    const v = (data[i] - 128) / 128;
    sumSquares += v * v;
  }
  return Math.sqrt(sumSquares / data.length);
}

function startOrbLoop() {
  if (orbRAF) return;
  const tick = () => {
    let level = 0;
    if (micBtn.classList.contains('speaking') && playbackAnalyser) {
      level = getRmsLevel(playbackAnalyser, playbackAnalyserData);
    } else if (micBtn.classList.contains('active') && micAnalyser) {
      level = getRmsLevel(micAnalyser, micAnalyserData);
    }
    micBtn.style.setProperty('--level', Math.min(1, level * 4).toFixed(3));
    orbRAF = requestAnimationFrame(tick);
  };
  orbRAF = requestAnimationFrame(tick);
}

function stopOrbLoopIfIdle() {
  if (micBtn.classList.contains('active') || micBtn.classList.contains('speaking')) return;
  if (orbRAF) { cancelAnimationFrame(orbRAF); orbRAF = null; }
  micBtn.style.setProperty('--level', 0);
}

function setStatus(msg) {
  statusEl.textContent = msg;
  micLabel.textContent = msg; // 음성 모드 오버레이의 캡션도 같은 상태 문구를 보여줌
}

function showTyping() {
  typingIndicator.style.display = 'flex';
  chatEl.appendChild(typingIndicator); // 항상 맨 아래로 이동
  chatEl.scrollTop = chatEl.scrollHeight;
}

function hideTyping() {
  typingIndicator.style.display = 'none';
}

function appendBubble(role, text) {
  hideTyping();
  if (role === 'ai' && currentAiBubble) {
    currentAiBubble.textContent += ' ' + text;
    chatEl.scrollTop = chatEl.scrollHeight;
    return;
  }
  const div = document.createElement('div');
  div.className = 'bubble ' + role + ' fade-in';
  div.textContent = text;
  chatEl.insertBefore(div, typingIndicator);
  chatEl.scrollTop = chatEl.scrollHeight;
  currentAiBubble = role === 'ai' ? div : null;
}

function loadScript(src) {
  return new Promise((resolve, reject) => {
    if (document.querySelector('script[src="' + src + '"]')) { resolve(); return; }
    const s = document.createElement('script');
    s.src = src; s.onload = resolve; s.onerror = reject;
    document.head.appendChild(s);
  });
}

function encodeWAV(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const w = (off, str) => { for (let i = 0; i < str.length; i++) view.setUint8(off + i, str.charCodeAt(i)); };
  w(0, 'RIFF'); view.setUint32(4, 36 + samples.length * 2, true); w(8, 'WAVE');
  w(12, 'fmt '); view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true); view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  w(36, 'data'); view.setUint32(40, samples.length * 2, true);
  let off = 44;
  for (let i = 0; i < samples.length; i++, off += 2) {
    let s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }
  return buffer; // ArrayBuffer — WS binary 프레임으로 그대로 전송 (base64 불필요)
}

function connectWS() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws/voice-chat?nickname=${encodeURIComponent(nickname)}`);
  ws.binaryType = 'arraybuffer';

  ws.onopen = () => setStatus('🔌 연결됨 — 마이크 버튼을 눌러 시작하세요');
  ws.onclose = () => setStatus('🔌 연결 끊김 — 새로고침 해주세요');
  ws.onerror = () => setStatus('❌ 연결 오류');

  ws.onmessage = (event) => {
    if (event.data instanceof ArrayBuffer) {
      enqueueAudio(event.data);
      return;
    }
    let msg;
    try {
      msg = JSON.parse(event.data);
    } catch (e) {
      console.warn('[ws] JSON 파싱 실패', e);
      return;
    }
    if (msg.type === 'transcript') {
      appendBubble('user', msg.text);
      showTyping();
    } else if (msg.type === 'text_chunk') {
      appendBubble('ai', msg.text);
    } else if (msg.type === 'done') {
      currentAiBubble = null; // 턴 종료 — 다음 AI 문장은 새 풍선부터
      hideTyping();
      if (msg.conversation_ended) {
        endRequested = true;
      }
    } else if (msg.type === 'error') {
      hideTyping();
      setStatus('❌ ' + msg.detail);
      console.error('[ws] error', msg.detail);
    }
  };
}

function enqueueAudio(buf) {
  audioQueue.push(buf);
  playNextInQueue();
}

async function playNextInQueue() {
  if (isPlaying || audioQueue.length === 0) return;
  isPlaying = true;
  const buf = audioQueue.shift();
  const blob = new Blob([buf], { type: 'audio/wav' });
  const url = URL.createObjectURL(blob);
  const audio = new Audio(url);
  if (vad) { try { vad.pause(); } catch (_) {} }
  setStatus('🔊 답변 재생 중… (마이크 일시정지)');
  micBtn.classList.add('speaking');

  // 모바일에서 AudioContext가 다시 suspended로 빠지면, 분석용으로 그래프에
  // 연결된 <audio>가 무음이 되는 문제가 있었다(모바일에서 음성이 안 들리는
  // 버그, 2026-06). 재생마다 미리 resume을 시도하고, 그래도 running이 아니면
  // 분석 연결 자체를 건너뛰어 일반 재생(소리는 반드시 남)으로 폴백한다.
  await ensureAudioContext();
  if (audioCtx && audioCtx.state === 'running') {
    attachPlaybackAnalyser(audio);
    startOrbLoop();
  } else {
    console.warn('[orb] AudioContext 상태가 running이 아님(' + (audioCtx && audioCtx.state) + ') — 이번 클립은 분석 없이 일반 재생');
  }

  const cleanup = () => {
    URL.revokeObjectURL(url);
    isPlaying = false;
    if (audioQueue.length > 0) {
      playNextInQueue();
    } else {
      resumeAfterPlayback();
    }
  };
  audio.addEventListener('ended', cleanup);
  audio.addEventListener('error', cleanup);
  const p = audio.play();
  if (p && p.catch) {
    p.catch((err) => {
      console.warn('[audio] play() 차단:', err && err.message);
      cleanup();
    });
  }
}

// 답변 재생(큐 전체)이 끝나면 마이크를 다시 켠다. 1.5s 지연은 스피커 잔향이
// 마이크로 들어와 self-trigger 되는 것을 막기 위함(render/app.py와 동일 패턴).
function resumeAfterPlayback() {
  micBtn.classList.remove('speaking');
  if (endRequested) {
    endRequested = false;
    active = false;
    if (vad) { try { vad.pause(); } catch (_) {} }
    setStatus('✅ 대화가 종료되었습니다.');
    micBtn.classList.remove('active');
    stopOrbLoopIfIdle();
    ignoreVAD = false;
    setTimeout(closeVoiceOverlay, 1800); // 마지막 인사를 잠깐 보여준 뒤 자동으로 닫힘
    return;
  }
  if (!active) {
    setStatus('⏸️ 중지됨 (버튼을 다시 누르면 시작)');
    stopOrbLoopIfIdle();
    ignoreVAD = false;
    return;
  }
  setStatus('🎧 듣는 중… (마이크 안정화)');
  setTimeout(() => {
    if (active && vad) {
      try { vad.start(); } catch (e) {}
      setStatus('🎧 듣는 중…');
    }
    ignoreVAD = false;
  }, 1500);
}

async function initVAD() {
  if (!vadLoaded) {
    await loadScript('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.14.0/dist/ort.js');
    await loadScript('https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.22/dist/bundle.min.js');
    vadLoaded = true;
  }
  if (!vad) {
    vad = await window.vad.MicVAD.new({
      // getUserMedia를 VAD 안에서 자체적으로 부르게 두면(기본 동작), 오브 진폭
      // 분석용으로 따로 또 getUserMedia를 부를 때 모바일 Safari에서 두 스트림이
      // 충돌해 인식이 불안정해졌다. 여기서 한 번만 얻어 micStream에 저장하고
      // setupMicAnalyser()도 같은 스트림을 재사용한다.
      getStream: async () => {
        micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        return micStream;
      },
      baseAssetPath: 'https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.22/dist/',
      onnxWASMBasePath: 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.14.0/dist/',
      positiveSpeechThreshold: 0.5,
      negativeSpeechThreshold: 0.35,
      minSpeechFrames: 3,
      redemptionFrames: 24,
      onSpeechStart: () => {
        setStatus('🗣️ 말하는 중…');
        if (longSpeechTimer) clearTimeout(longSpeechTimer);
        longSpeechTimer = setTimeout(() => {
          setStatus('🗣️ 말하는 중… (문장을 마치고 잠깐 쉬시면 더 잘 알아들어요)');
        }, 25000);
      },
      onVADMisfire: () => {
        setStatus('🎧 듣는 중… (조금 더 또렷이 말씀해 주세요)');
        if (longSpeechTimer) { clearTimeout(longSpeechTimer); longSpeechTimer = null; }
      },
      onSpeechEnd: (audio) => {
        if (longSpeechTimer) { clearTimeout(longSpeechTimer); longSpeechTimer = null; }
        if (ignoreVAD) return;
        const now = Date.now();
        if (now - lastSpeechEndAt < 3000) return; // 3초 내 재진입 무시
        lastSpeechEndAt = now;
        ignoreVAD = true;
        if (vad) { try { vad.pause(); } catch (e) {} }
        setStatus('⏳ 처리 중…');
        const wavBuf = encodeWAV(audio, 16000);
        if (ws && ws.readyState === WebSocket.OPEN) {
          ws.send(wavBuf);
        } else {
          setStatus('❌ 서버에 연결되어 있지 않아요');
          ignoreVAD = false;
        }
      },
    });
  }
}

// 채팅 입력줄의 파형 버튼으로 들어오는, ChatGPT/Gemini 스타일 전체화면 음성 모드.
function openVoiceOverlay() {
  voiceOverlay.classList.add('open');
}
function closeVoiceOverlay() {
  voiceOverlay.classList.remove('open');
}

async function enterVoiceMode() {
  openVoiceOverlay();
  setStatus('🎤 마이크 준비 중…');
  try {
    await ensureAudioContext(); // 사용자 클릭(제스처) 시점에 생성/resume — 이후 TTS 재생도 이 컨텍스트를 씀
    await initVAD();
    // getStream은 new() 시점이 아니라 start()가 실제로 마이크를 잡을 때 호출되는
    // 듯하다(이 순서가 아니면 micStream이 아직 비어 있어 듣는중 오브가 진폭에
    // 반응하지 않는 버그가 있었음) — start() 이후에 호출해야 micStream이 채워짐.
    await vad.start();
    await setupMicAnalyser(); // 이제 채워진 micStream을 재사용(별도 getUserMedia 없음)
    active = true;
    setStatus('🎧 듣는 중… (말씀하시면 자동 인식)');
    micBtn.classList.add('active');
    startOrbLoop();
  } catch (e) {
    console.error('mic/VAD init error', e);
    setStatus('❌ 마이크/VAD 초기화 실패 — 브라우저 콘솔을 확인하세요');
  }
}

function exitVoiceMode() {
  if (vad) { try { vad.pause(); } catch (_) {} }
  active = false;
  micBtn.classList.remove('active');
  stopOrbLoopIfIdle();
  setStatus('대기 중');
  closeVoiceOverlay();
}

function sendText() {
  const text = textInput.value.trim();
  if (!text || !ws || ws.readyState !== WebSocket.OPEN) return;
  appendBubble('user', text);
  showTyping();
  ws.send(JSON.stringify({ type: 'text', text }));
  textInput.value = '';
}

// ── 탭 전환 ──
function switchTab(name) {
  tabButtons.forEach((b) => b.classList.toggle('active', b.dataset.tab === name));
  tabPanels.forEach((p) => p.classList.toggle('active', p.id === `tab-${name}`));
  if (name === 'profile') loadProfile();
}

// ── 프로필 탭 ──
// 동의 체크가 꺼져있으면 질환·특이사항 입력란을 잠근다(기존 Gradio toggle_health_fields와 동일).
function applyConsentGate() {
  const enabled = profileConsentInput.checked;
  profileConditionsInput.disabled = !enabled;
  profileNotesInput.disabled = !enabled;
}

async function loadProfile() {
  try {
    const resp = await fetch(`/profile/${encodeURIComponent(nickname)}`);
    if (resp.status === 404) { applyConsentGate(); return; } // 신규 사용자 — 빈 폼 유지
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    const data = await resp.json();
    const p = data.profile || {};
    profileNameInput.value = p.name || '';
    profileAgeInput.value = p.age || '';
    profileConsentInput.checked = !!p.health_info_consent;
    profileConditionsInput.value = p.conditions || '';
    profileEmergencyInput.value = p.emergency_contact || '';
    profileNotesInput.value = p.notes || '';
    applyConsentGate();
  } catch (e) {
    console.warn('[profile] 불러오기 실패', e);
  }
}

async function saveProfile() {
  const consent = profileConsentInput.checked;
  const body = {
    nickname,
    name: profileNameInput.value.trim() || null,
    age: profileAgeInput.value ? parseInt(profileAgeInput.value, 10) : null,
    emergency_contact: profileEmergencyInput.value.trim() || null,
    health_info_consent: consent,
    // 동의하지 않으면 건강정보(질환/특이사항)는 보내지 않음 — 기존 Gradio와 동일 동작.
    conditions: consent ? (profileConditionsInput.value.trim() || null) : null,
    notes: consent ? (profileNotesInput.value.trim() || null) : null,
  };
  try {
    const resp = await fetch('/profile', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    profileStatusEl.textContent = `✅ ${nickname}님의 프로필이 저장되었습니다.`;
  } catch (e) {
    profileStatusEl.textContent = '❌ 저장 실패: ' + e.message;
  }
}

// ── 관리자 탭 ──
function getAdminSource() {
  const checked = document.querySelector('input[name="admin-source"]:checked');
  return checked ? checked.value : 'history';
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
  ));
}

function renderAdminTable(columns, rows) {
  if (!columns || columns.length === 0) {
    adminTableWrap.innerHTML = '<p>표시할 대화 기록이 없습니다.</p>';
    return;
  }
  let html = '<table class="admin-table"><thead><tr>';
  for (const c of columns) html += `<th>${escapeHtml(c)}</th>`;
  html += '</tr></thead><tbody>';
  for (const row of rows) {
    html += '<tr>';
    for (const cell of row) html += `<td>${escapeHtml(cell == null ? '' : cell)}</td>`;
    html += '</tr>';
  }
  html += '</tbody></table>';
  adminTableWrap.innerHTML = html;
}

async function adminLogin() {
  const pw = adminPwInput.value.trim();
  if (!pw) {
    setAdminStatus('비밀번호를 입력해주세요.');
    return;
  }
  try {
    const resp = await fetch('/admin/login', { method: 'POST', headers: { 'X-Admin-Password': pw } });
    if (resp.status === 200) {
      adminPassword = pw;
      adminPanel.style.display = 'block';
      setAdminStatus('✅ 로그인 성공. 아래에서 대화 기록을 조회/다운로드하세요.');
    } else if (resp.status === 401) {
      setAdminStatus('❌ 비밀번호가 올바르지 않습니다.');
    } else {
      setAdminStatus(`❌ 로그인 실패 (status ${resp.status})`);
    }
  } catch (e) {
    setAdminStatus('❌ 서버 연결 실패: ' + e.message);
  }
}

function setAdminStatus(msg) {
  adminStatusEl.textContent = msg;
}

async function adminViewTable() {
  if (!adminPassword) { setAdminStatus('먼저 로그인해주세요.'); return; }
  const source = getAdminSource();
  try {
    const resp = await fetch(`/admin/conversations.json?source=${source}`, {
      headers: { 'X-Admin-Password': adminPassword },
    });
    if (resp.status === 401) { setAdminStatus('❌ 인증이 만료되었습니다. 다시 로그인해주세요.'); return; }
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    const data = await resp.json();
    renderAdminTable(data.columns, data.rows);
    const note = data.shown < data.total
      ? `총 ${data.total}행 중 최근 ${data.shown}행 표시`
      : `총 ${data.total}행 표시`;
    setAdminStatus(data.total === 0 ? '표시할 대화 기록이 없습니다.' : '✅ ' + note);
  } catch (e) {
    setAdminStatus('❌ 조회 실패: ' + e.message);
  }
}

async function adminDownloadCsv() {
  if (!adminPassword) { setAdminStatus('먼저 로그인해주세요.'); return; }
  const source = getAdminSource();
  try {
    const resp = await fetch(`/admin/conversations.csv?source=${source}`, {
      headers: { 'X-Admin-Password': adminPassword },
    });
    if (resp.status === 401) { setAdminStatus('❌ 인증이 만료되었습니다. 다시 로그인해주세요.'); return; }
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    const blob = await resp.blob();
    const cd = resp.headers.get('content-disposition') || '';
    const m = /filename="?([^"]+)"?/.exec(cd);
    const filename = m ? m[1] : `conversations_${source}.csv`;
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = filename;
    document.body.appendChild(a); a.click(); document.body.removeChild(a);
    URL.revokeObjectURL(url);
    setAdminStatus(`✅ ${filename} 다운로드 완료`);
  } catch (e) {
    setAdminStatus('❌ 다운로드 실패: ' + e.message);
  }
}

function startChat() {
  nickname = nicknameInput.value.trim();
  if (!nickname) {
    alert('닉네임을 입력해 주세요.');
    return;
  }
  startScreen.style.display = 'none';
  chatScreen.style.display = 'flex';
  // 사용자 클릭(제스처) 시점에 AudioContext를 미리 만들어둬야, 마이크를 한 번도
  // 안 켜고 텍스트로만 대화해도 TTS 재생 시 오브 진폭 분석이 정상 동작한다.
  ensureAudioContext();
  connectWS();
}

startBtn.addEventListener('click', startChat);
nicknameInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') startChat();
});

voiceModeBtn.addEventListener('click', enterVoiceMode);
voiceCloseBtn.addEventListener('click', exitVoiceMode);
sendBtn.addEventListener('click', sendText);
textInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') sendText();
});

tabButtons.forEach((b) => b.addEventListener('click', () => switchTab(b.dataset.tab)));
profileConsentInput.addEventListener('change', applyConsentGate);
profileSaveBtn.addEventListener('click', saveProfile);
adminLoginBtn.addEventListener('click', adminLogin);
adminViewBtn.addEventListener('click', adminViewTable);
adminCsvBtn.addEventListener('click', adminDownloadCsv);
