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
const voicePanel = document.getElementById('voice-panel');
const textInputRow = document.getElementById('text-input-row');

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

// ── iOS 자동재생 잠금 해제 ──────────────────────────────────────────────
// iOS Safari는 사용자 제스처 없이는 audio.play()를 막는다. TTS는 WebSocket으로
// 비동기 도착하므로 제스처와 분리돼 차단된다(맥북 데스크톱은 제한이 약해 잘 됨).
// 해결: <audio> 엘리먼트 하나를 만들어 두고, 사용자 제스처(시작/음성 버튼 탭)의
// '동기' 구간에서 무음 클립을 한 번 재생해 잠금을 풀어둔다. 이후 같은 엘리먼트의
// src만 바꿔 재생하면 iOS가 허용한다. 매번 new Audio()를 만들면 잠금이 안 풀린다.
const SILENT_WAV = 'data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEAgD4AAAB9AAACABAAZGF0YQAAAAA=';
let ttsAudio = null;
function getTtsAudio() {
  if (!ttsAudio) {
    ttsAudio = new Audio();
    ttsAudio.setAttribute('playsinline', '');
    ttsAudio.preload = 'auto';
  }
  return ttsAudio;
}
// 반드시 제스처 핸들러의 '동기' 구간에서(await 이전에) 호출해야 iOS가 잠금을 푼다.
function unlockAudio() {
  const a = getTtsAudio();
  try {
    a.src = SILENT_WAV;
    const p = a.play();
    if (p && p.catch) p.catch(() => {});
  } catch (_) {}
}

// 같은 AI 턴의 문장들을 풍선 하나에 이어붙이기 위한 참조.
// 사용자 발화(새 턴)가 들어오면 null로 리셋해 다음 AI 문장은 새 풍선부터 시작한다.
let currentAiBubble = null;

// ── 음성 오브 실시간 진폭 — ChatGPT/Gemini 음성모드 느낌의 원형 비주얼.
// 핵심 설계 원칙(2026-06 모바일 Safari 디버깅 후 재설계):
//  1) 소리 경로와 시각화 경로를 완전히 분리한다. 소리는 항상 <audio>로 직접 재생.
//  2) 마이크 진폭은 Web Audio AnalyserNode(모바일 Safari에서 안 갱신되는 버그)
//     대신, VAD가 이미 처리 중인 오디오 프레임(onFrameProcessed)에서 직접 RMS를
//     뽑는다 — 가장 확실하게 동작하는 경로.
let audioCtx = null;
let orbRAF = null;
let orbLevel = 0;       // 현재 진폭(0~1) — CSS 폴백·WebGL 오브가 함께 읽는다.
let micFrameRms = 0;    // VAD onFrameProcessed가 갱신하는 마이크 진폭
let micFrameAt = 0;     // 마지막 프레임 수신 시각(ms) — 신선도 판단용
let speakingEnv = null; // 현재 재생 중 TTS의 RMS 엔벨로프 {rms, hop, max}

// WAV(PCM 16-bit) 바이트에서 RMS 엔벨로프를 직접 계산한다. Web Audio API를
// 일절 쓰지 않으므로(파싱은 순수 산술) iOS 무음 버그와 무관하다. 재생 중
// audio.currentTime으로 이 배열을 샘플링해 '말하는중' 오브를 실제 음압대로
// 움직이게 한다. 지원 못 하는 포맷이면 null을 돌려주고 합성 펄스로 폴백.
function parseWavEnvelope(buf) {
  try {
    const dv = new DataView(buf);
    if (dv.getUint32(0, false) !== 0x52494646) return null; // 'RIFF'
    if (dv.getUint32(8, false) !== 0x57415645) return null; // 'WAVE'
    let off = 12, fmt = null, dataOff = -1, dataLen = 0;
    while (off + 8 <= dv.byteLength) {
      const id = dv.getUint32(off, false);
      const sz = dv.getUint32(off + 4, true);
      if (id === 0x666d7420) {        // 'fmt '
        fmt = {
          format: dv.getUint16(off + 8, true),
          channels: dv.getUint16(off + 10, true),
          rate: dv.getUint32(off + 12, true),
          bits: dv.getUint16(off + 22, true),
        };
      } else if (id === 0x64617461) { // 'data'
        dataOff = off + 8; dataLen = sz; break;
      }
      off += 8 + sz + (sz & 1);
    }
    if (!fmt || dataOff < 0 || fmt.format !== 1 || fmt.bits !== 16) return null;
    const ch = fmt.channels || 1;
    const total = Math.floor(dataLen / 2);            // 16bit 샘플 수(채널 포함)
    const frames = Math.floor(total / ch);
    const hopSec = 0.04;                              // 40ms 창
    const hop = Math.max(1, Math.floor(fmt.rate * hopSec));
    const nWin = Math.max(1, Math.ceil(frames / hop));
    const rms = new Float32Array(nWin);
    let max = 1e-6;
    for (let w = 0; w < nWin; w++) {
      let sum = 0, n = 0;
      const startF = w * hop, endF = Math.min(frames, startF + hop);
      for (let f = startF; f < endF; f++) {
        const s = dv.getInt16(dataOff + (f * ch) * 2, true) / 32768;
        sum += s * s; n++;
      }
      const v = n ? Math.sqrt(sum / n) : 0;
      rms[w] = v;
      if (v > max) max = v;
    }
    return { rms, hop: hopSec, max };
  } catch (_) {
    return null;
  }
}

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

function startOrbLoop() {
  if (orbRAF) return;
  const tick = () => {
    let target = 0;
    if (micBtn.classList.contains('speaking')) {
      // 말하는중: 재생 중 TTS의 '실제' RMS 엔벨로프를 audio.currentTime으로
      // 샘플링해 음압대로 출렁이게 한다(Web Audio 미사용 → iOS 무음 버그 무관).
      if (speakingEnv && ttsAudio && !ttsAudio.paused) {
        const idx = Math.floor(ttsAudio.currentTime / speakingEnv.hop);
        const v = speakingEnv.rms[Math.min(idx, speakingEnv.rms.length - 1)] || 0;
        target = 0.12 + 0.88 * Math.min(1, v / speakingEnv.max);
      } else {
        // 청크 사이 짧은 공백 — 엔벨로프 없으면 약한 합성 출렁임으로 메운다.
        const tt = performance.now() / 1000;
        const e = 0.5 * Math.sin(tt * 6.3) + 0.3 * Math.sin(tt * 11.7 + 1.3) +
                  0.2 * Math.sin(tt * 19.1 + 2.7);
        target = 0.16 + 0.3 * Math.abs(e);
      }
    } else if (micBtn.classList.contains('active')) {
      // 듣는중: VAD 프레임 RMS가 최근(250ms 내) 들어왔으면 그걸 쓰고, 아니면 0.
      // 위에서 오브 자체의 기본 일렁임(셰이더/CSS)이 항상 돌고 있으므로 0이어도
      // 죽은 듯 보이지 않는다.
      const level = (performance.now() - micFrameAt < 250) ? micFrameRms * 6 : 0;
      target = Math.min(1, level * 4);
    }
    // 살짝 부드럽게(감쇠) — 갑작스런 튐 방지.
    orbLevel += (target - orbLevel) * 0.35;
    micBtn.style.setProperty('--level', orbLevel.toFixed(3)); // CSS 폴백 오브용
    window.__orbLevel = orbLevel;                              // WebGL 오브(orb.js)용
    orbRAF = requestAnimationFrame(tick);
  };
  orbRAF = requestAnimationFrame(tick);
}

function stopOrbLoopIfIdle() {
  if (micBtn.classList.contains('active') || micBtn.classList.contains('speaking')) return;
  if (orbRAF) { cancelAnimationFrame(orbRAF); orbRAF = null; }
  orbLevel = 0;
  micBtn.style.setProperty('--level', 0);
}

function setStatus(msg) {
  statusEl.textContent = msg;
  // mic-label은 더 이상 갱신하지 않음 — 패널이 인라인으로 바뀌면서 상단 상태바와
  // 동시에 보여 같은 문구가 두 번 뜨는 문제가 있었다(2026-06). CSS로 숨김.
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

// <audio> 재생은 Web Audio API(AudioContext/AnalyserNode/decodeAudioData)와
// 절대 엮지 않는다 — iOS에서 오디오 리소스를 다퉈 TTS가 통째로 무음이 되는
// 버그가 있었다(2026-06). 오브의 '말하는중' 애니메이션은 합성 펄스로 대체.
// 또, unlockAudio()로 제스처 때 풀어둔 '동일' 엘리먼트를 재사용해 iOS 자동재생
// 차단을 피한다(매번 new Audio()면 차단됨).
function playNextInQueue() {
  if (isPlaying || audioQueue.length === 0) return;
  isPlaying = true;
  const buf = audioQueue.shift();
  speakingEnv = parseWavEnvelope(buf); // 실제 음압 엔벨로프(Web Audio 안 씀)
  const blob = new Blob([buf], { type: 'audio/wav' });
  const url = URL.createObjectURL(blob);
  const audio = getTtsAudio();
  if (vad) { try { vad.pause(); } catch (_) {} }
  setStatus('🔊 답변 재생 중…');
  micBtn.classList.add('speaking');
  startOrbLoop();

  const cleanup = (reason) => {
    if (reason) console.log('[audio] cleanup: ' + reason);
    URL.revokeObjectURL(url);
    speakingEnv = null;
    audio.onended = null; audio.onerror = null; // 재사용 엘리먼트라 리스너를 매번 정리
    isPlaying = false;
    if (audioQueue.length > 0) {
      playNextInQueue();
    } else {
      resumeAfterPlayback();
    }
  };
  audio.onended = () => cleanup('ended');
  audio.onerror = () => {
    const codes = { 1: 'ABORTED', 2: 'NETWORK', 3: 'DECODE', 4: 'SRC_NOT_SUPPORTED' };
    const code = audio.error && audio.error.code;
    const detail = code ? (codes[code] || code) : '알수없음';
    setStatus('🔇 음성 재생 오류(' + detail + ')');
    cleanup('error:' + detail);
  };
  audio.src = url;
  const p = audio.play();
  if (p && p.catch) {
    p.catch((err) => {
      setStatus('🔇 음성 재생 차단(' + (err && err.name) + ')');
      cleanup('play-rejected');
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
      baseAssetPath: 'https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.22/dist/',
      onnxWASMBasePath: 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.14.0/dist/',
      positiveSpeechThreshold: 0.5,
      negativeSpeechThreshold: 0.35,
      minSpeechFrames: 3,
      redemptionFrames: 24,
      // VAD가 이미 처리 중인 오디오 프레임에서 직접 RMS를 뽑아 오브 진폭으로 쓴다.
      // (Web Audio AnalyserNode가 모바일 Safari에서 안 갱신되는 문제를 우회)
      // 라이브러리 버전에 따라 frame 인자가 없을 수 있어 방어적으로 처리.
      onFrameProcessed: (_probs, frame) => {
        if (!frame || !frame.length) return;
        let sum = 0;
        for (let i = 0; i < frame.length; i++) sum += frame[i] * frame[i];
        micFrameRms = Math.sqrt(sum / frame.length);
        micFrameAt = performance.now();
      },
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

// 채팅 입력줄의 파형 버튼으로 들어오는 음성 모드. 전체화면으로 가리지 않고
// 입력줄 자리에 오브만 인라인으로 나타난다 — 채팅 내용(말풍선)은 항상 그대로
// 보여 글로 읽는 사용자(청각장애인 등)도 대화를 따라갈 수 있게 한다.
function openVoiceOverlay() {
  textInputRow.style.display = 'none';
  voicePanel.classList.add('open');
}
function closeVoiceOverlay() {
  voicePanel.classList.remove('open');
  textInputRow.style.display = 'flex';
}

// 음성 모드 시작 효과음 — 짧은 상승 톤(오디오 파일 없이 오실레이터로 생성).
function playStartChime() {
  if (!audioCtx) return;
  try {
    const o = audioCtx.createOscillator();
    const g = audioCtx.createGain();
    const t = audioCtx.currentTime;
    o.type = 'sine';
    o.frequency.setValueAtTime(620, t);
    o.frequency.exponentialRampToValueAtTime(960, t + 0.13);
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(0.13, t + 0.03);
    g.gain.exponentialRampToValueAtTime(0.0001, t + 0.22);
    o.connect(g); g.connect(audioCtx.destination);
    o.start(t); o.stop(t + 0.24);
  } catch (_) {}
}

// 햅틱(진동) — Android Chrome은 navigator.vibrate 지원, iOS Safari는 미지원이라
// 아이폰에선 조용히 무시된다(웹에서 iOS 진동은 불가).
function haptic(pattern) {
  try { if (navigator.vibrate) navigator.vibrate(pattern); } catch (_) {}
}

async function enterVoiceMode() {
  unlockAudio(); // ★ 제스처 동기 구간에서 iOS 오디오 잠금 해제(반드시 await 이전)
  openVoiceOverlay();
  setStatus('🎤 마이크 준비 중…');
  try {
    await ensureAudioContext(); // 사용자 클릭(제스처) 시점에 생성/resume
    playStartChime();
    haptic(30);
    await initVAD();
    await vad.start(); // 진폭은 initVAD의 onFrameProcessed에서 직접 뽑는다(Web Audio 불필요)
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
  // 시작 버튼 탭(제스처)에서 iOS 오디오 잠금을 미리 풀어둔다 — 텍스트로만 대화해도
  // TTS가 비동기로 도착하므로, 여기서 안 풀면 모바일에서 소리가 차단된다.
  unlockAudio();
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
