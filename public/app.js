/* Real-Time Voice AI Agent — browser client.
 *
 * Flow per turn:
 *   mic -> MediaRecorder -> /api/transcribe -> /api/chat -> /api/speak -> playback
 *
 * The Web Audio analyser drives both the waveform and the silence detector
 * that ends your turn automatically, so a hands-free conversation never needs
 * a second click.
 */

(() => {
  'use strict';

  const MAX_RECORD_MS = 30000;     // hard stop; also keeps uploads small
  const MIN_RECORD_MS = 350;       // ignore accidental taps
  const MAX_UPLOAD_BYTES = 4_000_000;
  const HISTORY_LIMIT = 24;

  // ---------------------------------------------------------------- state

  const state = {
    ready: false,
    caps: { asr: false, llm: false, tts: false },
    phase: 'idle',                 // idle | recording | transcribing | thinking | speaking
    history: [],                   // API-shaped messages
    turns: 0,
    stream: null,
    audioCtx: null,
    analyser: null,
    micSource: null,
    playerSource: null,
    recorder: null,
    chunks: [],
    recordStart: 0,
    rafId: null,
    level: 0,
    noiseFloor: 0.008,
    speechSeen: false,
    silenceSince: 0,
    autoStopTimer: null,
    player: new Audio(),
    lastObjectUrl: null,
    clips: new Map(),              // message id -> Blob for replay
    settings: load('vagent.settings', {
      theme: 'dark',
      voiceId: '',
      speed: 1,
      silence: 1.2,
      systemPrompt: '',
      useTools: true,
      autoPlay: true,
      browserVoice: false,
      handsFree: false,
    }),
  };

  // ------------------------------------------------------------------ dom

  const $ = (id) => document.getElementById(id);
  const el = {
    transcript: $('transcript'),
    empty: $('emptyState'),
    turnCount: $('turnCount'),
    micBtn: $('micBtn'),
    micRing: $('micRing'),
    micHint: $('micHint'),
    scope: $('scope'),
    stateLabel: $('stateLabel'),
    timer: $('timer'),
    pipeline: $('pipeline'),
    composer: $('composer'),
    textInput: $('textInput'),
    sendBtn: $('sendBtn'),
    clearBtn: $('clearBtn'),
    exportBtn: $('exportBtn'),
    stopAudioBtn: $('stopAudioBtn'),
    handsFree: $('handsFree'),
    themeToggle: $('themeToggle'),
    settingsToggle: $('settingsToggle'),
    drawer: $('drawer'),
    drawerClose: $('drawerClose'),
    scrim: $('scrim'),
    voiceSelect: $('voiceSelect'),
    voiceNote: $('voiceNote'),
    speedRange: $('speedRange'),
    speedValue: $('speedValue'),
    silenceRange: $('silenceRange'),
    silenceValue: $('silenceValue'),
    systemPrompt: $('systemPrompt'),
    useTools: $('useTools'),
    autoPlay: $('autoPlay'),
    browserVoice: $('browserVoice'),
    toolset: $('toolset'),
    buildInfo: $('buildInfo'),
    capabilities: $('capabilities'),
    banner: $('banner'),
    bannerText: $('bannerText'),
    bannerClose: $('bannerClose'),
    toasts: $('toasts'),
  };

  const ctx2d = el.scope.getContext('2d');

  // -------------------------------------------------------------- helpers

  function load(key, fallback) {
    try {
      return { ...fallback, ...JSON.parse(localStorage.getItem(key) || '{}') };
    } catch {
      return { ...fallback };
    }
  }

  function saveSettings() {
    try {
      localStorage.setItem('vagent.settings', JSON.stringify(state.settings));
    } catch { /* private mode — settings just won't persist */ }
  }

  function toast(message, kind = '') {
    const node = document.createElement('div');
    node.className = `toast ${kind}`.trim();
    node.textContent = message;
    el.toasts.appendChild(node);
    setTimeout(() => {
      node.style.opacity = '0';
      node.style.transition = 'opacity .25s';
      setTimeout(() => node.remove(), 260);
    }, 3600);
  }

  function banner(message) {
    el.bannerText.textContent = message;
    el.banner.hidden = false;
  }

  const escapeHtml = (value) =>
    String(value).replace(/[&<>"']/g, (char) => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
    }[char]));

  // ------------------------------------------------------------ pipeline

  function setStage(stage, status, ms) {
    const row = el.pipeline.querySelector(`[data-stage="${stage}"]`);
    if (!row) return;
    row.classList.remove('active', 'done', 'failed');
    if (status) row.classList.add(status);
    row.querySelector('.ms').textContent = ms == null ? '' : `${ms} ms`;
  }

  function resetPipeline() {
    ['listen', 'asr', 'llm', 'tts'].forEach((stage) => setStage(stage, null, null));
  }

  function setPhase(phase) {
    state.phase = phase;
    const labels = {
      idle: 'Idle',
      recording: 'Recording',
      transcribing: 'Transcribing',
      thinking: 'Thinking',
      speaking: 'Speaking',
    };
    el.stateLabel.textContent = labels[phase] || phase;
    el.stateLabel.className = 'state-label ' +
      (phase === 'recording' ? 'rec'
        : phase === 'speaking' ? 'play'
        : phase === 'idle' ? '' : 'busy');

    el.micBtn.classList.toggle('recording', phase === 'recording');
    el.micBtn.classList.toggle('busy', phase === 'transcribing' || phase === 'thinking');
    el.micBtn.disabled = phase === 'transcribing' || phase === 'thinking';
    el.micBtn.setAttribute(
      'aria-label',
      phase === 'recording' ? 'Stop recording' : 'Start recording',
    );
    el.sendBtn.disabled = phase === 'transcribing' || phase === 'thinking';
    el.stopAudioBtn.disabled = phase !== 'speaking';

    el.micHint.innerHTML = phase === 'recording'
      ? 'Release to send · <kbd>Esc</kbd> to cancel'
      : phase === 'idle'
        ? 'Click, or hold <kbd>Space</kbd> to talk'
        : 'Working…';
  }

  // ------------------------------------------------------------ transcript

  function messageNode(role, options = {}) {
    el.empty?.remove();
    el.empty = null;

    const wrap = document.createElement('div');
    wrap.className = `msg ${role}${options.error ? ' error' : ''}`;

    const head = document.createElement('div');
    head.className = 'msg-head';
    const who = document.createElement('span');
    who.className = 'who';
    who.textContent = role === 'user' ? 'You' : 'Agent';
    const when = document.createElement('span');
    when.textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    head.append(who, when);

    const bubble = document.createElement('div');
    bubble.className = 'bubble';

    const meta = document.createElement('div');
    meta.className = 'meta';

    wrap.append(head, bubble, meta);
    el.transcript.appendChild(wrap);
    scrollDown();
    return { wrap, bubble, meta };
  }

  function scrollDown() {
    el.transcript.scrollTop = el.transcript.scrollHeight;
  }

  function addTag(meta, text, kind = '') {
    const tag = document.createElement('span');
    tag.className = `tag ${kind}`.trim();
    tag.textContent = text;
    meta.appendChild(tag);
    return tag;
  }

  function addToolChips(meta, tools) {
    tools.forEach((tool) => {
      const chip = document.createElement('button');
      chip.type = 'button';
      chip.className = 'tool-chip';
      const args = Object.values(tool.arguments || {}).join(', ');
      chip.innerHTML = `<b>${escapeHtml(tool.name)}</b>${args ? escapeHtml(String(args).slice(0, 28)) : ''}`;

      const detail = document.createElement('div');
      detail.className = 'tool-detail';
      detail.hidden = true;
      detail.textContent =
        `${tool.name}(${JSON.stringify(tool.arguments || {})})\n→ ${tool.result}`;

      chip.addEventListener('click', () => {
        detail.hidden = !detail.hidden;
        scrollDown();
      });
      meta.appendChild(chip);
      meta.parentElement.appendChild(detail);
    });
  }

  // Timer-driven rather than rAF-driven: a background tab suspends animation
  // frames entirely, and a half-written reply that never finishes would block
  // the turn behind it.
  function typeOut(node, text) {
    node.classList.add('caret');
    const chars = [...text];
    const perTick = Math.max(2, Math.ceil(chars.length / 45));
    let index = 0;
    return new Promise((resolve) => {
      const timer = setInterval(() => {
        index += perTick;
        node.textContent = chars.slice(0, index).join('');
        scrollDown();
        if (index >= chars.length) {
          clearInterval(timer);
          node.classList.remove('caret');
          resolve();
        }
      }, 16);
    });
  }

  function bumpTurns() {
    state.turns += 1;
    el.turnCount.textContent = `${state.turns} turn${state.turns === 1 ? '' : 's'}`;
  }

  // ------------------------------------------------------------ waveform

  function drawScope() {
    const width = el.scope.clientWidth;
    const height = el.scope.clientHeight;
    const dpr = window.devicePixelRatio || 1;
    if (el.scope.width !== width * dpr || el.scope.height !== height * dpr) {
      el.scope.width = width * dpr;
      el.scope.height = height * dpr;
      ctx2d.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    ctx2d.clearRect(0, 0, width, height);

    const styles = getComputedStyle(document.documentElement);
    const accent = styles.getPropertyValue('--accent').trim();
    const rec = styles.getPropertyValue('--rec').trim();
    const faint = styles.getPropertyValue('--surface-3').trim();

    const bars = Math.max(24, Math.floor(width / 9));
    const gap = 3;
    const barWidth = Math.max(2, (width - gap * (bars - 1)) / bars);
    const mid = height / 2;

    let data = null;
    if (state.analyser && (state.phase === 'recording' || state.phase === 'speaking')) {
      data = new Uint8Array(state.analyser.frequencyBinCount);
      state.analyser.getByteFrequencyData(data);
    }

    ctx2d.fillStyle = data ? (state.phase === 'recording' ? rec : accent) : faint;

    for (let i = 0; i < bars; i += 1) {
      let magnitude = 0.04;
      if (data) {
        // Sample the lower half of the spectrum, where speech actually lives.
        const index = Math.floor((i / bars) ** 1.35 * (data.length * 0.55));
        magnitude = Math.max(0.04, (data[index] / 255) ** 1.25);
      }
      const barHeight = Math.max(2, magnitude * (height - 6));
      const x = i * (barWidth + gap);
      ctx2d.beginPath();
      ctx2d.roundRect(x, mid - barHeight / 2, barWidth, barHeight, barWidth / 2);
      ctx2d.fill();
    }

    state.rafId = requestAnimationFrame(drawScope);
  }

  function pumpLevel() {
    if (!state.analyser) return 0;
    const buffer = new Uint8Array(state.analyser.fftSize);
    state.analyser.getByteTimeDomainData(buffer);
    let sum = 0;
    for (let i = 0; i < buffer.length; i += 1) {
      const deviation = (buffer[i] - 128) / 128;
      sum += deviation * deviation;
    }
    const rms = Math.sqrt(sum / buffer.length);
    state.level = state.level * 0.7 + rms * 0.3;
    return state.level;
  }

  // --------------------------------------------------------------- audio

  async function ensureAudio() {
    if (!state.audioCtx) {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      state.audioCtx = new AudioCtx();
    }
    if (state.audioCtx.state === 'suspended') await state.audioCtx.resume();

    if (!state.analyser) {
      state.analyser = state.audioCtx.createAnalyser();
      state.analyser.fftSize = 1024;
      state.analyser.smoothingTimeConstant = 0.72;
    }
    if (!state.playerSource) {
      state.player.crossOrigin = 'anonymous';
      state.playerSource = state.audioCtx.createMediaElementSource(state.player);
      state.playerSource.connect(state.analyser);
      state.playerSource.connect(state.audioCtx.destination);
    }
    return state.audioCtx;
  }

  async function ensureMic() {
    if (state.stream && state.stream.active) return state.stream;
    state.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
        channelCount: 1,
      },
    });
    await ensureAudio();
    state.micSource = state.audioCtx.createMediaStreamSource(state.stream);
    return state.stream;
  }

  function pickMimeType() {
    const candidates = [
      'audio/webm;codecs=opus',
      'audio/webm',
      'audio/ogg;codecs=opus',
      'audio/mp4',
    ];
    return candidates.find((type) => window.MediaRecorder?.isTypeSupported(type)) || '';
  }

  // ------------------------------------------------------------ recording

  async function startRecording() {
    if (state.phase === 'recording' || el.micBtn.disabled) return;
    stopPlayback();

    // Checked before touching the microphone: asking for permission and then
    // refusing to use it would be a rude way to report a server misconfig.
    if (!state.caps.asr) {
      banner('Speech-to-text is not configured on the server: set GROQ_API_KEY. Typing still works.');
      toast('Transcription is not configured', 'bad');
      return;
    }

    try {
      await ensureMic();
    } catch (error) {
      const denied = error && (error.name === 'NotAllowedError' || error.name === 'SecurityError');
      banner(denied
        ? 'Microphone access was blocked. Allow it in your browser’s site settings, then reload.'
        : 'No microphone was found. You can still type your message below.');
      toast('Microphone unavailable', 'bad');
      return;
    }

    const mimeType = pickMimeType();
    let recorder;
    try {
      recorder = new MediaRecorder(state.stream, mimeType ? { mimeType } : undefined);
    } catch {
      toast('This browser cannot record audio', 'bad');
      return;
    }

    state.recorder = recorder;
    state.chunks = [];
    state.speechSeen = false;
    state.silenceSince = 0;
    state.noiseFloor = 0.008;
    state.recordStart = performance.now();

    // Route the mic into the shared analyser only while recording, so the
    // waveform switches cleanly between input and playback.
    try { state.micSource.connect(state.analyser); } catch { /* already connected */ }

    recorder.ondataavailable = (event) => {
      if (event.data && event.data.size) state.chunks.push(event.data);
    };
    recorder.onstop = () => finishRecording(mimeType);
    recorder.start();

    resetPipeline();
    setStage('listen', 'active');
    setPhase('recording');

    // An interval, not animation frames: if the user switches tabs mid-turn
    // the silence detector has to keep running or the take never ends.
    state.autoStopTimer = setInterval(tickRecording, 50);
  }

  function tickRecording() {
    if (state.phase !== 'recording') return;

    const elapsed = performance.now() - state.recordStart;
    el.timer.textContent = `${(elapsed / 1000).toFixed(1)}s`;

    const level = pumpLevel();
    el.micRing.style.transform = `scale(${(0.86 + Math.min(level * 3.2, 0.5)).toFixed(3)})`;

    // First 400 ms establish the room's noise floor, so a noisy café needs a
    // louder voice to count as speech rather than triggering on hiss.
    if (elapsed < 400) {
      state.noiseFloor = Math.max(state.noiseFloor, level);
    } else {
      const threshold = Math.max(0.014, state.noiseFloor * 2.4);
      if (level > threshold) {
        state.speechSeen = true;
        state.silenceSince = 0;
      } else if (state.speechSeen) {
        if (!state.silenceSince) state.silenceSince = performance.now();
        if (performance.now() - state.silenceSince > state.settings.silence * 1000) {
          stopRecording();
          return;
        }
      }
    }

    if (elapsed > MAX_RECORD_MS) {
      toast('Reached the 30 second limit', 'bad');
      stopRecording();
    }
  }

  function stopRecording({ cancel = false } = {}) {
    if (state.phase !== 'recording') return;
    if (state.autoStopTimer) clearInterval(state.autoStopTimer);
    state.cancelled = cancel;
    try { state.recorder.stop(); } catch { /* already stopped */ }
    try { state.micSource.disconnect(state.analyser); } catch { /* not connected */ }
    el.micRing.style.transform = 'scale(0.86)';
  }

  async function finishRecording(mimeType) {
    const elapsed = performance.now() - state.recordStart;
    setStage('listen', 'done', Math.round(elapsed));
    setPhase('idle');

    if (state.cancelled) {
      state.cancelled = false;
      resetPipeline();
      toast('Recording cancelled');
      return;
    }
    if (elapsed < MIN_RECORD_MS || !state.chunks.length) {
      resetPipeline();
      toast('That was too short — hold a little longer');
      return;
    }

    const type = mimeType || state.chunks[0].type || 'audio/webm';
    const blob = new Blob(state.chunks, { type });
    if (blob.size > MAX_UPLOAD_BYTES) {
      toast('That clip is too large to upload — keep it under 30 seconds', 'bad');
      resetPipeline();
      return;
    }

    await handleAudio(blob, type);
  }

  // ----------------------------------------------------------- the turn

  async function handleAudio(blob, type) {
    setPhase('transcribing');
    setStage('asr', 'active');

    const extension = type.includes('mp4') ? 'mp4' : type.includes('ogg') ? 'ogg' : 'webm';
    const form = new FormData();
    form.append('audio', blob, `speech.${extension}`);

    let payload;
    try {
      const response = await fetch('/api/transcribe', { method: 'POST', body: form });
      payload = await readJson(response);
    } catch (error) {
      setStage('asr', 'failed');
      setPhase('idle');
      showError(`Transcription failed. ${error.message}`);
      return;
    }

    setStage('asr', 'done', payload.ms);

    if (payload.empty) {
      setPhase('idle');
      toast('I did not hear anything — try again a bit closer to the mic');
      if (state.settings.handsFree) setTimeout(startRecording, 400);
      return;
    }

    const node = messageNode('user');
    node.bubble.textContent = payload.text;
    addTag(node.meta, `heard in ${payload.ms} ms`);
    bumpTurns();

    await runAgent(payload.text);
  }

  async function sendTyped(text) {
    resetPipeline();
    setStage('listen', 'done', 0);
    setStage('asr', 'done', 0);
    const node = messageNode('user');
    node.bubble.textContent = text;
    addTag(node.meta, 'typed');
    bumpTurns();
    await runAgent(text);
  }

  async function runAgent(text) {
    setPhase('thinking');
    setStage('llm', 'active');

    state.history.push({ role: 'user', content: text });
    trimHistory();

    const node = messageNode('assistant');
    const dots = document.createElement('span');
    dots.className = 'thinking';
    dots.innerHTML = '<i></i><i></i><i></i>';
    node.bubble.appendChild(dots);
    scrollDown();

    let reply;
    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: state.history,
          use_tools: state.settings.useTools,
          system_prompt: state.settings.systemPrompt.trim() || null,
        }),
      });
      reply = await readJson(response);
    } catch (error) {
      setStage('llm', 'failed');
      setPhase('idle');
      node.wrap.remove();
      showError(`The agent could not answer. ${error.message}`);
      return;
    }

    setStage('llm', 'done', reply.ms);
    state.history = reply.messages.filter((message) => message.role !== 'system');
    trimHistory();

    node.bubble.textContent = '';
    if (reply.tools?.length) addToolChips(node.meta, reply.tools);
    addTag(node.meta, `thought in ${reply.ms} ms`);

    const speaking = state.settings.autoPlay;
    const typing = typeOut(node.bubble, reply.text);

    if (speaking) {
      await Promise.all([typing, speak(reply.text, node)]);
    } else {
      await typing;
      setPhase('idle');
    }

    addReplayButton(node, reply.text);

    if (state.settings.handsFree && state.caps.asr) {
      setTimeout(() => { if (state.phase === 'idle') startRecording(); }, 350);
    }
  }

  function trimHistory() {
    if (state.history.length <= HISTORY_LIMIT) return;
    state.history = state.history.slice(-HISTORY_LIMIT);
    // A tool result with no matching call confuses the model, so drop any
    // orphans left at the front by the trim.
    while (state.history.length && state.history[0].role === 'tool') {
      state.history.shift();
    }
  }

  // --------------------------------------------------------------- speech

  async function speak(text, node) {
    const useBrowser = state.settings.browserVoice || !state.caps.tts;
    setStage('tts', 'active');

    if (useBrowser) {
      if (node) addTag(node.meta, 'browser voice');
      const started = performance.now();
      const ok = await speakWithBrowser(text);
      setStage('tts', ok ? 'done' : 'failed', Math.round(performance.now() - started));
      return;
    }

    const started = performance.now();
    let blob;
    try {
      const response = await fetch('/api/speak', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text,
          voice_id: state.settings.voiceId || null,
          speed: state.settings.speed,
        }),
      });
      if (!response.ok) throw new Error(await errorText(response));
      blob = await response.blob();
    } catch (error) {
      setStage('tts', 'failed');
      toast(`Voice unavailable, using the browser voice. ${error.message}`, 'bad');
      speakWithBrowser(text);
      if (node) addTag(node.meta, 'browser voice');
      return;
    }

    setStage('tts', 'done', Math.round(performance.now() - started));
    if (node) {
      node.clipId = `clip-${Date.now()}`;
      state.clips.set(node.clipId, blob);
    }
    await play(blob);
  }

  function speakWithBrowser(text) {
    if (!('speechSynthesis' in window)) return Promise.resolve(false);

    const synth = window.speechSynthesis;
    synth.cancel();

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = state.settings.speed;

    return new Promise((resolve) => {
      let settled = false;
      // `onend` is unreliable — some browsers drop it on long utterances, and
      // a lost event would strand the UI in the speaking state forever.
      const deadline = 5000 + text.split(/\s+/).length * 450;
      const finish = (ok) => {
        if (settled) return;
        settled = true;
        clearInterval(watchdog);
        if (state.phase === 'speaking') setPhase('idle');
        resolve(ok);
      };

      const startedAt = performance.now();
      const watchdog = setInterval(() => {
        const waited = performance.now() - startedAt;
        // The 800 ms grace period covers the gap before speaking begins.
        if (waited > deadline || (waited > 800 && !synth.speaking && !synth.pending)) {
          finish(true);
        }
      }, 250);

      utterance.onstart = () => setPhase('speaking');
      utterance.onend = () => finish(true);
      utterance.onerror = () => finish(false);

      synth.speak(utterance);
    });
  }

  async function play(blob) {
    await ensureAudio();
    if (state.lastObjectUrl) URL.revokeObjectURL(state.lastObjectUrl);
    state.lastObjectUrl = URL.createObjectURL(blob);
    state.player.src = state.lastObjectUrl;

    setPhase('speaking');
    try {
      await state.player.play();
    } catch {
      setPhase('idle');
      toast('Tap anywhere once to let the browser play audio');
      return;
    }

    await new Promise((resolve) => {
      const done = () => {
        state.player.removeEventListener('ended', done);
        state.player.removeEventListener('error', done);
        resolve();
      };
      state.player.addEventListener('ended', done);
      state.player.addEventListener('error', done);
    });

    if (state.phase === 'speaking') setPhase('idle');
  }

  function stopPlayback() {
    if (!state.player.paused) {
      state.player.pause();
      state.player.currentTime = 0;
    }
    if ('speechSynthesis' in window) window.speechSynthesis.cancel();
    if (state.phase === 'speaking') setPhase('idle');
  }

  function addReplayButton(node, text) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'replay';
    button.innerHTML = '<svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M5 3l14 9-14 9z"/></svg>Replay';
    button.addEventListener('click', async () => {
      stopPlayback();
      const clip = node.clipId && state.clips.get(node.clipId);
      if (clip) await play(clip);
      else await speak(text, null);
    });
    node.meta.appendChild(button);
  }

  // ---------------------------------------------------------------- http

  async function readJson(response) {
    if (!response.ok) throw new Error(await errorText(response));
    return response.json();
  }

  async function errorText(response) {
    try {
      const payload = await response.json();
      return payload.detail || payload.error || `HTTP ${response.status}`;
    } catch {
      return `HTTP ${response.status}`;
    }
  }

  function showError(message) {
    const node = messageNode('assistant', { error: true });
    node.bubble.textContent = message;
    toast(message, 'bad');
  }

  // ---------------------------------------------------------------- init

  async function loadHealth() {
    let health;
    try {
      health = await readJson(await fetch('/api/health'));
    } catch {
      banner('The API is not reachable. Run "uvicorn api.index:app --reload" or check your deployment.');
      updateCapsules({ asr: false, llm: false, tts: false });
      return;
    }

    state.caps = { asr: health.asr, llm: health.llm, tts: health.tts };
    updateCapsules(state.caps);

    el.toolset.innerHTML = '';
    (health.tools || []).forEach((tool) => {
      const span = document.createElement('span');
      span.textContent = tool;
      el.toolset.appendChild(span);
    });

    el.buildInfo.textContent =
      `${health.models.asr} · ${health.models.llm} · ${health.models.tts}`;

    if (!health.llm) {
      banner('GROQ_API_KEY is missing on the server, so the agent cannot listen or answer yet.');
    } else if (!health.tts) {
      el.voiceNote.textContent = 'No ElevenLabs key on the server — replies use the browser voice.';
      state.settings.browserVoice = true;
      el.browserVoice.checked = true;
      el.browserVoice.disabled = true;
    }

    if (health.tts) loadVoices(health.default_voice);
    state.ready = true;
  }

  function updateCapsules(caps) {
    Object.entries(caps).forEach(([key, value]) => {
      const capsule = el.capabilities.querySelector(`[data-cap="${key}"]`);
      if (capsule) capsule.className = `capsule ${value ? 'on' : 'off'}`;
    });
  }

  async function loadVoices(defaultVoice) {
    let voices = [];
    try {
      voices = (await readJson(await fetch('/api/voices'))).voices || [];
    } catch { /* the picker is optional */ }

    el.voiceSelect.innerHTML = '';
    if (!voices.length) {
      el.voiceSelect.appendChild(new Option('Server default', ''));
      el.voiceSelect.disabled = true;
      return;
    }
    voices.forEach((voice) => {
      const label = voice.labels?.accent
        ? `${voice.name} — ${voice.labels.accent}`
        : voice.name;
      el.voiceSelect.appendChild(new Option(label, voice.id));
    });
    // Assigning an id the account does not have leaves the select empty, so
    // fall back to a voice that is definitely on the list.
    el.voiceSelect.value = state.settings.voiceId || defaultVoice || '';
    if (!el.voiceSelect.value) el.voiceSelect.value = voices[0].id;
    state.settings.voiceId = el.voiceSelect.value;
    saveSettings();
  }

  function applySettings() {
    document.documentElement.dataset.theme = state.settings.theme;
    el.speedRange.value = state.settings.speed;
    el.speedValue.textContent = `${Number(state.settings.speed).toFixed(2)}x`;
    el.silenceRange.value = state.settings.silence;
    el.silenceValue.textContent = `${Number(state.settings.silence).toFixed(1)}s`;
    el.systemPrompt.value = state.settings.systemPrompt;
    el.useTools.checked = state.settings.useTools;
    el.autoPlay.checked = state.settings.autoPlay;
    el.browserVoice.checked = state.settings.browserVoice;
    el.handsFree.checked = state.settings.handsFree;
  }

  function toggleDrawer(open) {
    el.drawer.hidden = !open;
    el.scrim.hidden = !open;
    el.settingsToggle.setAttribute('aria-expanded', String(open));
  }

  function bindEvents() {
    el.micBtn.addEventListener('click', () => {
      if (state.phase === 'recording') stopRecording();
      else startRecording();
    });

    // Hold-to-talk. Ignored while typing so Space still types a space.
    let spaceHeld = false;
    document.addEventListener('keydown', (event) => {
      const typing = ['INPUT', 'TEXTAREA', 'SELECT'].includes(event.target.tagName);
      if (event.code === 'Space' && !typing && !event.repeat) {
        event.preventDefault();
        spaceHeld = true;
        if (state.phase !== 'recording') startRecording();
      }
      if (event.key === 'Escape') {
        if (state.phase === 'recording') stopRecording({ cancel: true });
        else if (state.phase === 'speaking') stopPlayback();
        else if (!el.drawer.hidden) toggleDrawer(false);
      }
    });
    document.addEventListener('keyup', (event) => {
      if (event.code === 'Space' && spaceHeld) {
        spaceHeld = false;
        if (state.phase === 'recording') stopRecording();
      }
    });

    el.composer.addEventListener('submit', (event) => {
      event.preventDefault();
      const text = el.textInput.value.trim();
      if (!text || el.sendBtn.disabled) return;
      el.textInput.value = '';
      sendTyped(text);
    });

    el.transcript.addEventListener('click', (event) => {
      const chip = event.target.closest('.chip');
      if (chip) sendTyped(chip.dataset.prompt);
    });

    el.clearBtn.addEventListener('click', () => {
      stopPlayback();
      state.history = [];
      state.turns = 0;
      state.clips.clear();
      el.turnCount.textContent = '0 turns';
      el.transcript.innerHTML = '';
      resetPipeline();
      location.reload();
    });

    el.exportBtn.addEventListener('click', exportTranscript);
    el.stopAudioBtn.addEventListener('click', stopPlayback);

    el.themeToggle.addEventListener('click', () => {
      state.settings.theme = state.settings.theme === 'dark' ? 'light' : 'dark';
      document.documentElement.dataset.theme = state.settings.theme;
      saveSettings();
    });

    el.settingsToggle.addEventListener('click', () => toggleDrawer(el.drawer.hidden));
    el.drawerClose.addEventListener('click', () => toggleDrawer(false));
    el.scrim.addEventListener('click', () => toggleDrawer(false));
    el.bannerClose.addEventListener('click', () => { el.banner.hidden = true; });

    el.voiceSelect.addEventListener('change', () => {
      state.settings.voiceId = el.voiceSelect.value;
      saveSettings();
    });
    el.speedRange.addEventListener('input', () => {
      state.settings.speed = Number(el.speedRange.value);
      el.speedValue.textContent = `${state.settings.speed.toFixed(2)}x`;
      saveSettings();
    });
    el.silenceRange.addEventListener('input', () => {
      state.settings.silence = Number(el.silenceRange.value);
      el.silenceValue.textContent = `${state.settings.silence.toFixed(1)}s`;
      saveSettings();
    });
    el.systemPrompt.addEventListener('change', () => {
      state.settings.systemPrompt = el.systemPrompt.value;
      saveSettings();
    });
    [['useTools', 'useTools'], ['autoPlay', 'autoPlay'], ['browserVoice', 'browserVoice']]
      .forEach(([id, key]) => {
        el[id].addEventListener('change', () => {
          state.settings[key] = el[id].checked;
          saveSettings();
        });
      });
    el.handsFree.addEventListener('change', () => {
      state.settings.handsFree = el.handsFree.checked;
      saveSettings();
      if (el.handsFree.checked && state.phase === 'idle') startRecording();
    });

    window.addEventListener('beforeunload', () => {
      if (state.lastObjectUrl) URL.revokeObjectURL(state.lastObjectUrl);
    });
  }

  function exportTranscript() {
    const lines = [...el.transcript.querySelectorAll('.msg')].map((node) => {
      const who = node.querySelector('.who')?.textContent || '';
      const body = node.querySelector('.bubble')?.textContent || '';
      return `${who.toUpperCase()}: ${body}`;
    });
    if (!lines.length) {
      toast('Nothing to export yet');
      return;
    }
    const blob = new Blob(
      [`Voice Agent transcript — ${new Date().toLocaleString()}\n\n${lines.join('\n\n')}\n`],
      { type: 'text/plain' },
    );
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `voice-agent-${Date.now()}.txt`;
    link.click();
    URL.revokeObjectURL(url);
    toast('Transcript downloaded', 'good');
  }

  // Older Safari has no roundRect; a plain rectangle is a fine stand-in.
  if (!CanvasRenderingContext2D.prototype.roundRect) {
    CanvasRenderingContext2D.prototype.roundRect = function (x, y, w, h) {
      this.rect(x, y, w, h);
      return this;
    };
  }

  applySettings();
  bindEvents();
  setPhase('idle');
  drawScope();
  loadHealth();
})();
