import { HandLandmarker, FilesetResolver } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/+esm';
import { createYoloHttpBridge } from './yolo_bridge_client.js';

const humanBoard = document.getElementById('human-board');
const aiBoard = document.getElementById('ai-board');
const bgmEl = document.getElementById('bgm');
const roundEndAudioEl = document.getElementById('round-end-audio');
const missHitAudioEl = document.getElementById('miss-hit-audio');
const hitSuccessAudioEl = document.getElementById('hit-success-audio');
const cameraFeed = document.getElementById('camera-feed');
const cameraPreview = document.getElementById('camera-preview');
const cameraPreviewOverlay = document.getElementById('camera-preview-overlay');
const cameraPreviewCtx = cameraPreviewOverlay.getContext('2d');
const handOverlay = document.getElementById('hand-overlay');
const handCtx = handOverlay.getContext('2d');
const aiVisionOverlay = document.getElementById('ai-vision-overlay');
const aiVisionCtx = aiVisionOverlay.getContext('2d');
const humanScoreEl = document.getElementById('human-score');
const aiScoreEl = document.getElementById('ai-score');
const humanHitRateEl = document.getElementById('human-hit-rate');
const aiHitRateEl = document.getElementById('ai-hit-rate');
const timeLeftEl = document.getElementById('time-left');
const roundLabelEl = document.getElementById('round-label');
const gameStatusEl = document.getElementById('game-status');
const humanHitsEl = document.getElementById('human-hits');
const humanMissesEl = document.getElementById('human-misses');
const aiHitsEl = document.getElementById('ai-hits');
const aiMissesEl = document.getElementById('ai-misses');
const cameraStatusEl = document.getElementById('camera-status');
const handStatusEl = document.getElementById('hand-status');
const trackerStatusTextEl = document.getElementById('tracker-status-text');
const aiVisionStatusEl = document.getElementById('ai-vision-status');
const aiPolicyStatusEl = document.getElementById('ai-policy-status');
const aiDetectorModeEl = document.getElementById('ai-detector-mode');
const aiLastDetectionEl = document.getElementById('ai-last-detection');
const aiBridgeProtocolEl = document.getElementById('ai-bridge-protocol');
const aiBridgeStateEl = document.getElementById('ai-bridge-state');
const difficultyRange = document.getElementById('difficulty-range');
const difficultyValueEl = document.getElementById('difficulty-value');
const sensitivityRange = document.getElementById('sensitivity-range');
const sensitivityValueEl = document.getElementById('sensitivity-value');
const smoothingRange = document.getElementById('smoothing-range');
const smoothingValueEl = document.getElementById('smoothing-value');
const sensitivityPreset = document.getElementById('sensitivity-preset');
const durationSelect = document.getElementById('duration-select');
const roundsSelect = document.getElementById('rounds-select');
const startButton = document.getElementById('start-button');
const pauseButton = document.getElementById('pause-button');
const resetButton = document.getElementById('reset-button');
const historyList = document.getElementById('history-list');
const matchScoreEl = document.getElementById('match-score');

const HOLE_COUNT = 9;
const HISTORY_KEY = 'mole-arena-history';
const MEDIAPIPE_WASM = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm';
const HAND_MODEL =
  'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';
const YOLO_HTTP_ENDPOINT =
  window.localStorage.getItem('mole-arena-yolo-endpoint') || 'http://127.0.0.1:8765/api/detections';

const audioCtx = window.AudioContext ? new AudioContext() : null;

const state = {
  status: 'ready',
  durationSec: Number(durationSelect.value),
  difficulty: Number(difficultyRange.value),
  totalRounds: Number(roundsSelect.value),
  currentRound: 1,
  roundTimeLeftMs: Number(durationSelect.value) * 1000,
  lastTickAt: 0,
  handLandmarker: null,
  handTrackingReady: false,
  handTrackingBusy: false,
  lastVideoTime: -1,
  lastHandLandmarks: null,
  aiVisionBridge: {
    mode: 'synthetic-yolo',
    protocol: 'window.postMessage',
    state: 'listening',
    detections: [],
    lastScanAt: 0,
    lastExternalAt: 0,
    scanTimer: null,
    httpBridge: null,
  },
  human: {
    score: 0,
    hits: 0,
    misses: 0,
    attempts: 0,
    roundWins: 0,
  },
  ai: {
    score: 0,
    hits: 0,
    misses: 0,
    attempts: 0,
    roundWins: 0,
  },
  boards: {
    human: createBoardState('human'),
    ai: createBoardState('ai'),
  },
  history: loadHistory(),
  roundSpawnTimer: null,
  aiThinkTimer: null,
  cameraStream: null,
  handSensitivity: Number(sensitivityRange.value) / 10,
  handSmoothing: Number(smoothingRange.value) / 10,
  handReach: {
    xMin: 0.18,
    xMax: 0.82,
    yMin: 0.05,
    yMax: 0.72,
  },
  pointer: {
    x: handOverlay.width * 0.5,
    y: handOverlay.height * 0.72,
    lastX: handOverlay.width * 0.5,
    lastY: handOverlay.height * 0.72,
    speed: 0,
    insideIndex: null,
    source: 'mouse',
    pinchArmed: true,
    smashUntil: 0,
  },
};

function createBoardState(owner) {
  return {
    owner,
    holes: Array.from({ length: HOLE_COUNT }, (_, index) => ({
      index,
      element: null,
      active: false,
      type: 'normal',
      expiresAt: 0,
      locked: false,
    })),
  };
}

function ensureAudioReady() {
  if (!audioCtx) {
    return;
  }
  if (audioCtx.state === 'suspended') {
    audioCtx.resume().catch(() => {});
  }
}

function playTone({ frequency = 440, duration = 0.08, type = 'square', gain = 0.05, sweep = 0 } = {}) {
  if (!audioCtx) {
    return;
  }
  ensureAudioReady();
  const now = audioCtx.currentTime;
  const osc = audioCtx.createOscillator();
  const amp = audioCtx.createGain();
  osc.type = type;
  osc.frequency.setValueAtTime(frequency, now);
  if (sweep) {
    osc.frequency.linearRampToValueAtTime(frequency + sweep, now + duration);
  }
  amp.gain.setValueAtTime(0.001, now);
  amp.gain.exponentialRampToValueAtTime(gain, now + 0.01);
  amp.gain.exponentialRampToValueAtTime(0.001, now + duration);
  osc.connect(amp);
  amp.connect(audioCtx.destination);
  osc.start(now);
  osc.stop(now + duration + 0.02);
}

function playHitSound(kind = 'good') {
  ensureAudioReady();
  if (kind === 'good') {
    if (hitSuccessAudioEl) {
      try {
        hitSuccessAudioEl.currentTime = 0;
        hitSuccessAudioEl.volume = 0.9;
        hitSuccessAudioEl.play().catch(() => {});
        return;
      } catch {
        // Fall back to synthesized cue.
      }
    }
    playTone({ frequency: 440, duration: 0.06, type: 'square', gain: 0.07, sweep: 140 });
    window.setTimeout(() => playTone({ frequency: 620, duration: 0.07, type: 'triangle', gain: 0.05, sweep: 90 }), 30);
    return;
  }
  if (missHitAudioEl) {
    try {
      missHitAudioEl.currentTime = 0;
      missHitAudioEl.volume = 0.85;
      missHitAudioEl.play().catch(() => {});
      return;
    } catch {
      // Fall back to synthesized cue.
    }
  }
  playTone({ frequency: 180, duration: 0.12, type: 'sawtooth', gain: 0.06, sweep: -60 });
}

function playRoundEndSound() {
  ensureAudioReady();
  if (roundEndAudioEl) {
    try {
      roundEndAudioEl.currentTime = 0;
      roundEndAudioEl.volume = 0.9;
      roundEndAudioEl.play().catch(() => {});
      return;
    } catch {
      // Fall back to synthesized cue.
    }
  }
  playTone({ frequency: 520, duration: 0.08, type: 'triangle', gain: 0.05, sweep: 40 });
  window.setTimeout(() => playTone({ frequency: 660, duration: 0.08, type: 'triangle', gain: 0.05, sweep: 20 }), 90);
}

async function startBgm() {
  ensureAudioReady();
  if (!bgmEl) {
    return;
  }
  bgmEl.volume = 0.45;
  try {
    await bgmEl.play();
  } catch {
    // Browser gesture restrictions are expected on first load.
  }
}

function pauseBgm() {
  if (bgmEl) {
    bgmEl.pause();
  }
}

function loadHistory() {
  try {
    return JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]');
  } catch {
    return [];
  }
}

function saveHistory() {
  localStorage.setItem(HISTORY_KEY, JSON.stringify(state.history.slice(0, 12)));
}

function createHoleElement(owner, holeState) {
  const button = document.createElement('button');
  button.className = 'hole';
  button.type = 'button';
  button.dataset.owner = owner;
  button.dataset.index = String(holeState.index);
  button.innerHTML = '<span class="mole normal"></span>';
  button.addEventListener('mouseenter', () => {
    if (owner === 'human' && state.status === 'playing' && state.pointer.source === 'mouse') {
      button.classList.add('hand-target');
      handStatusEl.textContent = 'Hand Cursor Tracking';
    }
  });
  button.addEventListener('mouseleave', () => {
    button.classList.remove('hand-target');
  });
  button.addEventListener('click', () => {
    if (owner === 'human' && state.pointer.source === 'mouse') {
      strikeHole('human', holeState.index, 'tap');
    }
  });
  return button;
}

function mountBoards() {
  humanBoard.innerHTML = '';
  aiBoard.innerHTML = '';
  state.boards.human.holes.forEach((holeState) => {
    holeState.element = createHoleElement('human', holeState);
    humanBoard.appendChild(holeState.element);
  });
  state.boards.ai.holes.forEach((holeState) => {
    holeState.element = createHoleElement('ai', holeState);
    aiBoard.appendChild(holeState.element);
  });
}

function setOverlay(_tag, title, text, _visible = true) {
  gameStatusEl.textContent = title;
  aiPolicyStatusEl.textContent = text;
}

function updateControls() {
  const locked = state.status === 'playing' || state.status === 'paused' || state.status === 'countdown';
  durationSelect.disabled = locked;
  roundsSelect.disabled = locked;
  difficultyRange.disabled = locked;
  sensitivityRange.disabled = locked;
  smoothingRange.disabled = locked;
  sensitivityPreset.disabled = locked;
  pauseButton.disabled = !(state.status === 'playing' || state.status === 'paused');
}

function calcRate(player) {
  if (!player.attempts) {
    return 0;
  }
  return Math.round((player.hits / player.attempts) * 100);
}

function readableStatus(status) {
  return {
    ready: 'Ready',
    countdown: 'Countdown',
    playing: 'Playing',
    paused: 'Paused',
    round_result: 'Round Result',
    match_result: 'Match Finished',
  }[status];
}

function renderHistory() {
  if (!state.history.length) {
    historyList.innerHTML = '<p class="empty-history">No rounds finished yet.</p>';
    return;
  }

  historyList.innerHTML = state.history
    .map(
      (item) => `
        <article class="history-item">
          <span>Round ${item.round} · ${item.durationSec}s · Difficulty ${item.difficulty}</span>
          <strong>${item.humanScore} : ${item.aiScore} · ${item.winnerLabel}</strong>
          <span>Human ${item.humanHits} hits / AI ${item.aiHits} hits</span>
        </article>
      `
    )
    .join('');
}

function updateAIBridgeHud() {
  aiDetectorModeEl.textContent =
    state.aiVisionBridge.mode === 'external-yolo' ? 'External YOLO' : 'Synthetic YOLO';
  aiBridgeProtocolEl.textContent = `${state.aiVisionBridge.protocol} @ ${YOLO_HTTP_ENDPOINT}`;
  aiBridgeStateEl.textContent = state.aiVisionBridge.state;
}

function updateHud() {
  humanScoreEl.textContent = String(state.human.score);
  aiScoreEl.textContent = String(state.ai.score);
  humanHitsEl.textContent = `Hits ${state.human.hits}`;
  humanMissesEl.textContent = `Miss ${state.human.misses}`;
  aiHitsEl.textContent = `Hits ${state.ai.hits}`;
  aiMissesEl.textContent = `Miss ${state.ai.misses}`;
  humanHitRateEl.textContent = `Hit ${calcRate(state.human)}%`;
  aiHitRateEl.textContent = `Hit ${calcRate(state.ai)}%`;
  roundLabelEl.textContent = `${state.currentRound} / ${state.totalRounds}`;
  timeLeftEl.textContent = `${Math.max(0, Math.ceil(state.roundTimeLeftMs / 1000))}s`;
  gameStatusEl.textContent = readableStatus(state.status);
  difficultyValueEl.textContent = String(state.difficulty);
  sensitivityValueEl.textContent = `${state.handSensitivity.toFixed(1)}x`;
  smoothingValueEl.textContent = state.handSmoothing.toFixed(2);
  matchScoreEl.textContent = `${state.human.roundWins} : ${state.ai.roundWins}`;
  updateControls();
  updateAIBridgeHud();
  renderHistory();
}

function clearBoard(boardKey) {
  state.boards[boardKey].holes.forEach((hole) => {
    hole.active = false;
    hole.type = 'normal';
    hole.expiresAt = 0;
    hole.locked = false;
    hole.element.classList.remove('active', 'hit-flash', 'ai-lock', 'hand-target', 'hit-success', 'hit-bad');
    hole.element.querySelector('.mole').className = 'mole normal';
  });
}

function clearAllBoards() {
  clearBoard('human');
  clearBoard('ai');
}

function resetRoundStats() {
  state.human.score = 0;
  state.human.hits = 0;
  state.human.misses = 0;
  state.human.attempts = 0;
  state.ai.score = 0;
  state.ai.hits = 0;
  state.ai.misses = 0;
  state.ai.attempts = 0;
  state.roundTimeLeftMs = state.durationSec * 1000;
  clearAllBoards();
}

function getHoleCenter(boardElement, holeElement, overlayCanvas) {
  const boardRect = boardElement.getBoundingClientRect();
  const holeRect = holeElement.getBoundingClientRect();
  const scaleX = overlayCanvas.width / boardRect.width;
  const scaleY = overlayCanvas.height / boardRect.height;
  return {
    x: (holeRect.left - boardRect.left + holeRect.width / 2) * scaleX,
    y: (holeRect.top - boardRect.top + holeRect.height / 2) * scaleY,
  };
}

function drawHammerIcon(x, y, scale = 1) {
  const angle = Math.max(-0.38, Math.min(0.38, (state.pointer.x - state.pointer.lastX) / 120));
  const smashing = state.pointer.smashUntil > performance.now();
  const smashDrop = smashing ? 22 : 0;
  const smashScaleY = smashing ? 1.08 : 1;

  handCtx.save();
  handCtx.translate(x, y + smashDrop);
  handCtx.rotate(angle);
  handCtx.scale(scale, scale * smashScaleY);

  handCtx.strokeStyle = '#03170a';
  handCtx.lineWidth = 8;
  handCtx.lineJoin = 'round';
  handCtx.globalAlpha = 1;
  handCtx.shadowColor = '#0b3516';
  handCtx.shadowBlur = 0;

  handCtx.fillStyle = '#00e85f';
  handCtx.beginPath();
  handCtx.roundRect(-10, -82, 20, 92, 11);
  handCtx.fill();
  handCtx.stroke();

  handCtx.strokeStyle = '#f7fff8';
  handCtx.lineWidth = 3;
  handCtx.beginPath();
  handCtx.roundRect(-6, -78, 6, 74, 6);
  handCtx.stroke();

  handCtx.strokeStyle = '#03170a';
  handCtx.lineWidth = 8;
  handCtx.fillStyle = '#39ff14';
  handCtx.beginPath();
  handCtx.roundRect(-44, -124, 88, 34, 12);
  handCtx.fill();
  handCtx.stroke();

  handCtx.strokeStyle = '#f7fff8';
  handCtx.lineWidth = 3;
  handCtx.beginPath();
  handCtx.roundRect(-34, -118, 68, 10, 5);
  handCtx.stroke();

  handCtx.fillStyle = '#d9ff54';
  handCtx.beginPath();
  handCtx.roundRect(-28, -110, 56, 10, 5);
  handCtx.fill();

  handCtx.fillStyle = '#03170a';
  handCtx.beginPath();
  handCtx.arc(0, -106, 5, 0, Math.PI * 2);
  handCtx.fill();

  handCtx.restore();
}

function drawHandOverlay(landmarks = null) {
  handCtx.clearRect(0, 0, handOverlay.width, handOverlay.height);
  if (state.status === 'ready' && !state.handTrackingReady) {
    return;
  }

  handCtx.save();
  handCtx.lineCap = 'round';
  handCtx.lineWidth = 12;
  handCtx.strokeStyle = '#7dff4f';
  handCtx.beginPath();
  handCtx.moveTo(state.pointer.lastX, state.pointer.lastY);
  handCtx.lineTo(state.pointer.x, state.pointer.y);
  handCtx.stroke();

  const hitPoint = getHammerHitPoint();

  handCtx.strokeStyle = '#f3ff9b';
  handCtx.lineWidth = 6;
  handCtx.beginPath();
  handCtx.arc(hitPoint.x, hitPoint.y, 58, 0, Math.PI * 2);
  handCtx.stroke();

  handCtx.fillStyle = '#f3ff9b';
  handCtx.beginPath();
  handCtx.arc(hitPoint.x, hitPoint.y, 8, 0, Math.PI * 2);
  handCtx.fill();

  if (landmarks) {
    const keyIndices = [0, 4, 8, 12, 16, 20];
    handCtx.fillStyle = '#d7ffe3';
    handCtx.strokeStyle = '#24b653';
    handCtx.lineWidth = 2;
    for (const index of keyIndices) {
      const point = landmarks[index];
      if (!point) {
        continue;
      }
      const px = (1 - point.x) * handOverlay.width;
      const py = point.y * handOverlay.height;
      handCtx.beginPath();
      handCtx.arc(px, py, index === 8 ? 7 : 5, 0, Math.PI * 2);
      handCtx.fill();
      handCtx.stroke();
    }
  }

  handCtx.restore();

  handCtx.save();
  drawHammerIcon(state.pointer.x, state.pointer.y, 1);
  handCtx.restore();
}

function drawCameraPreviewOverlay(landmarks = null) {
  cameraPreviewCtx.clearRect(0, 0, cameraPreviewOverlay.width, cameraPreviewOverlay.height);
  if (!landmarks) {
    return;
  }

  let minX = cameraPreviewOverlay.width;
  let minY = cameraPreviewOverlay.height;
  let maxX = 0;
  let maxY = 0;

  for (const point of landmarks) {
    if (!point) {
      continue;
    }
    const px = (1 - point.x) * cameraPreviewOverlay.width;
    const py = point.y * cameraPreviewOverlay.height;
    minX = Math.min(minX, px);
    minY = Math.min(minY, py);
    maxX = Math.max(maxX, px);
    maxY = Math.max(maxY, py);
  }

  const padding = 10;
  const boxX = Math.max(0, minX - padding);
  const boxY = Math.max(0, minY - padding);
  const boxW = Math.min(cameraPreviewOverlay.width - boxX, maxX - minX + padding * 2);
  const boxH = Math.min(cameraPreviewOverlay.height - boxY, maxY - minY + padding * 2);

  cameraPreviewCtx.save();
  cameraPreviewCtx.strokeStyle = '#39ff14';
  cameraPreviewCtx.lineWidth = 1.5;
  cameraPreviewCtx.shadowColor = 'rgba(57, 255, 20, 0.28)';
  cameraPreviewCtx.shadowBlur = 6;
  cameraPreviewCtx.strokeRect(boxX, boxY, boxW, boxH);
  cameraPreviewCtx.restore();
}

function drawAIVisionOverlay() {
  aiVisionCtx.clearRect(0, 0, aiVisionOverlay.width, aiVisionOverlay.height);
  if (state.status !== 'playing' && state.status !== 'paused') {
    return;
  }

  aiVisionCtx.save();
  aiVisionCtx.strokeStyle = 'rgba(255, 255, 255, 0.14)';
  aiVisionCtx.lineWidth = 2;
  aiVisionCtx.beginPath();
  const scanX = ((Date.now() / 12) % aiVisionOverlay.width);
  aiVisionCtx.moveTo(scanX, 0);
  aiVisionCtx.lineTo(scanX, aiVisionOverlay.height);
  aiVisionCtx.stroke();
  aiVisionCtx.restore();

  state.aiVisionBridge.detections.forEach((detection) => {
    const hole = state.boards.ai.holes[detection.holeIndex];
    if (!hole) {
      return;
    }
    const center = getHoleCenter(aiBoard, hole.element, aiVisionOverlay);
    const boxWidth = 128;
    const boxHeight = 144;
    aiVisionCtx.save();
    aiVisionCtx.strokeStyle = detection.label === 'bomb' ? 'rgba(60, 44, 44, 0.76)' : 'rgba(46, 125, 107, 0.88)';
    aiVisionCtx.fillStyle = 'rgba(255, 255, 255, 0.92)';
    aiVisionCtx.lineWidth = detection.locked ? 5 : 4;
    aiVisionCtx.setLineDash(detection.locked ? [] : [10, 8]);
    aiVisionCtx.strokeRect(center.x - boxWidth / 2, center.y - boxHeight / 2, boxWidth, boxHeight);
    aiVisionCtx.setLineDash([]);
    aiVisionCtx.font = '700 18px Manrope';
    aiVisionCtx.fillText(
      `${detection.label.toUpperCase()} ${(detection.confidence * 100).toFixed(0)}%`,
      center.x - 58,
      center.y - boxHeight / 2 - 10
    );
    aiVisionCtx.font = '600 14px Manrope';
    aiVisionCtx.fillText(`H${detection.holeIndex + 1}`, center.x - 14, center.y + boxHeight / 2 + 18);
    aiVisionCtx.restore();
  });
}

function renderSensorLayers(landmarks = state.lastHandLandmarks) {
  drawHandOverlay(landmarks);
  drawCameraPreviewOverlay(landmarks);
  drawAIVisionOverlay();
}

function getHammerHitPoint() {
  return {
    x: state.pointer.x,
    y: state.pointer.y - 42,
  };
}

function syncPointerHitTarget() {
  let activeIndex = null;
  const hitPoint = getHammerHitPoint();
  state.boards.human.holes.forEach((hole) => {
    hole.element.classList.remove('hand-target');
    if (!hole.active) {
      return;
    }
    const center = getHoleCenter(humanBoard, hole.element, handOverlay);
    const distance = Math.hypot(center.x - hitPoint.x, center.y - hitPoint.y);
    if (distance < 88) {
      activeIndex = hole.index;
      hole.element.classList.add('hand-target');
    }
  });
  state.pointer.insideIndex = activeIndex;
}

function remapHandCoordinate(value, min, max, size) {
  const normalized = (value - min) / (max - min);
  return Math.max(0, Math.min(size, normalized * size));
}

function updatePointerPosition(x, y, source = 'mouse') {
  const targetX = source === 'hand'
    ? remapHandCoordinate(x / handOverlay.width, state.handReach.xMin, state.handReach.xMax, handOverlay.width)
    : x;
  const targetY = source === 'hand'
    ? remapHandCoordinate(y / handOverlay.height, state.handReach.yMin, state.handReach.yMax, handOverlay.height)
    : y;
  const followFactor = source === 'hand'
    ? Math.max(0.12, Math.min(1, state.handSensitivity * (1 - state.handSmoothing) + 0.08))
    : 1;
  const nextX = source === 'hand'
    ? state.pointer.x + (targetX - state.pointer.x) * followFactor
    : targetX;
  const nextY = source === 'hand'
    ? state.pointer.y + (targetY - state.pointer.y) * followFactor
    : targetY;
  const dx = nextX - state.pointer.x;
  const dy = nextY - state.pointer.y;
  state.pointer.lastX = state.pointer.x;
  state.pointer.lastY = state.pointer.y;
  state.pointer.x = Math.max(0, Math.min(handOverlay.width, nextX));
  state.pointer.y = Math.max(0, Math.min(handOverlay.height, nextY));
  state.pointer.speed = Math.hypot(dx, dy);
  state.pointer.source = source;
  syncPointerHitTarget();
}

function updatePointerFromEvent(event) {
  const rect = handOverlay.getBoundingClientRect();
  const x = ((event.clientX - rect.left) / rect.width) * handOverlay.width;
  const y = ((event.clientY - rect.top) / rect.height) * handOverlay.height;
  updatePointerPosition(x, y, 'mouse');
  handStatusEl.textContent = state.pointer.speed > 14 ? 'Mouse Swipe Simulated' : 'Mouse Tracking';
  trackerStatusTextEl.textContent = state.pointer.speed > 14 ? 'Mouse Swing Captured' : 'Mouse Tracking Active';
  renderSensorLayers();
}

async function setupCameraPreview() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    cameraStatusEl.textContent = 'Camera Unsupported';
    return;
  }

  try {
    state.cameraStream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 960 },
        height: { ideal: 540 },
        facingMode: 'user',
      },
      audio: false,
    });
    cameraFeed.srcObject = state.cameraStream;
    cameraPreview.srcObject = state.cameraStream;
    await cameraFeed.play().catch(() => {});
    await cameraPreview.play().catch(() => {});
    cameraStatusEl.textContent = 'Camera Live';
    trackerStatusTextEl.textContent = 'Camera Live · Waiting Hand';
  } catch {
    cameraStatusEl.textContent = 'Camera Blocked';
    trackerStatusTextEl.textContent = 'Camera Blocked · Mouse Fallback';
  }
}

async function initHandTracking() {
  if (!cameraFeed.srcObject) {
    handStatusEl.textContent = 'Waiting Camera';
    return;
  }

  try {
    const vision = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM);
    state.handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: HAND_MODEL,
        delegate: 'GPU',
      },
      runningMode: 'VIDEO',
      numHands: 1,
      minHandDetectionConfidence: 0.45,
      minHandPresenceConfidence: 0.45,
      minTrackingConfidence: 0.45,
    });
    state.handTrackingReady = true;
    handStatusEl.textContent = 'MediaPipe Ready';
    trackerStatusTextEl.textContent = 'Hand Tracker Ready';
    requestAnimationFrame(runHandTrackingLoop);
  } catch {
    handStatusEl.textContent = 'MediaPipe Load Failed';
    trackerStatusTextEl.textContent = 'Tracker Load Failed';
  }
}

function maybeTriggerHandStrike(landmarks) {
  if (state.status !== 'playing' || state.pointer.insideIndex === null) {
    return;
  }

  const indexTip = landmarks[8];
  const thumbTip = landmarks[4];
  if (!indexTip || !thumbTip) {
    return;
  }

  const pinchDistance = Math.hypot(indexTip.x - thumbTip.x, indexTip.y - thumbTip.y);
  const strikeReady = pinchDistance < 0.055 || state.pointer.speed > 26;

  if (strikeReady && state.pointer.pinchArmed) {
    strikeHole('human', state.pointer.insideIndex, pinchDistance < 0.055 ? 'pinch' : 'swipe');
    state.pointer.pinchArmed = false;
  }

  if (pinchDistance > 0.08) {
    state.pointer.pinchArmed = true;
  }
}

function runHandTrackingLoop() {
  if (!state.handLandmarker || !cameraFeed.videoWidth || !cameraFeed.videoHeight) {
    requestAnimationFrame(runHandTrackingLoop);
    return;
  }

  if (cameraFeed.currentTime === state.lastVideoTime || state.handTrackingBusy) {
    requestAnimationFrame(runHandTrackingLoop);
    return;
  }

  state.handTrackingBusy = true;
  state.lastVideoTime = cameraFeed.currentTime;

  const result = state.handLandmarker.detectForVideo(cameraFeed, performance.now());
  const landmarks = result.landmarks?.[0];
  if (landmarks) {
    state.lastHandLandmarks = landmarks;
    const palm = landmarks[9] || landmarks[0];
    updatePointerPosition((1 - palm.x) * handOverlay.width, palm.y * handOverlay.height, 'hand');
    maybeTriggerHandStrike(landmarks);
    handStatusEl.textContent = 'Hand Tracking Live';
    trackerStatusTextEl.textContent = state.pointer.speed > 18 ? 'Hand Swing Captured' : 'Hand Locked';
    renderSensorLayers(landmarks);
  } else {
    state.lastHandLandmarks = null;
    handStatusEl.textContent = state.handTrackingReady ? 'Show Hand To Camera' : 'Tracking Offline';
    trackerStatusTextEl.textContent = state.handTrackingReady ? 'Show Hand To Camera' : 'Tracking Offline';
    renderSensorLayers(null);
  }

  state.handTrackingBusy = false;
  requestAnimationFrame(runHandTrackingLoop);
}

function setAIDetections(detections, source = 'synthetic-yolo') {
  state.aiVisionBridge.detections = detections;
  state.aiVisionBridge.lastScanAt = performance.now();
  state.aiVisionBridge.mode = source === 'external-yolo' ? 'external-yolo' : 'synthetic-yolo';
  state.aiVisionBridge.state = source === 'external-yolo' ? 'receiving' : 'listening';

  if (!detections.length) {
    aiLastDetectionEl.textContent = 'None';
    updateHud();
    renderSensorLayers();
    return [];
  }

  const best = detections
    .filter((item) => item.label === 'mole')
    .sort((a, b) => a.expiresAt - b.expiresAt)[0] || detections[0];
  aiLastDetectionEl.textContent = `${best.label.toUpperCase()} @ H${best.holeIndex + 1}`;
  updateHud();
  renderSensorLayers();
  return detections;
}

function scanAIVisionTargets() {
  if (state.aiVisionBridge.lastExternalAt && performance.now() - state.aiVisionBridge.lastExternalAt < 450) {
    return state.aiVisionBridge.detections;
  }

  const detections = state.boards.ai.holes
    .filter((hole) => hole.active)
    .map((hole) => ({
      holeIndex: hole.index,
      label: hole.type === 'bomb' ? 'bomb' : 'mole',
      confidence: hole.type === 'bomb' ? 0.84 : 0.88 + Math.random() * 0.09,
      locked: hole.locked,
      expiresAt: hole.expiresAt,
    }));

  return setAIDetections(detections, 'synthetic-yolo');
}

function startAIVisionBridge() {
  stopAIVisionBridge();
  const loop = () => {
    if (state.status !== 'playing') {
      return;
    }
    scanAIVisionTargets();
    state.aiVisionBridge.scanTimer = window.setTimeout(loop, Math.max(120, 320 - state.difficulty * 12));
  };
  loop();
}

function stopAIVisionBridge() {
  if (state.aiVisionBridge.scanTimer) {
    window.clearTimeout(state.aiVisionBridge.scanTimer);
    state.aiVisionBridge.scanTimer = null;
  }
}

function getPreferredAIDetection() {
  const detections = state.aiVisionBridge.detections;
  if (!detections.length) {
    return null;
  }
  const safeMoles = detections.filter((item) => item.label === 'mole');
  return safeMoles.sort((a, b) => a.expiresAt - b.expiresAt)[0] || detections[0];
}

function mapExternalDetections(rawDetections) {
  const now = performance.now();
  return rawDetections
    .map((item) => {
      const holeIndex = Number(item.holeIndex);
      if (!Number.isInteger(holeIndex) || holeIndex < 0 || holeIndex >= HOLE_COUNT) {
        return null;
      }
      const boardHole = state.boards.ai.holes[holeIndex];
      return {
        holeIndex,
        label: item.label === 'bomb' ? 'bomb' : 'mole',
        confidence: Math.max(0, Math.min(1, Number(item.confidence ?? 0.9))),
        locked: boardHole?.locked || false,
        expiresAt: boardHole?.expiresAt || now + 300,
      };
    })
    .filter(Boolean);
}

function handleAIBridgeMessage(event) {
  const data = event.data;
  if (!data || typeof data !== 'object') {
    return;
  }

  if (data.type === 'YOLO_DETECTIONS' && Array.isArray(data.detections)) {
    state.aiVisionBridge.lastExternalAt = performance.now();
    state.aiVisionBridge.protocol = 'window.postMessage';
    const mapped = mapExternalDetections(data.detections);
    setAIDetections(mapped, 'external-yolo');
    return;
  }

  if (data.type === 'YOLO_BRIDGE_STATUS') {
    state.aiVisionBridge.protocol = data.protocol || 'window.postMessage';
    state.aiVisionBridge.state = data.state || 'listening';
    updateHud();
  }
}

function startHttpBridge() {
  if (!state.aiVisionBridge.httpBridge) {
    state.aiVisionBridge.httpBridge = createYoloHttpBridge({
      endpoint: YOLO_HTTP_ENDPOINT,
      pollMs: 250,
    });
  }
  state.aiVisionBridge.httpBridge.start();
}

function stopHttpBridge() {
  state.aiVisionBridge.httpBridge?.stop();
}

function publishBridgeContract() {
  window.MoleArenaAIBridge = {
    version: '0.2.0',
    protocol: 'window.postMessage',
    sendDetections(detections) {
      window.postMessage({ type: 'YOLO_DETECTIONS', detections }, '*');
    },
    setStatus(stateText, protocol = 'window.postMessage') {
      window.postMessage({ type: 'YOLO_BRIDGE_STATUS', state: stateText, protocol }, '*');
    },
    getSnapshot() {
      return {
        mode: state.aiVisionBridge.mode,
        status: state.status,
        difficulty: state.difficulty,
        detections: state.aiVisionBridge.detections,
      };
    },
  };
}

function resetMatch() {
  stopTimers();
  stopAIVisionBridge();
  pauseBgm();
  state.status = 'ready';
  state.durationSec = Number(durationSelect.value);
  state.difficulty = Number(difficultyRange.value);
  state.totalRounds = Number(roundsSelect.value);
  state.currentRound = 1;
  state.human.roundWins = 0;
  state.ai.roundWins = 0;
  resetRoundStats();
  aiVisionStatusEl.textContent = 'YOLO Bridge Ready';
  aiPolicyStatusEl.textContent = 'Auto Strike Demo';
  aiLastDetectionEl.textContent = 'None';
  state.aiVisionBridge.mode = 'synthetic-yolo';
  state.aiVisionBridge.protocol = 'window.postMessage + http-poll';
  state.aiVisionBridge.state = 'listening';
  state.aiVisionBridge.lastExternalAt = 0;
  setOverlay(
    'Prototype Ready',
    'Press Start',
    '这版已经接入真实摄像头和 MediaPipe 手部识别。左侧优先用手部控制，识别失败时也可以继续用鼠标模拟。'
  );
  updateHud();
  renderSensorLayers();
}

function startMatch() {
  ensureAudioReady();
  startBgm();
  if (state.status === 'playing') {
    return;
  }

  if (state.status === 'match_result') {
    resetMatch();
  }

  state.durationSec = Number(durationSelect.value);
  state.difficulty = Number(difficultyRange.value);
  state.totalRounds = Number(roundsSelect.value);
  state.status = 'countdown';
  updateHud();
  startCountdown(3);
}

function startCountdown(seconds) {
  let remaining = seconds;
  setOverlay('Get Ready', String(remaining), '双方正在锁定本局配置，马上开打。');
  const timer = setInterval(() => {
    remaining -= 1;
    if (remaining > 0) {
      setOverlay('Get Ready', String(remaining), '锁定镜头和 AI 目标中。');
      return;
    }

    clearInterval(timer);
    setOverlay('Round Live', 'Go!', '左侧可用手掌移动到地鼠上，捏合或快速挥动即可触发命中。', false);
    beginRound();
  }, 1000);
}

function beginRound() {
  state.status = 'playing';
  state.roundTimeLeftMs = state.durationSec * 1000;
  state.lastTickAt = performance.now();
  cameraStatusEl.textContent = state.cameraStream ? 'Camera Active' : 'Camera Missing';
  handStatusEl.textContent = state.handTrackingReady ? 'Hand Tracking Live' : 'Mouse Fallback';
  aiVisionStatusEl.textContent = 'YOLO Detecting';
  aiPolicyStatusEl.textContent = `Difficulty ${state.difficulty}`;
  aiDetectorModeEl.textContent = 'Synthetic YOLO';
  aiLastDetectionEl.textContent = 'Scanning';
  updateHud();
  scheduleSpawns();
  startAIVisionBridge();
  scheduleAI();
  renderSensorLayers();
  requestAnimationFrame(gameLoop);
}

function scheduleSpawns() {
  const nextDelay = spawnDelayMs();
  state.roundSpawnTimer = window.setTimeout(() => {
    if (state.status !== 'playing') {
      return;
    }
    spawnMole('human');
    spawnMole('ai');
    scheduleSpawns();
  }, nextDelay);
}

function spawnDelayMs() {
  return Math.max(380, 1200 - state.difficulty * 85 + Math.random() * 180);
}

function moleLifeMs() {
  return Math.max(650, 1700 - state.difficulty * 95 + Math.random() * 220);
}

function spawnMole(boardKey) {
  const board = state.boards[boardKey];
  const freeHoles = board.holes.filter((hole) => !hole.active);
  if (!freeHoles.length) {
    return;
  }

  const hole = freeHoles[Math.floor(Math.random() * freeHoles.length)];
  hole.active = true;
  hole.locked = false;
  hole.type = Math.random() < Math.min(0.12 + state.difficulty * 0.015, 0.28) ? 'bomb' : 'normal';
  hole.expiresAt = performance.now() + moleLifeMs();
  hole.element.classList.add('active');
  hole.element.querySelector('.mole').className = `mole ${hole.type}`;
  renderSensorLayers();
}

function despawnExpired(now) {
  ['human', 'ai'].forEach((boardKey) => {
    state.boards[boardKey].holes.forEach((hole) => {
      if (!hole.active || now < hole.expiresAt) {
        return;
      }
      hole.active = false;
      hole.locked = false;
      hole.element.classList.remove('active', 'ai-lock', 'hand-target');
      hole.element.querySelector('.mole').className = 'mole normal';
    });
  });
  renderSensorLayers();
}

function flashHole(element, outcome = 'good') {
  element.classList.add('hit-flash');
  element.classList.toggle('hit-success', outcome === 'good');
  element.classList.toggle('hit-bad', outcome === 'bad');
  window.setTimeout(() => {
    element.classList.remove('hit-flash', 'hit-success', 'hit-bad');
  }, 260);
}

function strikeHole(side, holeIndex, source) {
  if (side === 'human') {
    state.pointer.smashUntil = performance.now() + 120;
  }

  if (state.status !== 'playing') {
    return;
  }

  const board = state.boards[side];
  const hole = board.holes[holeIndex];
  const player = state[side];
  player.attempts += 1;

  if (!hole.active) {
    player.misses += 1;
    playHitSound('bad');
    flashHole(hole.element, 'bad');
    updateHud();
    renderSensorLayers();
    return;
  }

  const delta = hole.type === 'bomb' ? -1 : 1;
  if (delta > 0) {
    player.hits += 1;
  } else {
    player.misses += 1;
  }
  player.score = Math.max(0, player.score + delta);

  if (side === 'ai') {
    aiPolicyStatusEl.textContent = `Strike ${source}`;
  } else {
    handStatusEl.textContent = source === 'pinch' ? 'Pinch Hit' : source === 'swipe' ? 'Swipe Hit' : 'Tap Hit';
  }

  hole.active = false;
  hole.locked = false;
  hole.element.classList.remove('active', 'ai-lock', 'hand-target');
  hole.element.querySelector('.mole').className = 'mole normal';
  playHitSound(delta > 0 ? 'good' : 'bad');
  flashHole(hole.element, delta > 0 ? 'good' : 'bad');
  updateHud();
  renderSensorLayers();
}

function scheduleAI() {
  const thinkDelay = Math.max(180, 560 - state.difficulty * 28 + Math.random() * 80);
  state.aiThinkTimer = window.setTimeout(() => {
    if (state.status !== 'playing') {
      return;
    }

    const preferred = getPreferredAIDetection();
    const reactionBias = Math.min(0.42 + state.difficulty * 0.055, 0.93);

    if (preferred && preferred.label === 'mole' && Math.random() < reactionBias) {
      const target = state.boards.ai.holes[preferred.holeIndex];
      target.locked = true;
      target.element.classList.add('ai-lock');
      aiVisionStatusEl.textContent = `YOLO Target H${target.index + 1}`;
      aiLastDetectionEl.textContent = `${preferred.label.toUpperCase()} ${(preferred.confidence * 100).toFixed(0)}%`;
      renderSensorLayers();
      window.setTimeout(() => strikeHole('ai', target.index, 'YOLO'), Math.max(90, 320 - state.difficulty * 18));
    } else if (preferred && preferred.label === 'bomb' && Math.random() < 0.3) {
      aiVisionStatusEl.textContent = `Bomb Skip H${preferred.holeIndex + 1}`;
      aiPolicyStatusEl.textContent = 'Risk Rejected';
    } else {
      aiPolicyStatusEl.textContent = 'Scanning';
    }

    scheduleAI();
  }, thinkDelay);
}

function gameLoop(now) {
  if (state.status !== 'playing') {
    return;
  }

  const delta = now - state.lastTickAt;
  state.lastTickAt = now;
  state.roundTimeLeftMs -= delta;
  despawnExpired(now);
  updateHud();

  if (state.roundTimeLeftMs <= 0) {
    finishRound();
    return;
  }

  requestAnimationFrame(gameLoop);
}

function finishRound() {
  stopTimers();
  stopAIVisionBridge();
  clearAllBoards();
  state.status = 'round_result';

  const winner = state.human.score === state.ai.score ? 'draw' : state.human.score > state.ai.score ? 'human' : 'ai';
  if (winner === 'human') {
    state.human.roundWins += 1;
  } else if (winner === 'ai') {
    state.ai.roundWins += 1;
  }

  const record = {
    round: state.currentRound,
    durationSec: state.durationSec,
    difficulty: state.difficulty,
    humanScore: state.human.score,
    aiScore: state.ai.score,
    humanHits: state.human.hits,
    aiHits: state.ai.hits,
    winner,
    winnerLabel: winner === 'draw' ? 'Draw' : winner === 'human' ? 'Human Wins' : 'AI Wins',
    createdAt: new Date().toISOString(),
  };

  state.history.unshift(record);
  saveHistory();
  updateHud();

  const targetWins = Math.ceil(state.totalRounds / 2);
  const matchDone =
    state.currentRound >= state.totalRounds ||
    state.human.roundWins >= targetWins ||
    state.ai.roundWins >= targetWins;

  if (matchDone) {
    playRoundEndSound();
    pauseBgm();
    state.status = 'match_result';
    const finalWinner =
      state.human.roundWins === state.ai.roundWins
        ? 'Draw Match'
        : state.human.roundWins > state.ai.roundWins
          ? 'Human Wins Match'
          : 'AI Wins Match';
    setOverlay(
      'Match Finished',
      finalWinner,
      `本场最终比分 ${state.human.roundWins} : ${state.ai.roundWins}，最近一局 ${state.human.score} : ${state.ai.score}。可直接重新开始下一场。`
    );
    updateHud();
    renderSensorLayers();
    return;
  }

  setOverlay(
    'Round Complete',
    record.winnerLabel,
    `第 ${state.currentRound} 局结束，比分 ${state.human.score} : ${state.ai.score}。1.4 秒后自动进入下一局。`
  );

  window.setTimeout(() => {
    state.currentRound += 1;
    resetRoundStats();
    setOverlay('Next Round', `Round ${state.currentRound}`, '准备下一局。', false);
    beginRound();
  }, 1400);
}

function stopTimers() {
  if (state.roundSpawnTimer) {
    window.clearTimeout(state.roundSpawnTimer);
    state.roundSpawnTimer = null;
  }
  if (state.aiThinkTimer) {
    window.clearTimeout(state.aiThinkTimer);
    state.aiThinkTimer = null;
  }
}

function togglePause() {
  if (state.status === 'playing') {
    state.status = 'paused';
    stopTimers();
    stopAIVisionBridge();
    pauseBgm();
    setOverlay('Paused', 'Round Paused', '已冻结时间、地鼠刷新和 AI 行为。');
    updateHud();
    renderSensorLayers();
    return;
  }

  if (state.status === 'paused') {
    startBgm();
    state.status = 'playing';
    state.lastTickAt = performance.now();
    setOverlay('Round Live', 'Back To Game', '继续当前这一局。', false);
    scheduleSpawns();
    startAIVisionBridge();
    scheduleAI();
    updateHud();
    renderSensorLayers();
    requestAnimationFrame(gameLoop);
  }
}

startButton.addEventListener('click', startMatch);
pauseButton.addEventListener('click', togglePause);
resetButton.addEventListener('click', resetMatch);
difficultyRange.addEventListener('input', () => {
  state.difficulty = Number(difficultyRange.value);
  updateHud();
});
durationSelect.addEventListener('change', () => {
  state.durationSec = Number(durationSelect.value);
  if (state.status === 'ready') {
    state.roundTimeLeftMs = state.durationSec * 1000;
  }
  updateHud();
});
roundsSelect.addEventListener('change', () => {
  state.totalRounds = Number(roundsSelect.value);
  updateHud();
});
sensitivityRange.addEventListener('input', () => {
  state.handSensitivity = Number(sensitivityRange.value) / 10;
  syncPresetSelect();
  updateHud();
});
smoothingRange.addEventListener('input', () => {
  state.handSmoothing = Number(smoothingRange.value) / 10;
  syncPresetSelect();
  updateHud();
});
sensitivityPreset.addEventListener('change', () => {
  applyControlPreset(sensitivityPreset.value);
  updateHud();
});
document.addEventListener('keydown', (event) => {
  if (event.code === 'Space') {
    event.preventDefault();
    if (state.status === 'ready' || state.status === 'match_result') {
      startMatch();
      return;
    }
    togglePause();
  }

  if (event.key.toLowerCase() === 'r') {
    resetMatch();
  }
});

handOverlay.addEventListener('pointermove', updatePointerFromEvent);
humanBoard.addEventListener('pointermove', updatePointerFromEvent);
humanBoard.addEventListener('pointerdown', (event) => {
  updatePointerFromEvent(event);
  if (state.pointer.source === 'mouse' && state.pointer.insideIndex !== null) {
    strikeHole('human', state.pointer.insideIndex, state.pointer.speed > 14 ? 'swipe' : 'tap');
  }
});

async function boot() {
  window.addEventListener('message', handleAIBridgeMessage);
  publishBridgeContract();
  startHttpBridge();
  mountBoards();
  applyControlPreset(sensitivityPreset.value);
  await setupCameraPreview();
  await initHandTracking();
  resetMatch();
}

boot().catch((error) => {
  console.error('Boot failed:', error);
  cameraStatusEl.textContent = 'Boot Failed';
  handStatusEl.textContent = 'Check Console';
  resetMatch();
});
