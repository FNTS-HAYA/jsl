// ===================================================
// HANDIT app.js — 自由手話モード（free.html）専用
//
// 特徴量の計算と推論は js/features.js と js/recognizer.js に移した。
// このファイルは画面まわりだけを担当する。
// ※ free.html 側は <script type="module" src="app.js"> にすること。
// ===================================================

import { buildVec, pickFacePoints, FACE_KEYS, TARGET_FRAMES } from './js/features.js';
import { Recognizer } from './js/recognizer.js';

const HOLD_FRAMES   = 15;     // これだけ連続で確信できたら単語を確定
const FREE_COOLDOWN = 2000;   // 同じ単語が連続で並ぶのを防ぐ

let recognizer = null;
let frameBuffer = [];
let currentHands = {}, currentFace = null;
let handsDetector = null, faceMesh = null, mpCam = null;
let holdCount = 0, lastWord = null, lastTime = 0;

// ===== 初期化 =====
async function init() {
  loading('AIを読み込み中...');
  recognizer = new Recognizer();
  try {
    await recognizer.load();
  } catch (e) {
    console.error(e);
    loadingText('AIを読み込めませんでした。dataset/ フォルダを確認してください。');
    return;
  }
  // 学習済みになった単語のプロトタイプは要らないので掃除しておく
  recognizer.prunePromoted();

  loading('カメラを準備中...');
  handsDetector = new Hands({ locateFile: f => 'https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4/' + f });
  handsDetector.setOptions({ maxNumHands: 2, modelComplexity: 1, minDetectionConfidence: 0.65, minTrackingConfidence: 0.5 });
  handsDetector.onResults(onHandResults);

  faceMesh = new FaceMesh({ locateFile: f => 'https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4/' + f });
  faceMesh.setOptions({ maxNumFaces: 1, refineLandmarks: false, minDetectionConfidence: 0.5, minTrackingConfidence: 0.5 });
  faceMesh.onResults(r => {
    if (!r.multiFaceLandmarks || !r.multiFaceLandmarks.length) { currentFace = null; return; }
    currentFace = pickFacePoints(r.multiFaceLandmarks[0]);
  });

  await startCam();
  hideLoading();

  // コンソールから中身を見られるようにしておく（調査用）
  //   handit.recognizer.lastProto  … プロトタイプの類似度
  //   handit.recognizer.labels     … 学習済みの単語
  window.handit = { recognizer };
}

// ===== カメラ =====
async function startCam() {
  const video  = document.getElementById('video-free');
  const canvas = document.getElementById('canvas-free');
  const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' } });
  video.srcObject = stream;
  await new Promise(r => video.onloadedmetadata = r);
  canvas.width = video.videoWidth; canvas.height = video.videoHeight;

  let fc = 0;
  mpCam = new Camera(video, {
    onFrame: async () => {
      fc++;
      await handsDetector.send({ image: video });
      if (fc % 2 === 0) await faceMesh.send({ image: video });
    },
    width: 640, height: 480
  });
  mpCam.start();
}

// ===== 毎フレーム =====
function onHandResults(results) {
  const canvas = document.getElementById('canvas-free');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  currentHands = {};

  if (!results.multiHandLandmarks || !results.multiHandLandmarks.length) {
    frameBuffer = []; holdCount = 0;
    return;
  }

  for (let i = 0; i < results.multiHandLandmarks.length; i++) {
    const lm = results.multiHandLandmarks[i], side = results.multiHandedness[i].label;
    currentHands[side] = lm;
    drawConnectors(ctx, lm, HAND_CONNECTIONS, { color: 'rgba(255,255,255,0.3)', lineWidth: 2 });
    drawLandmarks(ctx, lm, { color: side === 'Right' ? '#FFD60A' : '#CE82FF', lineWidth: 1, radius: 3 });
  }
  if (currentFace) {
    FACE_KEYS.forEach(k => {
      const p = currentFace[k]; if (!p) return;
      ctx.beginPath(); ctx.arc(p.x * canvas.width, p.y * canvas.height, 4, 0, Math.PI * 2);
      ctx.fillStyle = '#0071E3'; ctx.fill();
    });
  }

  frameBuffer.push(buildVec(currentHands, currentFace));
  if (frameBuffer.length > TARGET_FRAMES) frameBuffer.shift();
  if (frameBuffer.length < TARGET_FRAMES) return;

  runInfer();
}

// ===== 推論 =====
async function runInfer() {
  const res = await recognizer.classify(frameBuffer);
  if (!res) return;
  const { label, conf } = res;
  const strong = conf >= recognizer.confThreshold;

  document.getElementById('free-dot').className   = 'detect-dot' + (strong ? ' active' : '');
  document.getElementById('free-label').textContent = conf >= 0.35 ? label : '手をカメラに向けてください';
  document.getElementById('free-conf').textContent  = conf >= 0.35 ? Math.round(conf * 100) + '%' : '';

  if (!strong) { holdCount = 0; return; }

  holdCount++;
  if (holdCount < HOLD_FRAMES) return;
  holdCount = 0;

  const now = Date.now();
  if (label === lastWord && now - lastTime < FREE_COOLDOWN) return;
  addWord(label);
  lastWord = label; lastTime = now;
}

// ===== 文を組み立てる =====
function addWord(word) {
  const box = document.getElementById('sentence-box');
  const ph = box.querySelector('.sentence-empty');
  if (ph) ph.remove();
  const el = document.createElement('span');
  el.className = 'sentence-word';
  el.textContent = word;
  box.appendChild(el);
}

const clr = document.getElementById('btn-clear');
if (clr) clr.addEventListener('click', () => {
  document.getElementById('sentence-box').innerHTML =
    '<span class="sentence-empty">手話をすると単語が並びます</span>';
  holdCount = 0; lastWord = null;
});

// ===== ローディング =====
function loading(t)     { document.getElementById('loading').classList.remove('hidden'); loadingText(t); }
function loadingText(t) { document.getElementById('loading-text').textContent = t; }
function hideLoading()  { document.getElementById('loading').classList.add('hidden'); }

init();
