// ===================================================
// HANDIT app.js
// ===================================================
// 単語・レッスンを追加するときはここを編集：
// LESSONS 配列に { id, title, icon, color, words } を追加するだけ。
// words に含まれる単語名が labels.json と一致していれば自動で有効になる。
// ===================================================

const LESSONS = [
  { id: 'l1', title: 'あいさつ',  icon: '👋', color: '#E6FAF2', words: ['おはよう', 'こんにちは', 'ありがとう'] },
  { id: 'l2', title: 'きもち',    icon: '❤️', color: '#FFECEC', words: ['好き', '嫌い'] },
  { id: 'l3', title: 'フレーズ',  icon: '💬', color: '#E8F1FC', words: ['もう一度'] },
  // ↑ここに単語を追加していく
];

// ===== 設定 =====
const MODEL_PATH      = 'dataset/model_single.onnx';
const LABELS_PATH     = 'dataset/labels.json';
const TARGET_FRAMES   = 64;
const CONF_THRESHOLD  = 0.65;
const HOLD_FRAMES     = 18;
const FREE_HOLD       = 12;
const FREE_COOLDOWN   = 1500;

const FACE_KEYS   = ['nose','forehead','chin','left_eye','right_eye','mouth'];
const FACE_POINTS = { nose:1, forehead:10, chin:152, left_eye:33, right_eye:263, mouth:13 };

// ===== 状態 =====
let ort_session = null;
let labels      = [];
let frameBuffer = [];
let currentHandsData = {};
let currentFaceData  = null;
let handsDetector = null;
let faceMesh      = null;
let mpCamLesson   = null;
let mpCamFree     = null;

let user     = JSON.parse(localStorage.getItem('handit_user') || 'null');
let totalXP  = parseInt(localStorage.getItem('handit_xp')     || '0');
let streak   = parseInt(localStorage.getItem('handit_streak') || '0');

let challengeQueue   = [];
let currentChallenge = null;
let currentIdx       = 0;
let holdCount        = 0;
let correctCount     = 0;
let answered         = false;
let freeHoldCount    = 0;
let lastAddedWord    = null;
let lastAddedTime    = 0;

// ===== 初期化 =====
async function init() {
  loading('モデルを読み込み中...');
  try {
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.16.3/dist/';
    ort_session = await ort.InferenceSession.create(MODEL_PATH);
    const ld = await fetch(LABELS_PATH).then(r => r.json());
    labels = ld.labels;
  } catch(e) {
    loadingText('モデルの読み込みに失敗しました。dataset/フォルダを確認してください。');
    console.error(e); return;
  }
  loading('MediaPipeを初期化中...');
  handsDetector = new Hands({ locateFile: f => 'https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4/' + f });
  handsDetector.setOptions({ maxNumHands:2, modelComplexity:1, minDetectionConfidence:0.65, minTrackingConfidence:0.5 });
  handsDetector.onResults(onHandResults);
  faceMesh = new FaceMesh({ locateFile: f => 'https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4/' + f });
  faceMesh.setOptions({ maxNumFaces:1, refineLandmarks:false, minDetectionConfidence:0.5, minTrackingConfidence:0.5 });
  faceMesh.onResults(onFaceResults);
  buildHomeView();
  hideLoading();
}

// ===== ホーム構築 =====
function buildHomeView() {
  document.getElementById('user-name').textContent  = user ? user.name : 'ゲスト';
  document.getElementById('user-sub').textContent   = 'JSL学習中';
  document.getElementById('stat-streak').textContent = streak;
  document.getElementById('stat-xp').textContent     = totalXP;
  const list = document.getElementById('lesson-list');
  list.innerHTML = '';
  LESSONS.forEach(lesson => {
    const ok = lesson.words.every(w => labels.includes(w));
    const el = document.createElement('div');
    el.className = 'lesson-item';
    el.innerHTML = `
      <div class="lesson-item-icon" style="background:${lesson.color}">${lesson.icon}</div>
      <div class="lesson-item-info">
        <div class="lesson-item-title">${lesson.title}</div>
        <div class="lesson-item-sub">${lesson.words.join(' · ')}</div>
      </div>
      <div class="lesson-item-badge ${ok?'':'locked'}">${ok?'スタート':'準備中'}</div>
    `;
    if (ok) el.addEventListener('click', () => startLesson(lesson));
    list.appendChild(el);
  });
}

// ===== カメラ =====
async function startCam(videoId, canvasId, cb) {
  const video = document.getElementById(videoId);
  const canvas = document.getElementById(canvasId);
  const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode:'user' } });
  video.srcObject = stream;
  await new Promise(r => video.onloadedmetadata = r);
  canvas.width = video.videoWidth; canvas.height = video.videoHeight;
  let fc = 0;
  const cam = new Camera(video, {
    onFrame: async () => {
      fc++;
      await handsDetector.send({ image: video });
      if (fc % 2 === 0) await faceMesh.send({ image: video });
    },
    width: 640, height: 480
  });
  cam.start();
  if (cb) cb(cam);
}

function stopCam(cam, videoId) {
  if (cam) cam.stop();
  const v = document.getElementById(videoId);
  if (v && v.srcObject) { v.srcObject.getTracks().forEach(t => t.stop()); v.srcObject = null; }
  frameBuffer = []; currentHandsData = {}; currentFaceData = null;
}

// ===== MediaPipe コールバック =====
function onFaceResults(results) {
  if (!results.multiFaceLandmarks || !results.multiFaceLandmarks.length) { currentFaceData = null; return; }
  const lm = results.multiFaceLandmarks[0];
  currentFaceData = {};
  for (const key in FACE_POINTS) { const i = FACE_POINTS[key]; currentFaceData[key] = { x:lm[i].x, y:lm[i].y, z:lm[i].z }; }
}

function onHandResults(results) {
  const isLesson  = document.getElementById('view-lesson').classList.contains('active');
  const isFree    = document.getElementById('view-free').classList.contains('active');
  if (!isLesson && !isFree) return;
  const cid = isLesson ? 'canvas-lesson' : 'canvas-free';
  const canvas = document.getElementById(cid);
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  currentHandsData = {};
  if (!results.multiHandLandmarks || !results.multiHandLandmarks.length) {
    frameBuffer = []; holdCount = 0; freeHoldCount = 0; return;
  }
  for (let i = 0; i < results.multiHandLandmarks.length; i++) {
    const lm = results.multiHandLandmarks[i];
    const side = results.multiHandedness[i].label;
    currentHandsData[side] = lm;
    drawConnectors(ctx, lm, HAND_CONNECTIONS, { color:'rgba(255,255,255,0.3)', lineWidth:2 });
    drawLandmarks(ctx, lm, { color: side==='Right'?'#00C46A':'#CE82FF', lineWidth:1, radius:3 });
  }
  if (currentFaceData) {
    FACE_KEYS.forEach(k => {
      const p = currentFaceData[k]; if (!p) return;
      ctx.beginPath(); ctx.arc(p.x*canvas.width, p.y*canvas.height, 4, 0, Math.PI*2);
      ctx.fillStyle = '#0071E3'; ctx.fill();
    });
  }
  frameBuffer.push(buildVec());
  if (frameBuffer.length > TARGET_FRAMES) frameBuffer.shift();
  if (frameBuffer.length === TARGET_FRAMES) {
    if (isLesson && !answered) runLessonInfer();
    if (isFree)                runFreeInfer();
  }
}

// ===== 特徴量 =====
function handVec(lm) {
  const v = [];
  for (const p of lm) v.push(p.x, p.y, p.z);
  const w=lm[0],m=lm[9], dx=m.x-w.x, dy=m.y-w.y, dz=m.z-w.z;
  const dl=Math.sqrt(dx*dx+dy*dy+dz*dz)||1;
  v.push(dx/dl,dy/dl,dz/dl);
  const p0=lm[0],p1=lm[5],p2=lm[17];
  const u=[p1.x-p0.x,p1.y-p0.y,p1.z-p0.z], t=[p2.x-p0.x,p2.y-p0.y,p2.z-p0.z];
  const nx=u[1]*t[2]-u[2]*t[1], ny=u[2]*t[0]-u[0]*t[2], nz=u[0]*t[1]-u[1]*t[0];
  const nl=Math.sqrt(nx*nx+ny*ny+nz*nz)||1;
  v.push(nx/nl,ny/nl,nz/nl);
  return v;
}
function distVec(lm, face) {
  const w = lm[0];
  return FACE_KEYS.map(k => {
    if (!face||!face[k]) return 0;
    const p=face[k], dx=w.x-p.x, dy=w.y-p.y, dz=w.z-p.z;
    return Math.sqrt(dx*dx+dy*dy+dz*dz);
  });
}
function buildVec() {
  const Z69=new Array(69).fill(0), Z6=new Array(6).fill(0), Z18=new Array(18).fill(0);
  const r = currentHandsData['Right'] ? handVec(currentHandsData['Right']) : Z69.slice();
  const l = currentHandsData['Left']  ? handVec(currentHandsData['Left'])  : Z69.slice();
  let fv = Z18.slice();
  if (currentFaceData) FACE_KEYS.forEach((k,i)=>{ const p=currentFaceData[k]; if(p){fv[i*3]=p.x;fv[i*3+1]=p.y;fv[i*3+2]=p.z;} });
  const rd = currentHandsData['Right'] ? distVec(currentHandsData['Right'],currentFaceData) : Z6.slice();
  const ld = currentHandsData['Left']  ? distVec(currentHandsData['Left'], currentFaceData) : Z6.slice();
  return [...r,...l,...fv,...rd,...ld];
}

// ===== 推論 =====
async function infer() {
  if (!ort_session) return null;
  const t = new ort.Tensor('float32', Float32Array.from(frameBuffer.flat()), [1,TARGET_FRAMES,168]);
  try {
    const out = await ort_session.run({ input: t });
    const lg  = Array.from(out.output.data);
    const mx  = Math.max(...lg);
    const ex  = lg.map(v => Math.exp(v-mx));
    const sm  = ex.reduce((a,b)=>a+b,0);
    const pb  = ex.map(v=>v/sm);
    const idx = pb.indexOf(Math.max(...pb));
    return { label: labels[idx], conf: pb[idx] };
  } catch(e) { console.error(e); return null; }
}

async function runLessonInfer() {
  const res = await infer(); if (!res) return;
  const { label, conf } = res;
  const ok = label === currentChallenge && conf >= CONF_THRESHOLD;
  document.getElementById('lesson-dot').className   = 'detect-dot' + (conf>=CONF_THRESHOLD?(ok?' active':' wrong'):'');
  document.getElementById('lesson-label').textContent = conf>=0.35 ? label : '手を検出中...';
  document.getElementById('lesson-conf').textContent  = conf>=0.35 ? Math.round(conf*100)+'%' : '';
  document.getElementById('hold-fill').style.width    = (holdCount/HOLD_FRAMES*100)+'%';
  if (ok) { holdCount++; if(holdCount>=HOLD_FRAMES){ answered=true; correctCount++; showFeedback(true); } }
  else    { holdCount=0; }
}

async function runFreeInfer() {
  const res = await infer(); if (!res) return;
  const { label, conf } = res;
  document.getElementById('free-dot').className    = 'detect-dot'+(conf>=CONF_THRESHOLD?' active':'');
  document.getElementById('free-label').textContent = conf>=0.35 ? label : '手を検出中...';
  document.getElementById('free-conf').textContent  = conf>=0.35 ? Math.round(conf*100)+'%' : '';
  if (conf >= CONF_THRESHOLD) {
    freeHoldCount++;
    if (freeHoldCount >= FREE_HOLD) {
      const now = Date.now();
      if (!(label===lastAddedWord && now-lastAddedTime<FREE_COOLDOWN)) {
        addWord(label); lastAddedWord=label; lastAddedTime=now;
      }
      freeHoldCount=0;
    }
  } else { freeHoldCount=0; }
}

function addWord(word) {
  const box = document.getElementById('sentence-box');
  const ph = box.querySelector('.sentence-empty'); if (ph) ph.remove();
  const el = document.createElement('span');
  el.className='sentence-word'; el.textContent=word; box.appendChild(el);
}

// ===== レッスン =====
async function startLesson(lesson) {
  correctCount=0; answered=false; holdCount=0;
  challengeQueue = [...lesson.words, ...lesson.words].sort(()=>Math.random()-0.5);
  currentIdx=0;
  showView('view-lesson');
  startCam('video-lesson','canvas-lesson', cam=>{ mpCamLesson=cam; });
  nextChallenge();
}

function nextChallenge() {
  if (currentIdx >= challengeQueue.length) { endLesson(); return; }
  currentChallenge = challengeQueue[currentIdx];
  frameBuffer=[]; holdCount=0; answered=false;
  document.getElementById('challenge-word').textContent = currentChallenge;
  document.getElementById('lesson-label').textContent   = '手をカメラに向けてください';
  document.getElementById('lesson-dot').className       = 'detect-dot';
  document.getElementById('lesson-conf').textContent    = '';
  document.getElementById('hold-fill').style.width      = '0%';
  document.getElementById('ls-progress').style.width    = (currentIdx/challengeQueue.length*100)+'%';
  currentIdx++;
}

function showFeedback(correct) {
  const fb = document.getElementById('feedback');
  fb.className = 'feedback-overlay show '+(correct?'correct':'wrong');
  document.getElementById('fb-title').textContent = correct?'素晴らしい！ 🎉':'もう一度！ 💪';
  document.getElementById('fb-sub').textContent   = correct?`「${currentChallenge}」正解！`:`「${currentChallenge}」をもう一度`;
}

function endLesson() {
  stopCam(mpCamLesson,'video-lesson'); mpCamLesson=null;
  streak++; totalXP += correctCount*10;
  localStorage.setItem('handit_streak', streak);
  localStorage.setItem('handit_xp', totalXP);
  document.getElementById('c-correct').textContent = correctCount;
  document.getElementById('c-xp').textContent      = '+' + correctCount*10;
  document.getElementById('c-streak').textContent  = streak;
  document.getElementById('complete-sub').textContent =
    correctCount===challengeQueue.length ? '全問正解！完璧です！' : `${challengeQueue.length}問中${correctCount}問正解！`;
  showView('view-complete');
}

// ===== 自由モード =====
function startFree() {
  freeHoldCount=0; lastAddedWord=null;
  const box = document.getElementById('sentence-box');
  box.innerHTML = '<span class="sentence-empty">手話をすると単語が並びます</span>';
  showView('view-free');
  startCam('video-free','canvas-free', cam=>{ mpCamFree=cam; });
}

// ===== ビュー切り替え =====
function showView(id) {
  ['view-home','view-lesson','view-free','view-complete'].forEach(v=>{
    const el = document.getElementById(v);
    if (!el) return;
    if (v==='view-home') { el.style.display = id==='view-home' ? 'block' : 'none'; }
    else { el.classList.toggle('active', v===id); }
  });
}

// ===== イベント =====
document.getElementById('btn-lesson-mode').addEventListener('click', ()=>{
  document.getElementById('lesson-list').scrollIntoView({ behavior:'smooth' });
});
document.getElementById('btn-free-mode').addEventListener('click', startFree);

document.getElementById('back-lesson').addEventListener('click', ()=>{
  stopCam(mpCamLesson,'video-lesson'); mpCamLesson=null;
  document.getElementById('feedback').className='feedback-overlay';
  showView('view-home'); buildHomeView();
});
document.getElementById('back-free').addEventListener('click', ()=>{
  stopCam(mpCamFree,'video-free'); mpCamFree=null;
  showView('view-home'); buildHomeView();
});
document.getElementById('fb-btn').addEventListener('click', ()=>{
  document.getElementById('feedback').className='feedback-overlay';
  answered=false; holdCount=0; frameBuffer=[]; nextChallenge();
});
document.getElementById('complete-home-btn').addEventListener('click', (e)=>{
  e.preventDefault(); showView('view-home'); buildHomeView();
});
document.getElementById('btn-clear').addEventListener('click', ()=>{
  const box = document.getElementById('sentence-box');
  box.innerHTML='<span class="sentence-empty">手話をすると単語が並びます</span>';
});

// ===== ローディング =====
function loading(t)    { document.getElementById('loading').classList.remove('hidden'); loadingText(t); }
function loadingText(t){ document.getElementById('loading-text').textContent=t; }
function hideLoading() { document.getElementById('loading').classList.add('hidden'); }

// ===== 起動 =====
init();
