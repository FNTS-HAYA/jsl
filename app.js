// ===================================================
// HANDIT app.js — Transformer版
// ===================================================
// レベル・単語を追加するときはLEVELS配列だけ編集
//
// 動画ファイルの追加方法：
// videosフォルダを作って mp4ファイルを置く
// LEVELS の videos に { word: 'おはよう', src: 'videos/ohayou.mp4' } を追加
// ===================================================

const LEVELS = [
  {
    id: 1, title: 'あいさつ', icon: '👋', color: '#FFFBDD',
    words: ['おはよう', 'こんにちは', 'ありがとう'],
    videos: {
      // 'おはよう': 'videos/ohayou.mp4',  // ← 動画を追加するときはここに書く
      // 'こんにちは': 'videos/konnichiwa.mp4',
      // 'ありがとう': 'videos/arigatou.mp4',
    }
  },
  {
    id: 2, title: 'きもち', icon: '❤️', color: '#FFECEC',
    words: ['好き', '嫌い'],
    videos: {}
  },
  {
    id: 3, title: 'フレーズ', icon: '💬', color: '#E8F1FC',
    words: ['もう一度'],
    videos: {}
  },
];

const MODEL_PATH     = 'dataset/model_single.onnx';
const LABELS_PATH    = 'dataset/labels.json';
const TARGET_FRAMES  = 64;
const CONF_THRESHOLD = 0.70;
const HOLD_FRAMES    = 15;
const FREE_COOLDOWN  = 2000;

const FACE_KEYS   = ['nose','forehead','chin','left_eye','right_eye','mouth'];
const FACE_POINTS = { nose:1, forehead:10, chin:152, left_eye:33, right_eye:263, mouth:13 };

let ort_session = null;
let labels      = [];
let frameBuffer = [];
let currentHandsData = {};
let currentFaceData  = null;
let handsDetector = null;
let faceMesh      = null;
let mpCam         = null;
let inferring     = false;

// フェーズ管理
const PHASE = { LEARN: 'learn', REVIEW: 'review' };
let currentPhase   = PHASE.LEARN;
let currentLevel   = null;
let learnQueue     = [];  // 学習フェーズ（各単語1回）
let reviewQueue    = [];  // 復習フェーズ（全単語ランダム）
let currentChallenge = null;
let currentIdx     = 0;
let correctCount   = 0;
let answered       = false;
let holdCount      = 0;

let freeHoldCount = 0;
let lastAddedWord = null;
let lastAddedTime = 0;

function getClearedLevels() { return JSON.parse(localStorage.getItem('handit_cleared')||'[]'); }
function setClearedLevel(id) {
  const c=getClearedLevels();
  if(!c.includes(id)){c.push(id);localStorage.setItem('handit_cleared',JSON.stringify(c));}
}

// ===== 初期化 =====
async function init() {
  loading('モデルを読み込み中...');
  try {
    ort.env.wasm.wasmPaths='https://cdn.jsdelivr.net/npm/onnxruntime-web@1.16.3/dist/';
    ort_session=await ort.InferenceSession.create(MODEL_PATH);
    const ld=await fetch(LABELS_PATH).then(r=>r.json());
    labels=ld.labels;
  } catch(e) {
    loadingText('モデルの読み込みに失敗しました。dataset/フォルダを確認してください。');
    console.error(e); return;
  }
  loading('MediaPipeを初期化中...');
  handsDetector=new Hands({locateFile:f=>'https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4/'+f});
  handsDetector.setOptions({maxNumHands:2,modelComplexity:1,minDetectionConfidence:0.65,minTrackingConfidence:0.5});
  handsDetector.onResults(onHandResults);
  faceMesh=new FaceMesh({locateFile:f=>'https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4/'+f});
  faceMesh.setOptions({maxNumFaces:1,refineLandmarks:false,minDetectionConfidence:0.5,minTrackingConfidence:0.5});
  faceMesh.onResults(onFaceResults);

  const isLearn=document.getElementById('level-list');
  const isFree=document.getElementById('view-free');
  if(isLearn) buildLevelList();
  if(isFree)  startFreeMode();
  hideLoading();
}

// ===== レベル一覧 =====
function buildLevelList() {
  const cleared=getClearedLevels();
  const list=document.getElementById('level-list');
  list.innerHTML='';
  LEVELS.forEach((level,i)=>{
    const isCleared=cleared.includes(level.id);
    const isAvailable=i===0||cleared.includes(LEVELS[i-1].id);
    const hasWords=level.words.every(w=>labels.includes(w));
    const canPlay=isAvailable&&hasWords;
    const card=document.createElement('div');
    card.className='level-card '+(isCleared?'cleared':canPlay?'available':'locked');
    card.innerHTML=`
      <div class="level-num" style="${canPlay||isCleared?'background:'+level.color:''}">${level.id}</div>
      <div class="level-info">
        <div class="level-title">${level.icon} ${level.title}</div>
        <div class="level-words">${level.words.join(' · ')}</div>
      </div>
      <div class="level-status">${isCleared?'✅':canPlay?'›':'🔒'}</div>
    `;
    if(canPlay) card.addEventListener('click',()=>startLesson(level));
    list.appendChild(card);
  });
}

// ===== カメラ =====
async function startCam(videoId, canvasId) {
  const video=document.getElementById(videoId);
  const canvas=document.getElementById(canvasId);
  const stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:'user'}});
  video.srcObject=stream;
  await new Promise(r=>video.onloadedmetadata=r);
  canvas.width=video.videoWidth; canvas.height=video.videoHeight;
  let fc=0;
  mpCam=new Camera(video,{
    onFrame:async()=>{
      fc++;
      await handsDetector.send({image:video});
      if(fc%2===0) await faceMesh.send({image:video});
    },
    width:640, height:480
  });
  mpCam.start();
}

function stopCam(videoId) {
  if(mpCam){mpCam.stop();mpCam=null;}
  const v=document.getElementById(videoId);
  if(v&&v.srcObject){v.srcObject.getTracks().forEach(t=>t.stop());v.srcObject=null;}
  frameBuffer=[]; holdCount=0; freeHoldCount=0; inferring=false;
  currentHandsData={}; currentFaceData=null;
}

// ===== MediaPipe =====
function onFaceResults(results) {
  if(!results.multiFaceLandmarks||!results.multiFaceLandmarks.length){currentFaceData=null;return;}
  const lm=results.multiFaceLandmarks[0]; currentFaceData={};
  for(const key in FACE_POINTS){const i=FACE_POINTS[key];currentFaceData[key]={x:lm[i].x,y:lm[i].y,z:lm[i].z};}
}

function onHandResults(results) {
  const isLesson=!!document.getElementById('view-lesson')?.classList.contains('active');
  const isFree=!!document.getElementById('view-free')?.classList.contains('active');
  if(!isLesson&&!isFree) return;

  const cid=isLesson?'canvas-lesson':'canvas-free';
  const canvas=document.getElementById(cid); if(!canvas) return;
  const ctx=canvas.getContext('2d');
  ctx.clearRect(0,0,canvas.width,canvas.height);
  currentHandsData={};

  if(!results.multiHandLandmarks||!results.multiHandLandmarks.length){
    frameBuffer=[]; holdCount=0; freeHoldCount=0; return;
  }
  for(let i=0;i<results.multiHandLandmarks.length;i++){
    const lm=results.multiHandLandmarks[i],side=results.multiHandedness[i].label;
    currentHandsData[side]=lm;
    drawConnectors(ctx,lm,HAND_CONNECTIONS,{color:'rgba(255,255,255,0.3)',lineWidth:2});
    drawLandmarks(ctx,lm,{color:side==='Right'?'#FFD60A':'#CE82FF',lineWidth:1,radius:3});
  }
  if(currentFaceData){
    FACE_KEYS.forEach(k=>{
      const p=currentFaceData[k];if(!p)return;
      ctx.beginPath();ctx.arc(p.x*canvas.width,p.y*canvas.height,4,0,Math.PI*2);
      ctx.fillStyle='#0071E3';ctx.fill();
    });
  }

  frameBuffer.push(buildVec());
  if(frameBuffer.length>TARGET_FRAMES) frameBuffer.shift();
  if(frameBuffer.length<TARGET_FRAMES) return;

  if(isLesson&&!answered) runLessonInfer();
  if(isFree)              runFreeInfer();
}

// ===== 特徴量 =====
function handVec(lm){
  const v=[];for(const p of lm)v.push(p.x,p.y,p.z);
  const w=lm[0],m=lm[9],dx=m.x-w.x,dy=m.y-w.y,dz=m.z-w.z;
  const dl=Math.sqrt(dx*dx+dy*dy+dz*dz)||1;v.push(dx/dl,dy/dl,dz/dl);
  const p0=lm[0],p1=lm[5],p2=lm[17];
  const u=[p1.x-p0.x,p1.y-p0.y,p1.z-p0.z],t=[p2.x-p0.x,p2.y-p0.y,p2.z-p0.z];
  const nx=u[1]*t[2]-u[2]*t[1],ny=u[2]*t[0]-u[0]*t[2],nz=u[0]*t[1]-u[1]*t[0];
  const nl=Math.sqrt(nx*nx+ny*ny+nz*nz)||1;v.push(nx/nl,ny/nl,nz/nl);
  return v;
}
function distVec(lm,face){
  const w=lm[0];
  return FACE_KEYS.map(k=>{
    if(!face||!face[k])return 0;
    const p=face[k],dx=w.x-p.x,dy=w.y-p.y,dz=w.z-p.z;
    return Math.sqrt(dx*dx+dy*dy+dz*dz);
  });
}
function buildVec(){
  const Z69=new Array(69).fill(0),Z6=new Array(6).fill(0),Z18=new Array(18).fill(0);
  const r=currentHandsData['Right']?handVec(currentHandsData['Right']):Z69.slice();
  const l=currentHandsData['Left']?handVec(currentHandsData['Left']):Z69.slice();
  let fv=Z18.slice();
  if(currentFaceData)FACE_KEYS.forEach((k,i)=>{const p=currentFaceData[k];if(p){fv[i*3]=p.x;fv[i*3+1]=p.y;fv[i*3+2]=p.z;}});
  const rd=currentHandsData['Right']?distVec(currentHandsData['Right'],currentFaceData):Z6.slice();
  const ld=currentHandsData['Left']?distVec(currentHandsData['Left'],currentFaceData):Z6.slice();
  return [...r,...l,...fv,...rd,...ld];
}

// ===== 推論 =====
async function infer() {
  if(!ort_session) return null;
  const t=new ort.Tensor('float32',Float32Array.from(frameBuffer.flat()),[1,TARGET_FRAMES,168]);
  try{
    const out=await ort_session.run({input:t});
    const lg=Array.from(out.output.data);
    const mx=Math.max(...lg);
    const ex=lg.map(v=>Math.exp(v-mx));
    const sm=ex.reduce((a,b)=>a+b,0);
    const pb=ex.map(v=>v/sm);
    const i=pb.indexOf(Math.max(...pb));
    return{label:labels[i],conf:pb[i]};
  }catch(e){console.error(e);return null;}
}

async function runLessonInfer() {
  if(inferring) return;
  inferring=true;
  const res=await infer();
  inferring=false;
  if(!res) return;
  const{label,conf}=res;
  const ok=label===currentChallenge&&conf>=CONF_THRESHOLD;
  document.getElementById('lesson-dot').className='detect-dot'+(conf>=CONF_THRESHOLD?(ok?' active':' wrong'):'');
  document.getElementById('lesson-label').textContent=conf>=0.35?label:'手をカメラに向けてください';
  document.getElementById('lesson-conf').textContent=conf>=0.35?Math.round(conf*100)+'%':'';
  const hf=document.getElementById('hold-fill');
  if(hf) hf.style.width=(holdCount/HOLD_FRAMES*100)+'%';
  if(ok){holdCount++;if(holdCount>=HOLD_FRAMES){answered=true;correctCount++;showFeedback(true);}}
  else{holdCount=0;}
}

async function runFreeInfer() {
  if(inferring) return;
  inferring=true;
  const res=await infer();
  inferring=false;
  if(!res) return;
  const{label,conf}=res;
  document.getElementById('free-dot').className='detect-dot'+(conf>=CONF_THRESHOLD?' active':'');
  document.getElementById('free-label').textContent=conf>=0.35?label:'手をカメラに向けてください';
  document.getElementById('free-conf').textContent=conf>=0.35?Math.round(conf*100)+'%':'';
  if(conf>=CONF_THRESHOLD){
    freeHoldCount++;
    if(freeHoldCount>=HOLD_FRAMES){
      const now=Date.now();
      if(!(label===lastAddedWord&&now-lastAddedTime<FREE_COOLDOWN)){
        addWord(label);lastAddedWord=label;lastAddedTime=now;
      }
      freeHoldCount=0;
    }
  }else{freeHoldCount=0;}
}

// ===== レッスン =====
async function startLesson(level) {
  currentLevel  = level;
  currentPhase  = PHASE.LEARN;
  correctCount  = 0;
  answered      = false;
  holdCount     = 0;
  learnQueue    = [...level.words];  // 各単語1回
  reviewQueue   = [...level.words, ...level.words].sort(()=>Math.random()-0.5); // 復習は2回ランダム
  currentIdx    = 0;
  showView('view-lesson');
  await startCam('video-lesson','canvas-lesson');
  nextChallenge();
}

function nextChallenge() {
  const queue = currentPhase===PHASE.LEARN ? learnQueue : reviewQueue;

  if(currentIdx>=queue.length) {
    if(currentPhase===PHASE.LEARN) {
      // 学習フェーズ終了 → 復習フェーズへ
      currentPhase=PHASE.REVIEW;
      currentIdx=0;
      correctCount=0;
      document.getElementById('ls-phase-label').textContent='復習';
      document.getElementById('ls-prompt-label').textContent='復習テスト';
      nextChallenge();
      return;
    } else {
      // 復習フェーズ終了
      endLesson();
      return;
    }
  }

  currentChallenge = queue[currentIdx];
  answered=false; holdCount=0; frameBuffer=[];

  // お題更新
  document.getElementById('challenge-word').textContent=currentChallenge;
  document.getElementById('lesson-dot').className='detect-dot';
  document.getElementById('lesson-label').textContent='手をカメラに向けてください';
  document.getElementById('lesson-conf').textContent='';
  const hf=document.getElementById('hold-fill');if(hf)hf.style.width='0%';

  // プログレスバー
  const total=currentPhase===PHASE.LEARN?learnQueue.length:reviewQueue.length;
  const lp=document.getElementById('ls-progress');
  if(lp)lp.style.width=(currentIdx/total*100)+'%';

  // フェーズラベル
  const pl=document.getElementById('ls-phase-label');
  if(pl)pl.textContent=currentPhase===PHASE.LEARN?'学習':'復習';

  // 参考動画
  updateRefVideo(currentChallenge);

  currentIdx++;
}

function updateRefVideo(word) {
  const videoEl=document.getElementById('ref-video');
  const placeholder=document.getElementById('video-placeholder');
  if(!currentLevel||!currentLevel.videos) return;
  const src=currentLevel.videos[word];
  if(src){
    videoEl.src=src;
    videoEl.style.display='block';
    placeholder.style.display='none';
    videoEl.load();
  }else{
    videoEl.style.display='none';
    placeholder.style.display='flex';
  }
}

function showFeedback(correct) {
  const fb=document.getElementById('feedback');
  fb.className='feedback-overlay show '+(correct?'correct':'wrong');
  document.getElementById('fb-title').textContent=correct?'正解！ 🎉':'もう一度！ 💪';
  document.getElementById('fb-sub').textContent=correct
    ?`「${currentChallenge}」できました！`
    :`「${currentChallenge}」をもう一度試してみよう`;
}

async function endLesson() {
  stopCam('video-lesson');
  const xpGained=correctCount*10;
  if(currentLevel) {
    setClearedLevel(currentLevel.id);
    const uid=localStorage.getItem('handit_uid');
    if(uid){
      try{
        const{getFirestore,doc,getDoc,updateDoc,serverTimestamp}=await import("https://www.gstatic.com/firebasejs/10.12.0/firebase-firestore.js");
        const{initializeApp,getApps}=await import("https://www.gstatic.com/firebasejs/10.12.0/firebase-app.js");
        const apps=getApps();
        const app=apps.length?apps[0]:initializeApp({apiKey:"AIzaSyB-YZ16629_Eadt8mu2kxtZU5ehZ9-dsGA",authDomain:"handit-a5cfa.firebaseapp.com",projectId:"handit-a5cfa",storageBucket:"handit-a5cfa.firebasestorage.app",messagingSenderId:"398275846471",appId:"1:398275846471:web:eb7571dcd1e920964d2c62"});
        const db=getFirestore(app);
        const ref=doc(db,'users',uid);
        const snap=await getDoc(ref);
        const data=snap.exists()?snap.data():{xp:0,cleared:[]};
        const cleared=data.cleared||[];
        if(!cleared.includes(currentLevel.id))cleared.push(currentLevel.id);
        await updateDoc(ref,{xp:(data.xp||0)+xpGained,cleared,updatedAt:serverTimestamp()});
        localStorage.setItem('handit_xp',(data.xp||0)+xpGained);
        localStorage.setItem('handit_cleared',JSON.stringify(cleared));
      }catch(e){console.error('Firestore保存エラー:',e);}
    } else {
      const xp=parseInt(localStorage.getItem('handit_xp')||'0');
      localStorage.setItem('handit_xp',xp+xpGained);
    }
  }
  document.getElementById('c-correct').textContent=correctCount;
  document.getElementById('c-xp').textContent='+'+xpGained;
  document.getElementById('complete-sub').textContent=
    `復習テスト ${reviewQueue.length}問中${correctCount}問正解！`;
  showView('view-complete');
}

// ===== 自由モード =====
async function startFreeMode() {
  frameBuffer=[]; freeHoldCount=0; inferring=false;
  await startCam('video-free','canvas-free');
}

function addWord(word) {
  const box=document.getElementById('sentence-box');
  const ph=box.querySelector('.sentence-empty');if(ph)ph.remove();
  const el=document.createElement('span');
  el.className='sentence-word';el.textContent=word;box.appendChild(el);
}

// ===== ビュー =====
function showView(id) {
  const vl=document.getElementById('view-levels');
  if(vl)vl.style.display=id==='view-levels'?'block':'none';
  ['view-lesson','view-complete'].forEach(v=>{
    const el=document.getElementById(v);if(el)el.classList.toggle('active',v===id);
  });
}

// ===== イベント =====
const bl=document.getElementById('back-lesson');
if(bl)bl.addEventListener('click',()=>{
  stopCam('video-lesson');
  document.getElementById('feedback').className='feedback-overlay';
  showView('view-levels');buildLevelList();
});
const fb=document.getElementById('fb-btn');
if(fb)fb.addEventListener('click',()=>{
  document.getElementById('feedback').className='feedback-overlay';
  answered=false;holdCount=0;frameBuffer=[];nextChallenge();
});
const cb=document.getElementById('complete-back-btn');
if(cb)cb.addEventListener('click',e=>{
  e.preventDefault();showView('view-levels');buildLevelList();
});
const clr=document.getElementById('btn-clear');
if(clr)clr.addEventListener('click',()=>{
  document.getElementById('sentence-box').innerHTML='<span class="sentence-empty">手話をすると単語が並びます</span>';
  freeHoldCount=0;lastAddedWord=null;
});

// ===== ローディング =====
function loading(t){document.getElementById('loading').classList.remove('hidden');loadingText(t);}
function loadingText(t){document.getElementById('loading-text').textContent=t;}
function hideLoading(){document.getElementById('loading').classList.add('hidden');}

init();
