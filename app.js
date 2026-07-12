// ===================================================
// HANDIT app.js — 自由手話モード（free.html）専用
// ===================================================

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

let freeHoldCount = 0;
let lastAddedWord = null;
let lastAddedTime = 0;

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

  await startFreeMode();
  hideLoading();
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

// ===== MediaPipe =====
function onFaceResults(results) {
  if(!results.multiFaceLandmarks||!results.multiFaceLandmarks.length){currentFaceData=null;return;}
  const lm=results.multiFaceLandmarks[0]; currentFaceData={};
  for(const key in FACE_POINTS){const i=FACE_POINTS[key];currentFaceData[key]={x:lm[i].x,y:lm[i].y,z:lm[i].z};}
}

function onHandResults(results) {
  const canvas=document.getElementById('canvas-free'); if(!canvas) return;
  const ctx=canvas.getContext('2d');
  ctx.clearRect(0,0,canvas.width,canvas.height);
  currentHandsData={};

  if(!results.multiHandLandmarks||!results.multiHandLandmarks.length){
    frameBuffer=[]; freeHoldCount=0; return;
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

  runFreeInfer();
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

// ===== イベント =====
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
