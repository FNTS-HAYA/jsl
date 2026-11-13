// ===================================================
// HANDIT — 計測パネル（URL に ?debug を付けたときだけ出る）
//
// 例: https://fnts-haya.github.io/jsl/learn.html?debug
//
// 右下に1秒ごとに出すもの:
//   GPU      … Chrome が実際に使っているグラボ（NVIDIA か内蔵か）
//   カメラ   … 実際に開いた解像度・fps・カメラ名
//   検出     … MediaPipe（手＋顔）を1秒に何回回したか / 1回あたりの平均と最大(ms)
//   判定     … AI（ONNX）の1回あたりの平均(ms)
//   画面     … 画面の描画回数/秒 と 一番長く固まった時間(ms)
// 何もしないページでは表示しないので、普段の動作には影響しない。
// ===================================================

const ON = /[?&]debug\b/.test(location.search);

const buckets = {};   // name -> { n, sum, max }
let camInfo = '';

export function perfMark(name, ms) {
  if (!ON) return;
  const b = buckets[name] || (buckets[name] = { n: 0, sum: 0, max: 0 });
  b.n++; b.sum += ms; if (ms > b.max) b.max = ms;
}

export function perfCamera(text) { camInfo = text; }

function gpuName() {
  try {
    const gl = document.createElement('canvas').getContext('webgl');
    const ext = gl && gl.getExtension('WEBGL_debug_renderer_info');
    return ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : '不明';
  } catch { return '不明'; }
}

if (ON) {
  const box = document.createElement('pre');
  Object.assign(box.style, {
    position: 'fixed', right: '8px', bottom: '8px', zIndex: 99999, margin: 0,
    background: 'rgba(0,0,0,0.82)', color: '#9f9', font: '12px/1.5 monospace',
    padding: '8px 10px', borderRadius: '8px', pointerEvents: 'none', whiteSpace: 'pre',
  });
  const gpu = gpuName();
  const start = () => document.body.appendChild(box);
  if (document.body) start(); else addEventListener('DOMContentLoaded', start);

  // 画面の描画（固まっていないか）
  let frames = 0, last = performance.now(), worst = 0;
  const raf = t => { frames++; worst = Math.max(worst, t - last); last = t; requestAnimationFrame(raf); };
  requestAnimationFrame(raf);

  const fmt = (name, label) => {
    const b = buckets[name];
    if (!b || !b.n) return `${label}: -`;
    return `${label}: ${b.n}回/秒  平均${(b.sum / b.n).toFixed(0)}ms  最大${b.max.toFixed(0)}ms`;
  };

  setInterval(() => {
    box.textContent = [
      `GPU : ${gpu}`,
      `カメラ: ${camInfo || '-'}`,
      fmt('frame', '検出'),
      fmt('infer', '判定'),
      `画面: ${frames}fps  最長の固まり${worst.toFixed(0)}ms`,
    ].join('\n');
    for (const k in buckets) buckets[k] = { n: 0, sum: 0, max: 0 };
    frames = 0; worst = 0;
  }, 1000);
}
