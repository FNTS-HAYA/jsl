// ===================================================
// HANDIT ベンチマーク用の計測パネル（6月版・今の版 共通）
//
// free.html の hands.js の直後に1行足すだけで動く:
//   <script src="bench.js"></script>
//
// 測れるもの
//   ・fps          … MediaPipe が1秒に何フレーム処理できているか
//   ・反応時間     … スペースキーを押してから単語が出るまでの秒数
//   ・判定の記録   … 出た単語を全部記録。正解/不正解は自分でボタンで付ける
//   ・誤認識       … 「誤認識テスト」中に出た単語の数（手話をしていないのに出たもの）
//
// 使い方は BENCHMARK.md を参照
// ===================================================
(function () {
  const VERSION = document.currentScript?.dataset.version || '不明';
  const log = [];             // {no, target, word, ms, ok}
  let target = '';
  let pressedAt = null;       // スペースを押した時刻
  let noiseMode = false, noiseStart = 0, noiseCount = 0, noiseLen = 0;

  // ---------- fps: MediaPipe の結果が返ってくる回数を数える ----------
  let frames = 0, fps = 0;
  function patchHands() {
    const H = window.Hands;
    if (!H || !H.prototype || H.prototype.__benchPatched) return !!(H && H.prototype.__benchPatched);
    const orig = H.prototype.onResults;
    H.prototype.onResults = function (cb) {
      return orig.call(this, (r) => { frames++; return cb(r); });
    };
    H.prototype.__benchPatched = true;
    return true;
  }
  if (!patchHands()) {
    const t = setInterval(() => { if (patchHands()) clearInterval(t); }, 20);
  }
  setInterval(() => { fps = frames; frames = 0; render(); }, 1000);

  // ---------- 単語が出たのを見張る ----------
  function watch() {
    const box = document.getElementById('sentence-box');
    if (!box) return setTimeout(watch, 300);
    new MutationObserver(muts => {
      for (const m of muts) for (const n of m.addedNodes) {
        if (!(n instanceof HTMLElement) || !n.classList.contains('sentence-word')) continue;
        onWord(n.textContent.trim());
      }
    }).observe(box, { childList: true });
  }
  watch();

  function onWord(word) {
    const now = performance.now();
    if (noiseMode) { noiseCount++; render(); return; }
    const ms = pressedAt == null ? null : Math.round(now - pressedAt);
    const ok = target ? (word === target) : null;
    log.push({ no: log.length + 1, target: target || '-', word, ms, ok });
    pressedAt = null;
    render();
  }

  // ---------- キー操作 ----------
  document.addEventListener('keydown', e => {
    if (e.target && ['INPUT', 'SELECT', 'TEXTAREA'].includes(e.target.tagName)) return;
    if (e.code === 'Space') { e.preventDefault(); pressedAt = performance.now(); render(); }
  });

  // ---------- パネル ----------
  const panel = document.createElement('div');
  panel.style.cssText = 'position:fixed;right:12px;bottom:12px;z-index:9999;width:300px;max-height:70vh;overflow:auto;' +
    'background:#111;color:#fff;font:12px/1.5 system-ui,sans-serif;border-radius:12px;padding:12px;box-shadow:0 6px 24px rgba(0,0,0,.35)';
  document.addEventListener('DOMContentLoaded', () => document.body.appendChild(panel));
  if (document.body) document.body.appendChild(panel);

  panel.addEventListener('click', e => {
    const a = e.target.dataset.act; if (!a) return;
    if (a === 'noise') {
      noiseMode = !noiseMode;
      if (noiseMode) { noiseStart = performance.now(); noiseCount = 0; }
      else noiseLen = (performance.now() - noiseStart) / 1000;
    }
    if (a === 'undo') log.pop();
    if (a === 'clear' && confirm('記録を全部消しますか？')) log.length = 0;
    if (a === 'csv') copyCsv();
    render();
  });

  function stats() {
    const judged = log.filter(r => r.ok !== null);
    const okN = judged.filter(r => r.ok).length;
    const times = log.filter(r => r.ms != null && r.ok !== false).map(r => r.ms).sort((a, b) => a - b);
    const med = times.length ? times[Math.floor(times.length / 2)] : null;
    return { judged: judged.length, okN, acc: judged.length ? Math.round(okN / judged.length * 100) : null, med };
  }

  // 骨組みは1回だけ作る（入力中のお題が毎秒消えないように）
  panel.innerHTML = `
    <div style="font-weight:700;font-size:13px;margin-bottom:6px">計測パネル（${VERSION}）</div>
    <div>fps: <b id="b-fps" style="font-size:16px">0</b></div>
    <div style="margin-top:8px">お題: <input id="bench-target" placeholder="例: おはよう"
      style="width:120px;border-radius:6px;border:0;padding:3px 6px"></div>
    <div id="b-state" style="margin-top:6px"></div>
    <div id="b-stats" style="margin-top:8px;border-top:1px solid #333;padding-top:8px"></div>
    <div style="margin-top:8px;border-top:1px solid #333;padding-top:8px">
      <button id="b-noise" data-act="noise"></button>
      <div id="b-noise-info"></div>
    </div>
    <div id="b-log" style="margin-top:8px;border-top:1px solid #333;padding-top:8px;max-height:160px;overflow:auto"></div>
    <div style="margin-top:8px;display:flex;gap:6px;flex-wrap:wrap">
      <button data-act="undo" style="${btn()}">1つ取り消す</button>
      <button data-act="csv" style="${btn()}">CSVをコピー</button>
      <button data-act="clear" style="${btn()}">全部消す</button>
    </div>`;
  const $ = id => panel.querySelector('#' + id);
  // お題を入れたら入力欄から抜ける（そのままだとスペースキーが効かない）
  $('bench-target').addEventListener('keydown', e => { if (e.key === 'Enter') e.target.blur(); });
  $('bench-target').addEventListener('blur', e => { target = e.target.value.trim(); render(); });

  function render() {
    const s = stats();
    const noiseSec = noiseMode ? (performance.now() - noiseStart) / 1000 : 0;
    $('b-fps').textContent = fps;
    $('b-state').style.color = pressedAt ? '#FFD60A' : '#888';
    $('b-state').textContent = pressedAt ? '● 計測中… 手話をしてください'
      : (target ? 'スペースを押してから手話を始める' : 'まずお題を入れてEnter');
    $('b-stats').innerHTML = `正解率: <b>${s.acc == null ? '—' : s.acc + '%'}</b>（${s.okN}/${s.judged}）<br>
      反応時間の中央値: <b>${s.med == null ? '—' : (s.med / 1000).toFixed(2) + '秒'}</b>`;
    $('b-noise').style.cssText = btn(noiseMode);
    $('b-noise').textContent = noiseMode ? '誤認識テストを終える' : '誤認識テストを始める';
    $('b-noise-info').textContent = noiseMode
      ? `手話をせずに普通に動く… ${Math.floor(noiseSec)}秒 / 誤認識 ${noiseCount}回`
      : (noiseStart ? `前回: ${Math.round(noiseLen)}秒で誤認識 ${noiseCount}回` : '');
    $('b-log').innerHTML = log.slice().reverse().map(r => `<div>${r.no}. お題:${r.target} → <b>${r.word}</b>
        ${r.ok === null ? '' : r.ok ? '<span style="color:#00C46A">○</span>' : '<span style="color:#FF3B30">×</span>'}
        ${r.ms == null ? '' : (r.ms / 1000).toFixed(2) + '秒'}</div>`).join('') || '<span style="color:#888">まだ記録なし</span>';
  }
  function btn(on) {
    return `border:0;border-radius:6px;padding:4px 8px;cursor:pointer;font:inherit;` +
      (on ? 'background:#FFD60A;color:#111' : 'background:#333;color:#fff');
  }

  function copyCsv() {
    const rows = [['版', '番号', 'お題', '出た単語', '正解', '反応時間(秒)']]
      .concat(log.map(r => [VERSION, r.no, r.target, r.word, r.ok === null ? '' : r.ok ? '○' : '×',
        r.ms == null ? '' : (r.ms / 1000).toFixed(2)]));
    const csv = rows.map(r => r.join(',')).join('\n');
    navigator.clipboard.writeText(csv).then(() => alert('コピーしました。スプレッドシートに貼り付けてください'),
      () => prompt('コピーできなかったので、ここから手でコピーしてください', csv));
  }

  window.handitBench = { log, stats };
  render();
})();
