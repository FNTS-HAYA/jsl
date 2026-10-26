// ===================================================
// HANDIT — 吹き出しチュートリアル（コーチマーク）
//
// 画面の一部だけを明るく残し、横に吹き出しを出して目線を誘導する。
// Life360 / クラシル / ZOZO のオンボーディングと同じ形。
//
//   await coachCard({ title, body, buttons })   … 画面中央のカード
//   await coachSpot({ target, title, body })    … 要素を照らして吹き出し
//   coachEnd()                                  … 全部消す
//
// どちらも Promise を返し、押されたボタンの value（スキップなら 'skip'）で解決する。
// waitClick: true にすると、照らした要素そのものを押すまで進まない。
// ===================================================

let root = null, hole = null, bubble = null, blockers = [];
let raf = 0, current = null;

function ensureDom() {
  if (root) return;
  root = document.createElement('div');
  root.className = 'coach-root';

  // 穴の周りの4枚。クリックを止めつつ、穴の中は押せるようにする。
  blockers = ['t', 'r', 'b', 'l'].map(() => {
    const d = document.createElement('div');
    d.className = 'coach-block';
    root.appendChild(d);
    return d;
  });

  hole = document.createElement('div');
  hole.className = 'coach-hole';
  root.appendChild(hole);

  bubble = document.createElement('div');
  bubble.className = 'coach-bubble';
  root.appendChild(bubble);

  document.body.appendChild(root);
}

function resolveTarget(t) {
  const el = typeof t === 'function' ? t() : (typeof t === 'string' ? document.querySelector(t) : t);
  return el && el.getClientRects().length ? el : null;
}

function clear() {
  cancelAnimationFrame(raf);
  if (current && current.cleanup) current.cleanup();
  current = null;
}

function render({ title, body, buttons, skip, step }) {
  const dots = step ? `<div class="coach-step">${step}</div>` : '';
  bubble.innerHTML = `
    ${dots}
    ${title ? `<div class="coach-title">${title}</div>` : ''}
    ${body  ? `<div class="coach-body">${body}</div>`   : ''}
    <div class="coach-actions">
      ${skip ? `<button class="coach-skip" data-v="skip">スキップ</button>` : '<span></span>'}
      <div class="coach-btns">
        ${(buttons || []).map((b, i) =>
          `<button class="coach-btn${b.primary ? ' primary' : ''}${b.selected ? ' selected' : ''}" data-i="${i}">${b.label}</button>`
        ).join('')}
      </div>
    </div>`;
}

function wire(buttons, resolve) {
  bubble.querySelectorAll('[data-i]').forEach(btn => {
    btn.addEventListener('click', () => resolve(buttons[+btn.dataset.i].value));
  });
  const sk = bubble.querySelector('[data-v="skip"]');
  if (sk) sk.addEventListener('click', () => resolve('skip'));
}

// ---------------------------------------------------
// 画面中央のカード（穴なし）
// ---------------------------------------------------
export function coachCard(opts) {
  ensureDom();
  clear();
  root.classList.add('show', 'card-mode');
  hole.style.display = 'none';
  blockers.forEach((b, i) => {
    Object.assign(b.style, i === 0
      ? { top: 0, left: 0, width: '100vw', height: '100vh', display: 'block' }
      : { display: 'none' });
  });
  render(opts);
  bubble.className = 'coach-bubble center' + (opts.wide ? ' wide' : '');
  bubble.style.left = bubble.style.top = '';
  bubble.style.removeProperty('--arrow-x');

  return new Promise(resolve => {
    current = {};
    wire(opts.buttons || [], v => { resolve(v); });
    if (opts.onRender) opts.onRender(bubble, resolve);
  });
}

// ---------------------------------------------------
// 要素を照らして吹き出し
// ---------------------------------------------------
export function coachSpot(opts) {
  ensureDom();
  clear();
  root.classList.add('show');
  root.classList.remove('card-mode');
  hole.style.display = 'block';
  blockers.forEach(b => (b.style.display = 'block'));
  render(opts);
  bubble.className = 'coach-bubble';

  const pad = opts.pad ?? 8;

  return new Promise(resolve => {
    let el = resolveTarget(opts.target);
    if (!el) { resolve('missing'); return; }
    el.scrollIntoView({ block: 'center', behavior: 'smooth' });

    let done = false;
    const finish = v => { if (done) return; done = true; clear(); resolve(v); };
    wire(opts.buttons || [], finish);

    // 照らした要素を押したら進む
    let onTargetClick = null;
    if (opts.waitClick) {
      hole.classList.add('pass');
      onTargetClick = () => finish('click');
      el.addEventListener('click', onTargetClick, { capture: true, once: true });
    } else {
      hole.classList.remove('pass');
    }

    current = {
      cleanup() {
        if (onTargetClick) el.removeEventListener('click', onTargetClick, { capture: true });
        hole.classList.remove('pass');
      },
    };

    // 位置は毎フレーム追いかける（スクロールや画面回転でずれないように）
    const tick = () => {
      const now = resolveTarget(opts.target);
      if (now) el = now;
      place(el.getBoundingClientRect(), pad);
      raf = requestAnimationFrame(tick);
    };
    tick();
  });
}

function place(r, pad) {
  const vw = window.innerWidth, vh = window.innerHeight;
  const x = Math.max(0, r.left - pad), y = Math.max(0, r.top - pad);
  const w = Math.min(vw, r.right + pad) - x, h = Math.min(vh, r.bottom + pad) - y;

  Object.assign(hole.style, { left: x + 'px', top: y + 'px', width: w + 'px', height: h + 'px' });

  const [t, rt, b, l] = blockers;
  Object.assign(t.style,  { left: 0, top: 0, width: vw + 'px', height: y + 'px' });
  Object.assign(b.style,  { left: 0, top: (y + h) + 'px', width: vw + 'px', height: Math.max(0, vh - y - h) + 'px' });
  Object.assign(l.style,  { left: 0, top: y + 'px', width: x + 'px', height: h + 'px' });
  Object.assign(rt.style, { left: (x + w) + 'px', top: y + 'px', width: Math.max(0, vw - x - w) + 'px', height: h + 'px' });

  // 吹き出し：下に入るなら下、入らなければ上。それも無理なら穴の中に重ねる。
  const bw = bubble.offsetWidth, bh = bubble.offsetHeight, gap = 14, m = 12;
  let top, dir;
  if (y + h + gap + bh + m <= vh)      { top = y + h + gap; dir = 'down'; }
  else if (y - gap - bh - m >= 0)      { top = y - gap - bh; dir = 'up'; }
  else                                 { top = vh - bh - m;  dir = 'none'; }
  const cx = x + w / 2;
  const left = Math.min(vw - bw - m, Math.max(m, cx - bw / 2));

  bubble.style.left = left + 'px';
  bubble.style.top  = top + 'px';
  bubble.style.setProperty('--arrow-x', Math.min(bw - 24, Math.max(24, cx - left)) + 'px');
  bubble.dataset.dir = dir;
}

export function coachEnd() {
  clear();
  if (root) root.classList.remove('show', 'card-mode');
}

export function coachActive() {
  return !!(root && root.classList.contains('show'));
}
