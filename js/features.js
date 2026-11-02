// ===================================================
// HANDIT — 特徴量ユーティリティ（全ページ共通）
//
// これまで app.js / learn.html / tutorial.html / collect_motion.html に
// 同じ handVec / distVec / buildVec がコピーされていたが、
// ここ1箇所に集約した。特徴量を変えるときはこのファイルだけ直す。
//
// 168次元ベクトルの内訳:
//   [  0.. 69) 右手  … 21点xyz(63) + 手の向き(3) + 手のひら法線(3)
//   [ 69..138) 左手  … 同上
//   [138..156) 顔6点のxyz(18)
//   [156..162) 右手首→顔6点の距離(6)
//   [162..168) 左手首→顔6点の距離(6)
// ===================================================

export const FEATURE_DIM  = 168;
export const HAND_DIM     = 69;
export const TARGET_FRAMES = 64;

export const FACE_KEYS   = ['nose', 'forehead', 'chin', 'left_eye', 'right_eye', 'mouth'];
export const FACE_POINTS = { nose: 1, forehead: 10, chin: 152, left_eye: 33, right_eye: 263, mouth: 13 };

// ベクトル内のブロック位置
export const SLICE = {
  RIGHT:  [0, 69],
  LEFT:   [69, 138],
  FACE:   [138, 156],
  RDIST:  [156, 162],
  LDIST:  [162, 168],
};

// FACE_KEYS のうち左右がある点（ミラー時に入れ替える）
const EYE_L = FACE_KEYS.indexOf('left_eye');
const EYE_R = FACE_KEYS.indexOf('right_eye');

// ---------------------------------------------------
// 片手 → 69次元
// ---------------------------------------------------
export function handVec(lm) {
  const v = [];
  for (const p of lm) v.push(p.x, p.y, p.z);

  // 手首(0) → 中指の付け根(9) の正規化方向ベクトル
  const w = lm[0], m = lm[9];
  const dx = m.x - w.x, dy = m.y - w.y, dz = m.z - w.z;
  const dl = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1;
  v.push(dx / dl, dy / dl, dz / dl);

  // 手のひらの法線（手首・人差し指付け根・小指付け根の外積）
  const p0 = lm[0], p1 = lm[5], p2 = lm[17];
  const u = [p1.x - p0.x, p1.y - p0.y, p1.z - p0.z];
  const t = [p2.x - p0.x, p2.y - p0.y, p2.z - p0.z];
  const nx = u[1] * t[2] - u[2] * t[1];
  const ny = u[2] * t[0] - u[0] * t[2];
  const nz = u[0] * t[1] - u[1] * t[0];
  const nl = Math.sqrt(nx * nx + ny * ny + nz * nz) || 1;
  v.push(nx / nl, ny / nl, nz / nl);

  return v;
}

// ---------------------------------------------------
// 手首 → 顔6点の距離 → 6次元
// ---------------------------------------------------
export function distVec(lm, face) {
  const w = lm[0];
  return FACE_KEYS.map(k => {
    if (!face || !face[k]) return 0;
    const p = face[k];
    const dx = w.x - p.x, dy = w.y - p.y, dz = w.z - p.z;
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
  });
}

// ---------------------------------------------------
// 1フレーム → 168次元
//   handsData: { Right: landmarks[], Left: landmarks[] }（無い手は省略可）
//   faceData:  { nose:{x,y,z}, ... } または null
// ---------------------------------------------------
export function buildVec(handsData, faceData) {
  const Z69 = new Array(69).fill(0);
  const Z18 = new Array(18).fill(0);
  const Z6  = new Array(6).fill(0);

  const r = handsData['Right'] ? handVec(handsData['Right']) : Z69.slice();
  const l = handsData['Left']  ? handVec(handsData['Left'])  : Z69.slice();

  const fv = Z18.slice();
  if (faceData) {
    FACE_KEYS.forEach((k, i) => {
      const p = faceData[k];
      if (p) { fv[i * 3] = p.x; fv[i * 3 + 1] = p.y; fv[i * 3 + 2] = p.z; }
    });
  }

  const rd = handsData['Right'] ? distVec(handsData['Right'], faceData) : Z6.slice();
  const ld = handsData['Left']  ? distVec(handsData['Left'],  faceData) : Z6.slice();

  return [...r, ...l, ...fv, ...rd, ...ld];
}

// ---------------------------------------------------
// 顔ランドマーク → 使う6点だけ抜き出す
// ---------------------------------------------------
export function pickFacePoints(faceLandmarks) {
  const out = {};
  for (const k in FACE_POINTS) {
    const p = faceLandmarks[FACE_POINTS[k]];
    out[k] = { x: p.x, y: p.y, z: p.z };
  }
  return out;
}

// ===================================================
// ミラー変換（左右反転）
//
// 手話は利き手が右でも左でも同じ意味になる。
// 学習時にこの変換でデータを2倍に増やすことで、
// どちらの手でやっても認識できるようになる。
//
// 画像を左右反転する（x → 1-x）と何が起きるか:
//   ・右手と左手が入れ替わる         → ブロックを swap
//   ・ランドマークの x が反転        → x = 1-x
//   ・向きベクトルは x成分だけ符号反転 → (-dx, dy, dz)
//   ・法線は外積なので y,z が符号反転  → (nx, -ny, -nz)
//   ・距離はスカラーなので値は不変    → ブロックを swap するだけ
//   ・顔の左目と右目も入れ替わる      → 3番目と4番目を swap
// ===================================================

function isZeroBlock(v, from, to) {
  for (let i = from; i < to; i++) if (v[i] !== 0) return false;
  return true;
}

// 69次元の片手ブロックをミラーする
function mirrorHandBlock(src, from, dst, dstFrom) {
  // 手が写っていないフレームはゼロのまま（1-0=1 にしてはいけない）
  if (isZeroBlock(src, from, from + HAND_DIM)) {
    for (let i = 0; i < HAND_DIM; i++) dst[dstFrom + i] = 0;
    return;
  }
  // 21点のランドマーク: x を反転
  for (let i = 0; i < 21; i++) {
    dst[dstFrom + i * 3]     = 1 - src[from + i * 3];
    dst[dstFrom + i * 3 + 1] =     src[from + i * 3 + 1];
    dst[dstFrom + i * 3 + 2] =     src[from + i * 3 + 2];
  }
  // 向きベクトル (dx, dy, dz) → (-dx, dy, dz)
  dst[dstFrom + 63] = -src[from + 63];
  dst[dstFrom + 64] =  src[from + 64];
  dst[dstFrom + 65] =  src[from + 65];
  // 法線 (nx, ny, nz) → (nx, -ny, -nz)
  dst[dstFrom + 66] =  src[from + 66];
  dst[dstFrom + 67] = -src[from + 67];
  dst[dstFrom + 68] = -src[from + 68];
}

// 168次元1フレームをミラーする
export function mirrorVec(v) {
  const out = new Array(FEATURE_DIM).fill(0);

  // 右手 → 左手スロット、左手 → 右手スロット
  mirrorHandBlock(v, SLICE.RIGHT[0], out, SLICE.LEFT[0]);
  mirrorHandBlock(v, SLICE.LEFT[0],  out, SLICE.RIGHT[0]);

  // 顔6点: x反転 + 左目/右目を入れ替え
  const fb = SLICE.FACE[0];
  if (!isZeroBlock(v, fb, fb + 18)) {
    for (let i = 0; i < 6; i++) {
      const src = (i === EYE_L) ? EYE_R : (i === EYE_R) ? EYE_L : i;
      out[fb + i * 3]     = 1 - v[fb + src * 3];
      out[fb + i * 3 + 1] =     v[fb + src * 3 + 1];
      out[fb + i * 3 + 2] =     v[fb + src * 3 + 2];
    }
  }

  // 距離: ブロックごと入れ替え + 左目/右目を入れ替え
  for (let i = 0; i < 6; i++) {
    const src = (i === EYE_L) ? EYE_R : (i === EYE_R) ? EYE_L : i;
    out[SLICE.LDIST[0]  + i] = v[SLICE.RDIST[0] + src];
    out[SLICE.RDIST[0]  + i] = v[SLICE.LDIST[0] + src];
  }

  return out;
}

// 1サンプル（64フレーム分）をまるごとミラーする
export function mirrorSequence(seq) {
  return seq.map(mirrorVec);
}

// ---------------------------------------------------
// 任意長のフレーム列を TARGET_FRAMES に揃える
// 収録の長さが多少ぶれても同じ形になる
// ---------------------------------------------------
export function resample(seq, n = TARGET_FRAMES) {
  if (seq.length === 0) return Array.from({ length: n }, () => new Array(FEATURE_DIM).fill(0));
  if (seq.length === n) return seq.map(f => f.slice());
  const out = [];
  for (let i = 0; i < n; i++) {
    const pos = (i * (seq.length - 1)) / (n - 1);
    const lo = Math.floor(pos), hi = Math.min(lo + 1, seq.length - 1);
    const t = pos - lo;
    const a = seq[lo], b = seq[hi];
    const f = new Array(FEATURE_DIM);
    for (let d = 0; d < FEATURE_DIM; d++) f[d] = a[d] * (1 - t) + b[d] * t;
    out.push(f);
  }
  return out;
}

// ===================================================
// 判定用のフレームのため方（時間で管理する）
//
// これまでは「直近64フレームがたまるまで判定しない」作りだった。
// そのせいで、手を上げてすぐ手話をすると、64フレーム（約2秒、
// 重いPCだと3〜4秒）たまる前に手話が終わってしまい、
// 最初の1回がほぼ認識されなかった。単語が変わるたびに
// ためた分を消すので、毎回の1回目が同じ目にあっていた。
//
// しかもスタジオは「2.2秒ぶんを64に縮める」収録なのに、判定側は
// 「64フレーム」で切っていたので、カメラが遅いPCでは時間の長さがずれていた。
//
// ここでは各フレームに時刻を付けてため、
//   ・2.2秒 / 1.54秒 / 1.1秒 の「時間」で切り出して64フレームにそろえる
//     （スタジオの収録とまったく同じ切り方。fpsに左右されない）
//   ・まだその長さぶんたまっていなければ、足りない前の部分を
//     「手が写っていないフレーム」で埋める（収録で手を入れる前と同じ状態）
// とすることで、手が写って約0.6秒後から判定を始められるようにした。
// ===================================================

export const RECORD_MS   = 2200;   // スタジオの1テイクの長さ（admin/studio.html と合わせる）
export const MIN_SPAN_MS = 600;    // これだけたまったら判定を始める

export class FrameWindow {
  constructor(keepMs = RECORD_MS + 400) {
    this.keepMs = keepMs;
    this.frames = [];               // [{ v: 168次元, t: ms }]
  }

  push(vec, t = performance.now()) {
    this.frames.push({ v: vec, t });
    const from = t - this.keepMs;
    let i = 0;
    while (i < this.frames.length - 1 && this.frames[i].t < from) i++;
    if (i) this.frames.splice(0, i);
  }

  clear() { this.frames = []; }
  get length() { return this.frames.length; }
  get span() {
    const f = this.frames;
    return f.length > 1 ? f[f.length - 1].t - f[0].t : 0;
  }

  // 判定を始めてよいか
  ready(minMs = MIN_SPAN_MS) { return this.frames.length >= 8 && this.span >= minMs; }

  // 直近 ms ミリ秒の手首の速さ（画面の幅=1 として 1秒あたり）。両手のうち速い方。
  // 手話が終わって手が止まったかを見るのに使う。手が写っていなければ 0。
  motion(ms = 250) {
    const f = this.frames;
    if (f.length < 2) return 0;
    const last = f[f.length - 1];
    let k = f.length - 2;
    while (k > 0 && last.t - f[k].t < ms) k--;
    const first = f[k], dt = (last.t - first.t) / 1000;
    if (dt <= 0) return 0;
    let best = 0;
    for (const off of [SLICE.RIGHT[0], SLICE.LEFT[0]]) {
      const a = first.v, b = last.v;
      // その手が両方のフレームに写っているときだけ（0,0 は「手なし」）
      if ((!a[off] && !a[off + 1]) || (!b[off] && !b[off + 1])) continue;
      const d = Math.hypot(b[off] - a[off], b[off + 1] - a[off + 1]);
      best = Math.max(best, d / dt);
    }
    return best;
  }

  // 直近 durMs ミリ秒を n フレームに切り出す（時刻で線形補間）。
  // 足りない前の部分は「手なし」のフレームで埋める。
  window(durMs = RECORD_MS, n = TARGET_FRAMES) {
    const f = this.frames;
    if (!f.length) return Array.from({ length: n }, () => new Array(FEATURE_DIM).fill(0));
    const end = f[f.length - 1].t, start = end - durMs;
    const pad = noHandFrame(f[0].v);
    const out = [];
    let j = 0;
    for (let i = 0; i < n; i++) {
      const t = n === 1 ? end : start + (durMs * i) / (n - 1);
      if (t < f[0].t) { out.push(pad.slice()); continue; }
      while (j < f.length - 2 && f[j + 1].t < t) j++;
      const a = f[j], b = f[Math.min(j + 1, f.length - 1)];
      const span = b.t - a.t;
      const k = span > 0 ? Math.min(1, Math.max(0, (t - a.t) / span)) : 0;
      const v = new Array(FEATURE_DIM);
      for (let d = 0; d < FEATURE_DIM; d++) v[d] = a.v[d] * (1 - k) + b.v[d] * k;
      out.push(v);
    }
    return out;
  }
}

// 顔の位置はそのままで、手の部分だけ0にしたフレーム
// （buildVec({}, face) と同じ形。手を入れる前の収録フレームに当たる）
function noHandFrame(v) {
  const out = v.slice();
  out.fill(0, SLICE.RIGHT[0], SLICE.LEFT[1]);
  out.fill(0, SLICE.RDIST[0], SLICE.LDIST[1]);
  return out;
}
