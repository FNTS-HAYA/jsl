// ===================================================
// HANDIT — 利き手の設定
//
// 認識モデルは学習時に左右反転データで2倍にしているので、
// 右手でも左手でも同じように判定できる（train.py / features.js の mirror）。
// だからここで変えるのは「見せ方」だけ：
//   ・お手本（GIF / 動画）を左利きなら左右反転して見せる
//   ・説明文の「右手」「左手」を入れ替える
//
// 保存先は localStorage（handit_hand）。ログイン中は users/{uid}.hand にも残す
// （Firestore への書き込みは learn.html 側で行う）。
// ===================================================

const HAND_KEY = 'handit_hand';
const TUT_KEY  = 'handit_tutorial_done';

export const HANDS = [
  { id: 'right', name: '右きき', icon: '🫱' },
  { id: 'left',  name: '左きき', icon: '🫲' },
];

// 未設定なら null（チュートリアルで聞く）
export function getHand() {
  try {
    const v = localStorage.getItem(HAND_KEY);
    return v === 'left' || v === 'right' ? v : null;
  } catch { return null; }
}

// 未設定でも表示には右ききを使う
export function handOrDefault() { return getHand() || 'right'; }

export function setHand(id) {
  if (id !== 'left' && id !== 'right') return;
  try { localStorage.setItem(HAND_KEY, id); } catch {}
  applyHand();
}

// <html data-hand="left"> を付ける。反転は style.css の .ref-media で行う。
export function applyHand() {
  document.documentElement.dataset.hand = handOrDefault();
}

// 説明文の「右手」「左手」を利き手に合わせて入れ替える
export function handText(s) {
  if (!s || handOrDefault() === 'right') return s;
  return s.replace(/右手|左手/g, m => (m === '右手' ? '左手' : '右手'));
}

export function tutorialDone() {
  try { return localStorage.getItem(TUT_KEY) === '1'; } catch { return false; }
}
export function setTutorialDone(done = true) {
  try {
    if (done) localStorage.setItem(TUT_KEY, '1');
    else localStorage.removeItem(TUT_KEY);
  } catch {}
}
