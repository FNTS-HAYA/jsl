// ===================================================
// HANDIT — 連続日数（ストリーク）
//
// レッスンで1つでも手話ができた日を「やった日」として数える。
//   ・昨日もやっていれば +1
//   ・今日すでに数えていれば変わらない
//   ・1日でも空いたら 1 からやり直し
//
// 日付は端末の時計（ローカル時間）で決める。日本なら0時に切り替わる。
// 保存先は localStorage（handit_streak）。ログイン中は users/{uid}.streak にも残す
// （Firestore への書き込みは learn.html 側で行う）。
// ===================================================

const KEY = 'handit_streak';

// 'YYYY-MM-DD'（ローカル時間）
export function dayKey(d = new Date()) {
  const p = n => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}`;
}

function daysBetween(a, b) {
  // 'YYYY-MM-DD' どうしの差。時差やサマータイムに左右されないよう正午で比べる
  const t = s => new Date(s + 'T12:00:00').getTime();
  return Math.round((t(b) - t(a)) / 86400000);
}

function clean(s) {
  if (!s || typeof s !== 'object') return { count: 0, best: 0, last: null };
  const count = Number.isFinite(s.count) && s.count > 0 ? Math.floor(s.count) : 0;
  const best  = Number.isFinite(s.best)  && s.best  > 0 ? Math.floor(s.best)  : 0;
  const last  = typeof s.last === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(s.last) ? s.last : null;
  return { count, best: Math.max(best, count), last };
}

// 保存されている生の値
export function getRaw() {
  try { return clean(JSON.parse(localStorage.getItem(KEY) || 'null')); }
  catch { return clean(null); }
}

function save(s) {
  try { localStorage.setItem(KEY, JSON.stringify(clean(s))); } catch {}
}

// 画面に出す状態
//   count   … 今の連続日数（途切れていたら 0）
//   best    … これまでの最高
//   doneToday … 今日もうやったか
//   atRisk  … 昨日まで続いていて、今日まだやっていない
export function getStreak(now = new Date()) {
  const s = getRaw();
  const today = dayKey(now);
  if (!s.last) return { count: 0, best: s.best, doneToday: false, atRisk: false };
  const gap = daysBetween(s.last, today);
  if (gap <= 0) return { count: s.count, best: s.best, doneToday: true,  atRisk: false };
  if (gap === 1) return { count: s.count, best: s.best, doneToday: false, atRisk: true };
  return { count: 0, best: s.best, doneToday: false, atRisk: false };
}

// 今日やったことを記録する。
// 戻り値 { count, best, increased }（increased: 今日はじめて数えたか）
export function recordToday(now = new Date()) {
  const s = getRaw();
  const today = dayKey(now);
  if (s.last === today) return { count: s.count, best: s.best, increased: false };
  const gap = s.last ? daysBetween(s.last, today) : Infinity;
  if (gap < 0) return { count: s.count, best: s.best, increased: false };   // 端末の時計が戻った
  const count = gap === 1 ? s.count + 1 : 1;
  const best = Math.max(s.best, count);
  save({ count, best, last: today });
  return { count, best, increased: true };
}

// アカウントの値と端末の値をまとめる（新しい日付を持つ方を優先）
export function merge(remote) {
  const a = getRaw(), b = clean(remote);
  let pick = a;
  if (b.last && (!a.last || b.last > a.last || (b.last === a.last && b.count > a.count))) pick = b;
  const merged = { ...pick, best: Math.max(a.best, b.best, pick.count) };
  save(merged);
  return merged;
}

// Firestore に送る形
export function payload() {
  const s = getRaw();
  return s.last ? { streak: s } : {};
}

export function reset() {
  try { localStorage.removeItem(KEY); } catch {}
}
