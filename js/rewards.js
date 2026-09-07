// ===================================================
// HANDIT — XP で解放されるテーマと称号
//
// XP は貯まるだけで使い道が無かったので、
//   ・見た目のテーマ（配色）
//   ・名前の横に出る称号
// を解放できるようにした。
//
// 追加したいときは下の THEMES / TITLES に1行足すだけ。
// ===================================================

// ---------------------------------------------------
// テーマ（style.css の色をまるごと上書きする）
// ---------------------------------------------------
export const THEMES = [
  {
    id: 'default', name: 'たまご', xp: 0,
    swatch: ['#FFD60A', '#FAFAFA'],
    vars: {},                                  // style.css のまま
  },
  {
    id: 'sora', name: 'そら', xp: 100,
    swatch: ['#3B9DFF', '#F5FAFF'],
    vars: {
      '--yellow': '#3B9DFF', '--yellow-d': '#1B72C7', '--yellow-l': '#E8F3FF',
      '--bg': '#F5FAFF',
    },
  },
  {
    id: 'mori', name: 'もり', xp: 250,
    swatch: ['#35C46A', '#F4FBF6'],
    vars: {
      '--yellow': '#35C46A', '--yellow-d': '#1E9B4E', '--yellow-l': '#E7F8EE',
      '--bg': '#F4FBF6',
    },
  },
  {
    id: 'yuyake', name: 'ゆうやけ', xp: 500,
    swatch: ['#FF8A3D', '#FFF8F3'],
    vars: {
      '--yellow': '#FF8A3D', '--yellow-d': '#D2601C', '--yellow-l': '#FFF0E6',
      '--bg': '#FFF8F3',
    },
  },
  {
    id: 'sakura', name: 'さくら', xp: 800,
    swatch: ['#FF7EA8', '#FFF7FA'],
    vars: {
      '--yellow': '#FF7EA8', '--yellow-d': '#D6497A', '--yellow-l': '#FFEDF3',
      '--bg': '#FFF7FA',
    },
  },
  {
    id: 'yoru', name: 'よる', xp: 1500,
    swatch: ['#FFD60A', '#16161A'],
    vars: {
      '--yellow': '#FFD60A', '--yellow-d': '#C8A800', '--yellow-l': '#2A2820',
      '--ink': '#F2F2F0', '--ink-2': '#C9C9C5', '--ink-3': '#8B8B87',
      '--line': '#2E2E33', '--bg': '#16161A', '--white': '#1F1F24',
      '--green-l': '#123024', '--red-l': '#331B1B',
      '--shadow': '0 2px 14px rgba(0,0,0,0.45)',
      '--shadow-lg': '0 8px 32px rgba(0,0,0,0.55)',
    },
  },
];

// ---------------------------------------------------
// 称号
// ---------------------------------------------------
export const TITLES = [
  { id: 'start',   name: 'はじめの一歩',   xp: 0 },
  { id: 'rookie',  name: '手話みならい',   xp: 50 },
  { id: 'aisatsu', name: 'あいさつ名人',   xp: 150 },
  { id: 'talker',  name: '会話のたね',     xp: 300 },
  { id: 'hands',   name: '手が語る人',     xp: 600 },
  { id: 'fluent',  name: '手話つかい',     xp: 1000 },
  { id: 'master',  name: '手話マスター',   xp: 2000 },
];

const THEME_KEY = 'handit_theme';
const TITLE_KEY = 'handit_title';

// ---------------------------------------------------
export function getXP() {
  return parseInt(localStorage.getItem('handit_xp') || '0', 10) || 0;
}

export function unlockedThemes(xp = getXP()) { return THEMES.filter(t => xp >= t.xp); }
export function unlockedTitles(xp = getXP()) { return TITLES.filter(t => xp >= t.xp); }

// 次に解放されるもの（あと何XPかを見せるため）
export function nextUnlock(xp = getXP()) {
  const cands = [
    ...THEMES.filter(t => xp < t.xp).map(t => ({ kind: 'テーマ', ...t })),
    ...TITLES.filter(t => xp < t.xp).map(t => ({ kind: '称号',  ...t })),
  ].sort((a, b) => a.xp - b.xp);
  return cands[0] || null;
}

// ---------------------------------------------------
// 選択の保存と読み出し。持っていないものを選んでいたら既定に戻す。
// ---------------------------------------------------
export function currentTheme(xp = getXP()) {
  const id = localStorage.getItem(THEME_KEY) || 'default';
  const t = THEMES.find(x => x.id === id);
  return (t && xp >= t.xp) ? t : THEMES[0];
}

export function currentTitle(xp = getXP()) {
  const id = localStorage.getItem(TITLE_KEY) || 'start';
  const t = TITLES.find(x => x.id === id);
  return (t && xp >= t.xp) ? t : TITLES[0];
}

export function setTheme(id) {
  localStorage.setItem(THEME_KEY, id);
  applyTheme();
}

export function setTitle(id) {
  localStorage.setItem(TITLE_KEY, id);
}

// ---------------------------------------------------
// テーマを実際に適用する。
// :root の CSS 変数を書き換えるだけなので、
// style.css を触らずに全ページの見た目が変わる。
// どのページでも読み込み時に1回呼べばよい。
// ---------------------------------------------------
export function applyTheme() {
  const t = currentTheme();
  const root = document.documentElement;

  // 前のテーマの残りを消す
  for (const th of THEMES) {
    for (const k of Object.keys(th.vars)) root.style.removeProperty(k);
  }
  for (const [k, v] of Object.entries(t.vars)) root.style.setProperty(k, v);

  root.dataset.handitTheme = t.id;
  return t;
}
