// ===================================================
// HANDIT — 単語マスター
//
// 単語は Firestore の shared/catalog が本体。
// スタジオで追加すると、その場で全ユーザーに反映される。
//
// 下の WORDS は「たね」で、Firestore に繋がらないときの代わり。
//
//   word … 単語（AIのラベル名と完全に一致させること）
//   cat  … カテゴリ。学習マップの区切りと図鑑のタブを兼ねる
//   gif  … お手本のパス。無ければ null
//   desc … 手の動かし方の説明
//
// レベルという概念は廃止した。カテゴリだけで分類する。
// 並び順は CAT_ORDER で決める。
// ===================================================

const WORDS = [
  { word: 'おはよう',     cat: 'あいさつ', gif: null,                    desc: '右手をグーにして額の横に当て、前方へ下げる。' },
  { word: 'こんにちは',   cat: 'あいさつ', gif: 'assets/konnichiwa.gif', desc: '人差し指と中指を額に当て、前方へ下げる。' },
  { word: 'ありがとう',   cat: 'あいさつ', gif: null,                    desc: '右手を胸の前に出し、前方へ傾ける。' },
  { word: '好き',         cat: 'きもち',   gif: null,                    desc: '胸の前で右手をグーにして、ハートを作るように動かす。' },
  { word: '嫌い',         cat: 'きもち',   gif: null,                    desc: '手を払うように外側へ振る。' },
  { word: 'もう一度',     cat: '会話',     gif: null,                    desc: '片手を上下に動かして「繰り返す」を表す。' },
  { word: 'わかりました', cat: '会話',     gif: null,                    desc: '右手を額の横から前方に開く。' },
  { word: 'ゆっくり',     cat: '会話',     gif: null,                    desc: '両手をゆっくり前方に押し出す。' },
  { word: '私',           cat: '自己紹介', gif: null,                    desc: '右手の人差し指で自分の胸を指す。' },
  { word: '名前',         cat: '自己紹介', gif: null,                    desc: '両手の人差し指を交差させる。' },
];

// 学習マップに出す順番。ここに無いカテゴリは後ろに回る。
const CAT_ORDER = ['あいさつ', 'きもち', '会話', '自己紹介'];

// カテゴリのアイコン。決めていないカテゴリには下のプールから自動で割り当てる。
// 変えたいときはここに1行足す。
const CAT_ICONS = {
  'あいさつ':   '👋',
  'きもち':     '❤️',
  '会話':       '💬',
  '自己紹介':   '🙋',
  '食べもの':   '🍚',
  '数':         '🔢',
  '時間':       '🕐',
  '場所':       '📍',
  '家族':       '👨‍👩‍👧',
  '学校':       '🏫',
  '未分類':     '📘',
};
const ICON_POOL = ['🌱', '⭐', '🎈', '🍀', '🔷', '🎵', '🌙', '🔥', '🌊', '🎯'];

// ---------------------------------------------------
// 以下は自動生成。触らなくてよい。
// ---------------------------------------------------

// 決めていないカテゴリにも、名前から決まる同じアイコンが毎回付くようにする
function iconFor(cat) {
  if (CAT_ICONS[cat]) return CAT_ICONS[cat];
  let h = 0;
  for (let i = 0; i < cat.length; i++) h = (h * 31 + cat.charCodeAt(i)) >>> 0;
  return ICON_POOL[h % ICON_POOL.length];
}

function catRank(cat) {
  const i = CAT_ORDER.indexOf(cat);
  return i === -1 ? CAT_ORDER.length + 1 : i;
}

function buildGroups(words) {
  const by = new Map();
  for (const w of words) {
    const cat = w.cat || '未分類';
    if (!by.has(cat)) by.set(cat, []);
    by.get(cat).push(w);
  }
  return [...by.keys()]
    .sort((a, b) => catRank(a) - catRank(b) || a.localeCompare(b, 'ja'))
    .map(cat => {
      const items = by.get(cat);
      const videos = {};
      items.forEach(w => { if (w.gif) videos[w.word] = w.gif; });
      return { id: cat, title: cat, icon: iconFor(cat), words: items.map(w => w.word), videos };
    });
}

function shape(words) {
  const cats = [...new Set(words.map(w => w.cat || '未分類'))]
    .sort((a, b) => catRank(a) - catRank(b) || a.localeCompare(b, 'ja'));
  return {
    words,
    groups:     buildGroups(words),
    levels:     buildGroups(words),   // 旧名。既存の呼び出しのために残してある
    dictionary: [...words].sort((a, b) => catRank(a.cat) - catRank(b.cat)),
    categories: cats,
    iconFor,
    find: (word) => words.find(w => w.word === word) || null,
  };
}

const HANDIT_WORDS = shape(WORDS);
HANDIT_WORDS.seed = WORDS;
HANDIT_WORDS.catIcons = CAT_ICONS;

// Firestore の単語一覧で中身を差し替える。繋がらないときは上の WORDS のまま。
HANDIT_WORDS.load = async function () {
  try {
    const { fetchCatalog } = await import('./js/cloud.js');
    const cloud = await fetchCatalog(WORDS);
    if (Array.isArray(cloud) && cloud.length) Object.assign(HANDIT_WORDS, shape(cloud));
  } catch (e) {
    console.warn('クラウドの単語一覧を読めませんでした。手元の一覧を使います。', e);
  }
  return HANDIT_WORDS;
};

if (typeof window !== 'undefined') window.HANDIT_WORDS = HANDIT_WORDS;
export default HANDIT_WORDS;
export { WORDS, CAT_ICONS, CAT_ORDER };
