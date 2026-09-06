// ===================================================
// HANDIT — 単語マスター
//
// 単語は Firestore の shared/catalog が本体。
// スタジオで追加すると、その場で全ユーザーに反映される。
//
// 下の WORDS は「たね」で、次の役割を持つ。
//   ・Firestore に繋がらないときの代わり
//   ・catalog をまだ作っていないときの初期値
//
//   word  … 単語（AIのラベル名と完全に一致させること）
//   cat   … 図鑑のカテゴリ
//   level … 学習マップのレベル番号
//   gif   … お手本のパス。無ければ null
//   desc  … 手の動かし方の説明
// ===================================================

const WORDS = [
  // ---- Lv.1 あいさつ ----
  { word: 'おはよう',     cat: 'あいさつ', level: 1, gif: null,                       desc: '右手をグーにして額の横に当て、前方へ下げる。' },
  { word: 'こんにちは',   cat: 'あいさつ', level: 1, gif: 'assets/konnichiwa.gif',    desc: '人差し指と中指を額に当て、前方へ下げる。' },
  { word: 'ありがとう',   cat: 'あいさつ', level: 1, gif: null,                       desc: '右手を胸の前に出し、前方へ傾ける。' },

  // ---- Lv.2 きもち ----
  { word: '好き',         cat: 'きもち',   level: 2, gif: null,                       desc: '胸の前で右手をグーにして、ハートを作るように動かす。' },
  { word: '嫌い',         cat: 'きもち',   level: 2, gif: null,                       desc: '手を払うように外側へ振る。' },

  // ---- Lv.3 会話 ----
  { word: 'もう一度',     cat: '会話',     level: 3, gif: null,                       desc: '片手を上下に動かして「繰り返す」を表す。' },
  { word: 'わかりました', cat: '会話',     level: 3, gif: null,                       desc: '右手を額の横から前方に開く。' },
  { word: 'ゆっくり',     cat: '会話',     level: 3, gif: null,                       desc: '両手をゆっくり前方に押し出す。' },

  // ---- Lv.4 自己紹介 ----
  { word: '私',           cat: '自己紹介', level: 4, gif: null,                       desc: '右手の人差し指で自分の胸を指す。' },
  { word: '名前',         cat: '自己紹介', level: 4, gif: null,                       desc: '両手の人差し指を交差させる。' },
];

// レベルの見た目（words の level 番号と対応）
const LEVEL_META = {
  1: { title: 'あいさつ',   icon: '👋' },
  2: { title: 'きもち',     icon: '❤️' },
  3: { title: '会話',       icon: '💬' },
  4: { title: '自己紹介',   icon: '🙋' },
};

// ---------------------------------------------------
// 以下は自動生成。触らなくてよい。
// ---------------------------------------------------
function buildLevels(words) {
  const byLevel = new Map();
  for (const w of words) {
    const lv = w.level || 1;
    if (!byLevel.has(lv)) byLevel.set(lv, []);
    byLevel.get(lv).push(w);
  }
  return [...byLevel.keys()].sort((a, b) => a - b).map(id => {
    const items = byLevel.get(id);
    const meta  = LEVEL_META[id] || { title: `レベル${id}`, icon: '📘' };
    const videos = {};
    items.forEach(w => { if (w.gif) videos[w.word] = w.gif; });
    return { id, title: meta.title, icon: meta.icon, words: items.map(w => w.word), videos };
  });
}

function shape(words) {
  const cats = [...new Set(words.map(w => w.cat || '未分類'))];
  return {
    words,
    levels:     buildLevels(words),
    dictionary: [...words].sort((a, b) => cats.indexOf(a.cat) - cats.indexOf(b.cat)),
    categories: cats,
    find:       (word) => words.find(w => w.word === word) || null,
  };
}

const HANDIT_WORDS = shape(WORDS);
HANDIT_WORDS.seed = WORDS;
HANDIT_WORDS.levelMeta = LEVEL_META;

// Firestore の単語一覧で中身を差し替える。
// 繋がらないときは上の WORDS のまま動く。
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
export { WORDS, LEVEL_META };
