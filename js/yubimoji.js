// ===================================================
// HANDIT — 指文字（ゆびもじ）
//
// 指文字は、ひらがな1文字ずつを手の形で表すもの。
// 名前や、手話の単語がまだない言葉を伝えるときに使う。
//
// ■ AIに覚えさせるには
//   スタジオで、単語名を「あ」「い」…のように ひらがな1文字、
//   カテゴリを「指文字」にして収録する。
//   ふつうの単語と同じ仕組みで判定されるので、ほかに作業はいらない。
//   （学習マップには出さず、指文字タブにだけ出す）
//
// ■ お手本
//   assets/signs/あ.gif（または .mp4）のように置けば自動で出る。
//
// ■ 説明文
//   形の説明は、確かめられた母音だけ先に入れてある。
//   ほかは desc: null のまま。分かったものから埋めていく。
//   スタジオで説明を書いた場合はそちらが優先される。
// ===================================================

export const YUBI_CAT = '指文字';

// 50音表の並び。null は空きマス
export const ROWS = [
  { name: 'あ行', kana: ['あ', 'い', 'う', 'え', 'お'] },
  { name: 'か行', kana: ['か', 'き', 'く', 'け', 'こ'] },
  { name: 'さ行', kana: ['さ', 'し', 'す', 'せ', 'そ'] },
  { name: 'た行', kana: ['た', 'ち', 'つ', 'て', 'と'] },
  { name: 'な行', kana: ['な', 'に', 'ぬ', 'ね', 'の'] },
  { name: 'は行', kana: ['は', 'ひ', 'ふ', 'へ', 'ほ'] },
  { name: 'ま行', kana: ['ま', 'み', 'む', 'め', 'も'] },
  { name: 'や行', kana: ['や', null, 'ゆ', null, 'よ'] },
  { name: 'ら行', kana: ['ら', 'り', 'る', 'れ', 'ろ'] },
  { name: 'わ行', kana: ['わ', null, null, null, 'を'] },
  { name: 'ん',   kana: ['ん', null, null, null, null] },
];

export const ALL_KANA = ROWS.flatMap(r => r.kana).filter(Boolean);

const DESC = {
  'あ': '手をにぎり、親指だけを伸ばして人差し指の横にそえる。',
  'い': '手をにぎり、小指だけを立てる。',
  'う': '人差し指と中指をそろえて立て、ほかの指はにぎる。',
  'え': '4本の指を曲げて指先をそろえ、親指は手のひら側に折る。',
  'お': '指先を丸めて、手で「O」の形を作る。',
};

// 指文字1文字ぶんのデータ（図鑑やお手本の表示で使う形にそろえる）
export function yubiItem(kana) {
  return { word: kana, cat: YUBI_CAT, gif: null, desc: DESC[kana] || null };
}

// 濁音・半濁音・小さい字の動かし方
//   move … 動かし方。how … 表で見せる説明
export const MARKS = {
  dakuten:    { label: '濁音',     move: '手を横（体の外側）へ動かす' },
  handakuten: { label: '半濁音',   move: '手を上へ動かす' },
  small:      { label: '小さい字', move: '手を手前に引く' },
};
for (const m of Object.values(MARKS)) m.how = `清音の形のまま、${m.move}`;

const DAKU = {
  'が':'か','ぎ':'き','ぐ':'く','げ':'け','ご':'こ',
  'ざ':'さ','じ':'し','ず':'す','ぜ':'せ','ぞ':'そ',
  'だ':'た','ぢ':'ち','づ':'つ','で':'て','ど':'と',
  'ば':'は','び':'ひ','ぶ':'ふ','べ':'へ','ぼ':'ほ',
  'ゔ':'う',
};
const HANDAKU = { 'ぱ':'は','ぴ':'ひ','ぷ':'ふ','ぺ':'へ','ぽ':'ほ' };
const SMALL = {
  'ぁ':'あ','ぃ':'い','ぅ':'う','ぇ':'え','ぉ':'お',
  'ゃ':'や','ゅ':'ゆ','ょ':'よ','っ':'つ','ゎ':'わ',
};

// カタカナをひらがなにする
function toHira(s) {
  return s.replace(/[ァ-ヶ]/g, c => String.fromCharCode(c.charCodeAt(0) - 0x60));
}

// 名前などを指文字の並びに分ける。
// 戻り値: [{ char: 'じ', kana: 'し', mark: 'dakuten' }, ...]
//   kana … 判定に使う清音（AIのラベル）
//   mark … 動きの指示（無ければ null）
// 長音「ー」と、ひらがなにできない字は skipped に入れて知らせる。
export function spell(text) {
  const out = [], skipped = [];
  for (const raw of toHira((text || '').normalize('NFC').replace(/\s+/g, ''))) {
    if (ALL_KANA.includes(raw))  { out.push({ char: raw, kana: raw, mark: null }); continue; }
    if (DAKU[raw])               { out.push({ char: raw, kana: DAKU[raw], mark: 'dakuten' }); continue; }
    if (HANDAKU[raw])            { out.push({ char: raw, kana: HANDAKU[raw], mark: 'handakuten' }); continue; }
    if (SMALL[raw])              { out.push({ char: raw, kana: SMALL[raw], mark: 'small' }); continue; }
    if (raw === 'ゐ')            { out.push({ char: raw, kana: 'い', mark: null }); continue; }
    if (raw === 'ゑ')            { out.push({ char: raw, kana: 'え', mark: null }); continue; }
    skipped.push(raw);
  }
  return { letters: out, skipped };
}
