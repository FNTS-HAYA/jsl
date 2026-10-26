// ===================================================
// HANDIT — お手本（GIF / 動画）の表示
//
// レッスン画面・図鑑タブ・図鑑ページ、全部この1つの関数で出す。
//
// お手本の場所は2通り：
//   1) 単語データの gif に書いたパス（例: assets/konnichiwa.gif）
//   2) 何も書いていなければ assets/signs/<単語>.gif を自動で探す
//
// つまり新しいお手本は assets/signs/ に「単語名.gif」で置くだけでよい。
// （例: assets/signs/ありがとう.gif）。Firestore やスタジオを触る必要はない。
// mp4 / webm を置きたいときは、単語データの gif にそのパスを書く。
// ===================================================

export const SIGN_DIR = 'assets/signs/';

// 単語データ（{ word, gif }）からお手本のURLを決める
export function refSrc(item, base = '') {
  if (!item) return null;
  if (item.gif) return /^(https?:)?\/\//.test(item.gif) ? item.gif : base + item.gif;
  return base + SIGN_DIR + encodeURIComponent(item.word) + '.gif';
}

const isVideo = src => /\.(mp4|webm|mov)(\?|$)/i.test(src || '');

// container の中身をお手本に差し替える。
// 見つからなければ placeholder を出す。
//   opts.placeholder … 無いときに出すHTML
//   opts.controls    … 動画にコントロールを付けるか
export function renderRef(container, item, opts = {}) {
  if (!container) return;
  const ph = opts.placeholder ??
    '<div class="ref-ph"><span>🤟</span><span>お手本準備中</span></div>';
  const src = refSrc(item, opts.base || '');
  if (!src) { container.innerHTML = ph; return; }

  const alt = item.word || '';
  let el;
  if (isVideo(src)) {
    el = document.createElement('video');
    Object.assign(el, { src, autoplay: true, loop: true, muted: true, playsInline: true, controls: !!opts.controls });
    el.setAttribute('playsinline', '');
  } else {
    el = document.createElement('img');
    el.src = src; el.alt = alt; el.loading = opts.lazy ? 'lazy' : 'eager';
  }
  el.className = 'ref-media ref-fill';
  el.addEventListener('error', () => { container.innerHTML = ph; }, { once: true });

  container.innerHTML = '';
  container.appendChild(el);
}
