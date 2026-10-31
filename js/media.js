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
const isGif   = src => /\.gif(\?|$)/i.test(src || '');

// GIF は再生するだけでCPUをかなり使う（16MBのGIFでファンが回っていた）。
// 同じ名前の .mp4 があればそちらを優先し、無ければGIFに戻す。
//   例: assets/konnichiwa.gif → まず assets/konnichiwa.mp4 を試す
// GIFから変換するには:
//   ffmpeg -i in.gif -movflags +faststart -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" out.mp4
const mp4Of = src => src.replace(/\.gif(\?|$)/i, '.mp4$1');

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
  const makeVideo = (vsrc, onFail) => {
    const v = document.createElement('video');
    Object.assign(v, { src: vsrc, autoplay: true, loop: true, muted: true, playsInline: true, controls: !!opts.controls });
    v.setAttribute('playsinline', '');
    v.setAttribute('aria-label', alt);
    v.className = 'ref-media ref-fill';
    v.addEventListener('error', onFail, { once: true });
    return v;
  };
  const makeImg = (isrc) => {
    const img = document.createElement('img');
    img.src = isrc; img.alt = alt; img.loading = opts.lazy ? 'lazy' : 'eager';
    img.className = 'ref-media ref-fill';
    img.addEventListener('error', () => { container.innerHTML = ph; }, { once: true });
    return img;
  };

  let el;
  if (isVideo(src)) {
    el = makeVideo(src, () => {
      // mp4 が無ければ、同じ名前の GIF を試す（Firestore に古いパスが残っていても動くように）
      const gif = src.replace(/\.(mp4|webm|mov)(\?|$)/i, '.gif$2');
      if (gif !== src) { container.innerHTML = ''; container.appendChild(makeImg(gif)); }
      else container.innerHTML = ph;
    });
  } else if (isGif(src)) {
    el = makeVideo(mp4Of(src), () => { container.innerHTML = ''; container.appendChild(makeImg(src)); });
  } else {
    el = makeImg(src);
  }

  container.innerHTML = '';
  container.appendChild(el);
}
