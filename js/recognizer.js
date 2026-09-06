// ===================================================
// HANDIT — 認識エンジン
//
// 2段構えで単語を判定する。
//
//  1) 学習済みモデル (model_single.onnx)
//     train.py で学習した単語。精度が高い。
//
//  2) プロトタイプ照合 (encoder.onnx)
//     スタジオで収録しただけで、まだ train.py を回していない単語。
//     エンコーダで埋め込みを作り、収録時の平均ベクトルとの
//     コサイン類似度で判定する。学習なしで即使える代わりに精度は落ちる。
//
// train.py を回すと 2) の単語が 1) に昇格する。
// ===================================================

import { TARGET_FRAMES, FEATURE_DIM, mirrorSequence } from './features.js';

const PROTO_KEY = 'handit_prototypes';

export class Recognizer {
  constructor(opts = {}) {
    this.modelPath   = opts.modelPath   || 'dataset/model_single.onnx';
    this.labelsPath  = opts.labelsPath  || 'dataset/labels.json';
    this.encoderPath = opts.encoderPath || 'dataset/encoder.onnx';
    this.confThreshold  = opts.confThreshold  ?? 0.70;
    this.protoThreshold = opts.protoThreshold ?? 0.82;

    this.session    = null;   // 分類モデル
    this.encoder    = null;   // 埋め込みエンコーダ
    this.labels     = [];     // 学習済みの単語
    this.prototypes = {};     // { 単語: [埋め込みベクトル, ...] }
    this.busy       = false;
  }

  // ---- 読み込み ----
  async load() {
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.16.3/dist/';

    // 分類モデル（まだ1単語も学習していない場合は無くてもよい）
    try {
      this.session = await ort.InferenceSession.create(this.modelPath);
      const ld = await fetch(this.labelsPath).then(r => r.json());
      this.labels = ld.labels || [];
    } catch (e) {
      console.warn('分類モデルを読み込めませんでした。プロトタイプのみで動作します。', e);
    }

    this.prototypes = loadPrototypes();

    // プロトタイプがある場合だけエンコーダを読む
    if (Object.keys(this.prototypes).length > 0 || this.session === null) {
      try {
        this.encoder = await ort.InferenceSession.create(this.encoderPath);
      } catch (e) {
        console.warn('encoder.onnx を読み込めませんでした。即席登録した単語は認識されません。', e);
      }
    }

    if (!this.session && !this.encoder) {
      throw new Error('モデルが1つも読み込めませんでした');
    }
    return this;
  }

  // 学習済み + 即席登録、両方あわせて認識できる単語
  get knownWords() {
    return [...new Set([...this.labels, ...Object.keys(this.prototypes)])];
  }

  isKnown(word) { return this.knownWords.includes(word); }

  // ---- 埋め込み（プロトタイプ作成にも使う） ----
  async embed(frames) {
    if (!this.encoder) return null;
    const t = new ort.Tensor('float32', Float32Array.from(frames.flat()), [1, TARGET_FRAMES, FEATURE_DIM]);
    const out = await this.encoder.run({ input: t });
    const key = this.encoder.outputNames[0];
    return l2normalize(Array.from(out[key].data));
  }

  // ---- 判定 ----
  // frames: 64 x 168 の配列
  // 戻り値: { label, conf, source } / null
  async classify(frames) {
    if (this.busy) return null;
    this.busy = true;
    try {
      let best = null;

      if (this.session) {
        const t = new ort.Tensor('float32', Float32Array.from(frames.flat()), [1, TARGET_FRAMES, FEATURE_DIM]);
        const out = await this.session.run({ input: t });
        const key = this.session.outputNames[0];
        const probs = softmax(Array.from(out[key].data));
        let bi = 0;
        for (let i = 1; i < probs.length; i++) if (probs[i] > probs[bi]) bi = i;
        best = { label: this.labels[bi], conf: probs[bi], source: 'model' };
      }

      // 学習済みモデルが自信を持てなかったときだけプロトタイプを見る
      const needProto = this.encoder
        && Object.keys(this.prototypes).length > 0
        && (!best || best.conf < this.confThreshold);

      if (needProto) {
        const emb = await this.embed(frames);
        if (emb) {
          let bw = null, bs = -1;
          for (const [word, vecs] of Object.entries(this.prototypes)) {
            for (const v of vecs) {
              const s = dot(emb, v);
              if (s > bs) { bs = s; bw = word; }
            }
          }
          if (bw && bs >= this.protoThreshold) {
            // 類似度をだいたい確信度のスケールに直す
            const conf = Math.min(0.99, (bs - this.protoThreshold) / (1 - this.protoThreshold) * 0.3 + 0.70);
            if (!best || conf > best.conf) best = { label: bw, conf, source: 'prototype' };
          }
        }
      }

      return best;
    } catch (e) {
      console.error('推論エラー:', e);
      return null;
    } finally {
      this.busy = false;
    }
  }

  // ---- プロトタイプ登録（スタジオから呼ぶ） ----
  // samples: [[64 x 168], ...] 収録した複数テイク
  async registerPrototype(word, samples) {
    if (!this.encoder) throw new Error('encoder.onnx が無いため即席登録できません');
    const embs = [];
    for (const s of samples) {
      const a = await this.embed(s);           if (a) embs.push(a);
      const b = await this.embed(mirrorSequence(s)); if (b) embs.push(b); // 左右反転版も登録
    }
    if (!embs.length) throw new Error('埋め込みを作れませんでした');

    // 元のテイクとミラー版で別々に平均を取る（混ぜると意味が薄れる）
    const half = embs.length / 2;
    const orig = embs.filter((_, i) => i % 2 === 0);
    const mirr = embs.filter((_, i) => i % 2 === 1);
    this.prototypes[word] = [meanNorm(orig), meanNorm(mirr)].filter(Boolean);
    savePrototypes(this.prototypes);
    return this.prototypes[word].length;
  }

  removePrototype(word) {
    delete this.prototypes[word];
    savePrototypes(this.prototypes);
  }

  // train.py 後に呼ぶと、学習済みになった単語のプロトタイプを掃除する
  prunePromoted() {
    let n = 0;
    for (const w of Object.keys(this.prototypes)) {
      if (this.labels.includes(w)) { delete this.prototypes[w]; n++; }
    }
    if (n) savePrototypes(this.prototypes);
    return n;
  }
}

// ---- localStorage ----
export function loadPrototypes() {
  try { return JSON.parse(localStorage.getItem(PROTO_KEY) || '{}'); }
  catch { return {}; }
}
export function savePrototypes(p) {
  try { localStorage.setItem(PROTO_KEY, JSON.stringify(p)); }
  catch (e) { console.warn('プロトタイプを保存できませんでした（容量超過の可能性）', e); }
}

// ---- 数値ユーティリティ ----
function softmax(logits) {
  const mx = Math.max(...logits);
  const ex = logits.map(v => Math.exp(v - mx));
  const sm = ex.reduce((a, b) => a + b, 0) || 1;
  return ex.map(v => v / sm);
}
function l2normalize(v) {
  let n = 0; for (const x of v) n += x * x;
  n = Math.sqrt(n) || 1;
  return v.map(x => x / n);
}
function dot(a, b) {
  let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}
function meanNorm(vecs) {
  if (!vecs.length) return null;
  const out = new Array(vecs[0].length).fill(0);
  for (const v of vecs) for (let i = 0; i < v.length; i++) out[i] += v[i];
  return l2normalize(out.map(x => x / vecs.length));
}
