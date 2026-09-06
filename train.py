#!/usr/bin/env python3
# ===================================================
# HANDIT — 手話認識モデルの学習
#
# 使い方
#   1) スタジオで書き出したファイルを取り込んで学習する（ふだんはこれ）
#        python train.py --add ~/Downloads/handit_dataset_2026-09-04.json
#
#   2) すでに取り込み済みのデータだけで学習し直す
#        python train.py
#
#   3) 取り込みだけして学習しない
#        python train.py --add bundle.json --no-train
#
# 特徴
#   ・過去のデータは dataset/words/*.npz に残り続ける。再収録は不要。
#   ・新しい単語が増えても、前回の重みから続きを学習する（ウォームスタート）。
#     分類ヘッドだけ行を増やすので、既存の単語の精度は落ちにくい。
#   ・学習時に左右反転したデータを自動生成するので、右手でも左手でも認識できる。
#
# 出力
#   dataset/model_single.onnx … 分類モデル（learn.html / free.html が使う）
#   dataset/encoder.onnx      … 埋め込みエンコーダ（studio.html の即席登録が使う）
#   dataset/labels.json       … 単語の並び順
#   dataset/checkpoint.pt     … 次回のウォームスタート用
# ===================================================

import argparse, hashlib, json, os, sys, glob, time
import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    sys.exit("PyTorch が要ります:  pip install torch numpy onnx")

# ---------- 定数（features.js と必ず揃えること） ----------
FRAMES, DIM = 64, 168
HAND_DIM = 69
RIGHT, LEFT, FACE, RDIST, LDIST = 0, 69, 138, 156, 162
FACE_KEYS = ['nose', 'forehead', 'chin', 'left_eye', 'right_eye', 'mouth']
EYE_L, EYE_R = FACE_KEYS.index('left_eye'), FACE_KEYS.index('right_eye')

ROOT      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(ROOT, 'dataset')
WORDS_DIR = os.path.join(DATA_DIR, 'words')
CKPT      = os.path.join(DATA_DIR, 'checkpoint.pt')
LABELS    = os.path.join(DATA_DIR, 'labels.json')


# ===================================================
# ミラー変換 —— features.js の mirrorVec と同じ計算
# ===================================================
def _mirror_hand(block):
    """block: [..., 69] → 左右反転した69次元"""
    out = block.copy()
    present = np.abs(block).sum(-1, keepdims=True) > 0     # 手が写っていないフレームは触らない

    lm = block[..., :63].reshape(*block.shape[:-1], 21, 3)
    lm2 = lm.copy()
    lm2[..., 0] = 1.0 - lm[..., 0]                          # x を反転
    out[..., :63] = lm2.reshape(*block.shape[:-1], 63)

    out[..., 63] = -block[..., 63]                          # 向き: dx だけ符号反転
    out[..., 64] = block[..., 64]
    out[..., 65] = block[..., 65]

    out[..., 66] = block[..., 66]                           # 法線: ny, nz が符号反転
    out[..., 67] = -block[..., 67]
    out[..., 68] = -block[..., 68]

    return np.where(present, out, block)


def mirror(x):
    """x: [N, T, 168] → 左右反転版"""
    out = np.zeros_like(x)

    # 右手 ↔ 左手を入れ替える
    out[..., LEFT:LEFT + HAND_DIM]   = _mirror_hand(x[..., RIGHT:RIGHT + HAND_DIM])
    out[..., RIGHT:RIGHT + HAND_DIM] = _mirror_hand(x[..., LEFT:LEFT + HAND_DIM])

    # 顔6点: x反転 + 左目/右目を入れ替え
    face = x[..., FACE:FACE + 18].reshape(*x.shape[:-1], 6, 3)
    order = list(range(6)); order[EYE_L], order[EYE_R] = EYE_R, EYE_L
    f2 = face[..., order, :].copy()
    f2[..., 0] = 1.0 - f2[..., 0]
    f_present = (np.abs(face).sum((-1, -2), keepdims=True) > 0)
    f2 = np.where(f_present, f2, face)
    out[..., FACE:FACE + 18] = f2.reshape(*x.shape[:-1], 18)

    # 距離: 値は不変。ブロックを入れ替え、左目/右目も入れ替え
    out[..., LDIST:LDIST + 6] = x[..., RDIST:RDIST + 6][..., order]
    out[..., RDIST:RDIST + 6] = x[..., LDIST:LDIST + 6][..., order]
    return out


# ---------- そのほかのデータ拡張 ----------
_X_IDX = ([RIGHT + i * 3 for i in range(21)] + [LEFT + i * 3 for i in range(21)] +
          [FACE + i * 3 for i in range(6)])
_Y_IDX = [i + 1 for i in _X_IDX]


def augment(x, rng):
    """x: [N, T, 168] を軽くゆらす（平行移動・ノイズ・時間の伸縮）"""
    x = x.copy()
    n = x.shape[0]

    # 立ち位置のずれ（距離・向き・法線は平行移動で変わらないので座標だけ動かす）
    dx = rng.normal(0, 0.02, (n, 1, 1)).astype(np.float32)
    dy = rng.normal(0, 0.02, (n, 1, 1)).astype(np.float32)
    for idx, d in ((_X_IDX, dx), (_Y_IDX, dy)):
        block = x[..., idx]
        x[..., idx] = np.where(np.abs(block) > 0, block + d, block)

    # 検出のぶれ
    x += rng.normal(0, 0.004, x.shape).astype(np.float32)

    # 手話の速さの違い
    for i in range(n):
        if rng.random() < 0.5:
            s = rng.uniform(0.82, 1.22)
            src = np.clip(np.linspace(0, (FRAMES - 1) * s, FRAMES), 0, FRAMES - 1)
            lo = np.floor(src).astype(int); hi = np.minimum(lo + 1, FRAMES - 1)
            t = (src - lo)[:, None].astype(np.float32)
            x[i] = x[i][lo] * (1 - t) + x[i][hi] * t
    return x


# ===================================================
# データの保管（過去のデータは消えない）
# ===================================================
def word_path(word):
    safe = hashlib.sha1(word.encode('utf-8')).hexdigest()[:10]
    return os.path.join(WORDS_DIR, f'{safe}.npz')


def load_word(word):
    p = word_path(word)
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=True)
    return {'word': str(z['word']), 'samples': z['samples'].astype(np.float32),
            'meta': json.loads(str(z['meta']))}


def save_word(word, samples, meta):
    os.makedirs(WORDS_DIR, exist_ok=True)
    np.savez_compressed(word_path(word), word=word,
                        samples=samples.astype(np.float32), meta=json.dumps(meta, ensure_ascii=False))


def all_words():
    out = []
    for p in sorted(glob.glob(os.path.join(WORDS_DIR, '*.npz'))):
        z = np.load(p, allow_pickle=True)
        out.append({'word': str(z['word']), 'samples': z['samples'].astype(np.float32),
                    'meta': json.loads(str(z['meta']))})
    return out


def _hashes(arr):
    return {hashlib.sha1(np.round(s, 5).tobytes()).hexdigest() for s in arr}


def ingest(bundle_path):
    """スタジオが書き出した JSON を dataset/words/ に取り込む。既にあるテイクは飛ばす。"""
    with open(bundle_path, encoding='utf-8') as f:
        bundle = json.load(f)
    if bundle.get('featureDim', DIM) != DIM or bundle.get('frames', FRAMES) != FRAMES:
        sys.exit(f'特徴量の形が違います（このスクリプトは {FRAMES}x{DIM} 用）')

    added_total, new_words = 0, []
    for w in bundle['words']:
        word = w['word']
        incoming = np.asarray(w['samples'], dtype=np.float32).reshape(-1, FRAMES, DIM)
        cur = load_word(word)
        if cur is None:
            new_words.append(word)
            merged, meta = incoming, {}
        else:
            seen = _hashes(cur['samples'])
            keep = np.array([hashlib.sha1(np.round(s, 5).tobytes()).hexdigest() not in seen
                             for s in incoming], dtype=bool)
            merged = np.concatenate([cur['samples'], incoming[keep]], 0) if keep.any() else cur['samples']
            meta = cur['meta']
            incoming = incoming[keep]
        meta.update({k: w[k] for k in ('cat', 'level', 'desc') if w.get(k) is not None})
        save_word(word, merged, meta)
        added_total += len(incoming)
        print(f'  {word:12s} +{len(incoming):3d}  → 合計 {len(merged)} テイク')

    print(f'\n取り込み完了: {added_total} テイク追加、新しい単語 {len(new_words)} 個 {new_words}')
    return added_total


# ===================================================
# モデル
# ===================================================
class SignTransformer(nn.Module):
    def __init__(self, num_classes, d_model=128, nhead=4, layers=3):
        super().__init__()
        self.inp = nn.Linear(DIM, d_model)
        self.pos = nn.Parameter(torch.zeros(1, FRAMES, d_model))
        enc = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model * 2,
                                         dropout=0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(0.2)
        self.head = nn.Linear(d_model, num_classes)
        nn.init.trunc_normal_(self.pos, std=0.02)

    def embed(self, x):
        h = self.inp(x) + self.pos
        h = self.enc(h)
        return self.norm(h.mean(1))          # 時間方向の平均プーリング

    def forward(self, x):
        return self.head(self.drop(self.embed(x)))


class EncoderOnly(nn.Module):
    """ONNX 書き出し用。埋め込みをそのまま返す。

    以前はここで L2 正規化していたが、その割り算が opset 18 では
    「軸を入力として受け取る ReduceL2」に変換され、
    ONNX Runtime Web が読めずにモデルの読み込みごと失敗していた。
    正規化は js/recognizer.js の embed() が受け取った後に行うので、
    グラフ側では何もしない。
    """
    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, x):
        return self.m.embed(x)


def warm_start(model, labels):
    """前回の重みを引き継ぐ。新しい単語の分だけヘッドの行が増える。"""
    if not os.path.exists(CKPT):
        print('前回のチェックポイントが無いので最初から学習します')
        return 0
    try:
        ck = torch.load(CKPT, map_location='cpu')
        old_labels, old_sd = ck['labels'], ck['state_dict']
        sd = model.state_dict()
        copied = 0
        for k, v in old_sd.items():
            if k.startswith('head.'):
                continue                                    # ヘッドは別扱い
            if k in sd and sd[k].shape == v.shape:
                sd[k] = v; copied += 1
        # ヘッド: 前からある単語の行だけコピー、新しい単語はランダムのまま
        idx = {w: i for i, w in enumerate(old_labels)}
        reused = 0
        for i, w in enumerate(labels):
            if w in idx:
                sd['head.weight'][i] = old_sd['head.weight'][idx[w]]
                sd['head.bias'][i]   = old_sd['head.bias'][idx[w]]
                reused += 1
        model.load_state_dict(sd)
        print(f'ウォームスタート: 重み {copied} 個を引き継ぎ、'
              f'{reused}/{len(labels)} 単語の分類ヘッドを再利用')
        return reused
    except Exception as e:
        print(f'ウォームスタートできませんでした（最初から学習します）: {e}')
        return 0


# ===================================================
# 学習
# ===================================================
def train(epochs=120, val_ratio=0.2, batch=32, lr=3e-4, seed=0):
    data = all_words()
    if len(data) < 2:
        sys.exit('学習するには単語が2つ以上必要です。スタジオで収録して --add してください。')

    # ラベルの並び順は前回を保つ（ヘッドの行を再利用するため）
    prev = []
    if os.path.exists(LABELS):
        prev = json.load(open(LABELS, encoding='utf-8')).get('labels', [])
    present = [d['word'] for d in data]
    labels = [w for w in prev if w in present] + sorted(w for w in present if w not in prev)

    rng = np.random.default_rng(seed)
    Xtr, Ytr, Xva, Yva = [], [], [], []
    print('\n--- データ ---')
    for d in data:
        y = labels.index(d['word'])
        s = d['samples']
        if len(s) < 2:
            print(f'  {d["word"]:12s} {len(s):3d} テイク  ← 少なすぎるので学習から除外')
            continue
        order = rng.permutation(len(s))
        nv = max(1, int(len(s) * val_ratio))
        Xva.append(s[order[:nv]]);  Yva += [y] * nv
        Xtr.append(s[order[nv:]]);  Ytr += [y] * (len(s) - nv)
        print(f'  {d["word"]:12s} {len(s):3d} テイク  (学習 {len(s)-nv} / 検証 {nv})')

    Xtr = np.concatenate(Xtr); Xva = np.concatenate(Xva)
    Ytr = np.array(Ytr); Yva = np.array(Yva)

    # 左右反転でデータを2倍にする ＝ どちらの手でやっても認識できるようになる
    Xtr = np.concatenate([Xtr, mirror(Xtr)]); Ytr = np.concatenate([Ytr, Ytr])
    Xva = np.concatenate([Xva, mirror(Xva)]); Yva = np.concatenate([Yva, Yva])
    print(f'\n左右反転でデータを2倍に: 学習 {len(Xtr)} / 検証 {len(Xva)}、単語 {len(labels)} 個')

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SignTransformer(len(labels)).to(dev)
    warm_start(model, labels)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.05)
    Xva_t = torch.from_numpy(Xva).to(dev); Yva_t = torch.from_numpy(Yva).long().to(dev)

    best, best_sd, patience = -1.0, None, 0
    print(f'\n--- 学習 ({dev}) ---')
    for ep in range(1, epochs + 1):
        model.train()
        xb_all = augment(Xtr, rng)
        perm = rng.permutation(len(xb_all))
        tot = 0.0
        for i in range(0, len(perm), batch):
            sl = perm[i:i + batch]
            xb = torch.from_numpy(xb_all[sl]).to(dev)
            yb = torch.from_numpy(Ytr[sl]).long().to(dev)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(sl)
        sched.step()

        model.eval()
        with torch.no_grad():
            acc = (model(Xva_t).argmax(1) == Yva_t).float().mean().item()
        if acc > best:
            best, best_sd, patience = acc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
        if ep % 10 == 0 or ep == 1:
            print(f'  epoch {ep:3d}  loss {tot/len(perm):.4f}  検証精度 {acc*100:5.1f}%  (best {best*100:.1f}%)')
        if patience >= 40:
            print(f'  {ep} epoch で頭打ちになったので終了')
            break

    model.load_state_dict(best_sd)
    print(f'\n検証精度: {best*100:.1f}%')
    if best < 0.8:
        print('  ※ 精度が低い。テイク数を増やすか、似すぎている単語がないか確認する。')

    # ---- 保存 ----
    os.makedirs(DATA_DIR, exist_ok=True)
    torch.save({'state_dict': model.state_dict(), 'labels': labels,
                'trained_at': time.strftime('%Y-%m-%d %H:%M'), 'val_acc': best}, CKPT)
    json.dump({'labels': labels}, open(LABELS, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

    model.cpu().eval()
    dummy = torch.zeros(1, FRAMES, DIM)
    export_onnx(model, dummy, os.path.join(DATA_DIR, 'model_single.onnx'), 'output')
    export_onnx(EncoderOnly(model).eval(), dummy, os.path.join(DATA_DIR, 'encoder.onnx'), 'embedding')

    export_prototypes(model, data, labels)

    print('\n書き出した:')
    print('  dataset/model_single.onnx')
    print('  dataset/encoder.onnx')
    print('  dataset/labels.json')
    print('  dataset/prototypes.json')
    print('\n単語:', ' / '.join(labels))
    print('\nこの4つを push すれば公開版に反映される。')


# ---------------------------------------------------
# ONNX の書き出し
#
# ブラウザ側は onnxruntime-web 1.16 を使っている。これが読めるのは
# IR バージョン 9 まで、opset は 19 まで。
# ところが今の PyTorch は IR バージョン 10 以降で書き出すため、
# そのままだと ORT Web が読み込みに失敗し、
# 「10034032」のような数字だけのエラーになる。
#
# そこで
#   ・古い方（TorchScript）の書き出し器を使う … 素直なグラフになる
#   ・opset は 17 に固定                        … ORT Web 1.16 が確実に読める
#   ・書き出したあと IR バージョンを 9 に下げる
# の3点で古い ORT Web に合わせている。
#
# 将来 ORT Web を新しくしたら、この関数ごと
# torch.onnx.export の1行に戻してよい。
# ---------------------------------------------------
OPSET = 17
IR_VERSION = 9


def export_onnx(module, dummy, path, output_name):
    try:
        torch.onnx.export(module, dummy, path, input_names=['input'],
                          output_names=[output_name], opset_version=OPSET, dynamo=False)
    except TypeError:
        # dynamo 引数が無い古い PyTorch
        torch.onnx.export(module, dummy, path, input_names=['input'],
                          output_names=[output_name], opset_version=OPSET)

    try:
        import onnx
        m = onnx.load(path)
        if m.ir_version > IR_VERSION:
            m.ir_version = IR_VERSION
            onnx.save(m, path)
        ops = {o.version for o in m.opset_import if o.domain in ('', 'ai.onnx')}
        print(f'  {os.path.basename(path):22s} opset={sorted(ops)} IR={m.ir_version}')
    except ImportError:
        print(f'  {os.path.basename(path):22s} 書き出し済み（onnx 未導入のため IR は未調整）')

    # ブラウザに渡す前にここで読めるか確かめておく
    try:
        import onnxruntime as rt
        sess = rt.InferenceSession(path, providers=['CPUExecutionProvider'])
        sess.run(None, {'input': dummy.numpy()})
        print(f'      読み込みテスト OK')
    except ImportError:
        pass
    except Exception as e:
        print(f'      !! 読み込みテスト失敗: {e}')
        print(f'      !! このままだとブラウザでも読めない可能性が高い')


def export_prototypes(model, data, labels):
    """埋め込みの平均をプロトタイプとして書き出す。

    エンコーダが変わると古いプロトタイプは意味を失うので、
    学習のたびにここで作り直す。encoderVersion で新旧を見分ける。
    ブラウザ側（js/recognizer.js）と同じく、元のぶんと左右反転したぶんで
    別々に平均を取って2本持たせる。
    """
    enc = EncoderOnly(model).eval()
    proto = {}
    with torch.no_grad():
        for d in data:
            x = d['samples']
            if len(x) == 0:
                continue
            def mean_of_normalized(arr):
                e = enc(torch.from_numpy(arr))
                e = e / (e.norm(dim=-1, keepdim=True) + 1e-8)   # 1本ずつ正規化
                m = e.mean(0)
                return m / (m.norm() + 1e-8)                    # 平均してもう一度
            a = mean_of_normalized(x)
            b = mean_of_normalized(mirror(x))
            vecs = [[round(float(t), 4) for t in v] for v in (a, b)]
            proto[d['word']] = vecs

    enc_path = os.path.join(DATA_DIR, 'encoder.onnx')
    ver = hashlib.sha1(open(enc_path, 'rb').read()).hexdigest()[:12]
    json.dump({'encoderVersion': ver, 'words': proto},
              open(os.path.join(DATA_DIR, 'prototypes.json'), 'w', encoding='utf-8'),
              ensure_ascii=False)
    print(f'\nプロトタイプ {len(proto)} 単語を書き出した (encoderVersion={ver})')


# ===================================================
def main():
    ap = argparse.ArgumentParser(description='HANDIT 手話認識モデルの学習')
    ap.add_argument('--add', metavar='FILE', nargs='*', help='スタジオが書き出した JSON を取り込む')
    ap.add_argument('--no-train', action='store_true', help='取り込みだけして学習しない')
    ap.add_argument('--epochs', type=int, default=120)
    ap.add_argument('--fresh', action='store_true', help='ウォームスタートせず最初から学習する')
    ap.add_argument('--list', action='store_true', help='いま持っているデータを表示する')
    args = ap.parse_args()

    os.makedirs(WORDS_DIR, exist_ok=True)

    if args.list:
        data = all_words()
        if not data:
            print('データがまだありません'); return
        print(f'{len(data)} 単語 / {sum(len(d["samples"]) for d in data)} テイク\n')
        for d in sorted(data, key=lambda d: -len(d['samples'])):
            print(f'  {d["word"]:14s} {len(d["samples"]):3d} テイク  {d["meta"].get("cat","")}')
        return

    if args.add:
        for f in args.add:
            print(f'\n取り込み: {f}')
            ingest(f)

    if args.no_train:
        return

    if args.fresh and os.path.exists(CKPT):
        os.rename(CKPT, CKPT + '.bak')
        print('チェックポイントを退避して最初から学習します')

    train(epochs=args.epochs)


if __name__ == '__main__':
    main()
