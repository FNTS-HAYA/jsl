#!/usr/bin/env python3
# ===================================================
# HANDIT — 「手をかざしただけで正解になる」問題の検査
#
#   python tools/eval_none.py                 # dataset/model_single.onnx を調べる
#   python tools/eval_none.py --model 別の.onnx --labels 別の.json
#
# 調べること
#   1) 静止した手：収録データから手の形を1フレーム取り出し、
#      64フレームずっと止めた入力（＋手が下から上がってくる版）を入れる。
#      単語として 0.60 以上で出たら「誤って正解」。低いほど良い。
#   2) 本物の手話：収録済みのテイクを入れて、正しい単語が 0.60 以上で出るか。
#      ※ 学習に使ったデータも含むので、実際より高めに出る。
#        「なし」を足したことで単語の認識が落ちていないかを見る目安。
#
# 必要なもの: pip install onnxruntime numpy
# ===================================================
import argparse, glob, json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from train import FRAMES, DIM, NONE_LABEL, _shift_hands, _no_hand, WORDS_DIR  # noqa: E402

try:
    import onnxruntime as ort
except ImportError:
    sys.exit('onnxruntime が要ります:  pip install onnxruntime')

THRESH = 0.60   # learn.html の CONF_THRESHOLD と同じ


def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=os.path.join(ROOT, 'dataset', 'model_single.onnx'))
    ap.add_argument('--labels', default=os.path.join(ROOT, 'dataset', 'labels.json'))
    ap.add_argument('--seed', type=int, default=123)
    args = ap.parse_args()

    sess = ort.InferenceSession(args.model)
    labels = json.load(open(args.labels, encoding='utf-8'))['labels']
    inp = sess.get_inputs()[0].name
    # 書き出したモデルはバッチ1固定なので1本ずつ回す
    run = lambda X: np.concatenate([softmax(sess.run(None, {inp: x[None].astype(np.float32)})[0]) for x in X])
    rng = np.random.default_rng(args.seed)

    words = []
    for p in sorted(glob.glob(os.path.join(WORDS_DIR, '*.npz'))):
        z = np.load(p, allow_pickle=True)
        w = str(z['word'])
        if w != NONE_LABEL and w in labels:
            words.append((w, z['samples'].astype(np.float32)))
    if not words:
        sys.exit('調べる単語がありません')

    print(f'モデル: {os.path.relpath(args.model, ROOT)}')
    print(f'ラベル: {" / ".join(labels)}')
    print(f'「なし」クラス: {"あり" if NONE_LABEL in labels else "なし"}\n')

    # ---- 1) 静止した手 ----
    still, raise_ = [], []
    for w, S in words:
        for smp in S[rng.permutation(len(S))[:25]]:
            has = np.where(np.abs(smp[:, [0, 1, 69, 70]]).sum(-1) > 0)[0]
            if not len(has):
                continue
            pose = smp[has[rng.integers(len(has))]]
            # ずっと止めている
            still.append(np.stack([_shift_hands(pose, *rng.normal(0, 0.003, 2)) for _ in range(FRAMES)]))
            # 手なし → 下から上げる → 止める
            lead, ramp = rng.integers(5, 25), rng.integers(4, 12)
            seq = []
            for t in range(FRAMES):
                if t < lead:
                    seq.append(_no_hand(pose))
                elif t < lead + ramp:
                    seq.append(_shift_hands(pose, 0, 0.3 * (1 - (t - lead + 1) / (ramp + 1))))
                else:
                    seq.append(_shift_hands(pose, *rng.normal(0, 0.003, 2)))
            raise_.append(np.stack(seq))

    def false_ok(X):
        P = run(np.stack(X))
        top = P.argmax(1)
        is_word = np.array([labels[i] != NONE_LABEL for i in top])
        return float(((P.max(1) >= THRESH) & is_word).mean())

    print('1) 手をかざしただけ（単語として 0.60 以上で出てしまう割合。低いほど良い）')
    print(f'   止めたまま         : {false_ok(still)*100:5.1f}%  ({len(still)} 本)')
    print(f'   上げて止める       : {false_ok(raise_)*100:5.1f}%  ({len(raise_)} 本)')

    # ---- 2) 本物の手話 ----
    print('\n2) 本物の手話（正しい単語が 0.60 以上で出る割合。高いほど良い）')
    tot_ok = tot_n = 0
    for w, S in words:
        P = run(S)
        ok = (P.argmax(1) == labels.index(w)) & (P.max(1) >= THRESH)
        none = (P.argmax(1) == labels.index(NONE_LABEL)).mean() if NONE_LABEL in labels else 0
        print(f'   {w:10s} {ok.mean()*100:5.1f}%  ({len(S)} テイク, 「なし」と判定 {none*100:.0f}%)')
        tot_ok += ok.sum(); tot_n += len(S)
    print(f'   全体       {tot_ok/tot_n*100:5.1f}%')


if __name__ == '__main__':
    main()
