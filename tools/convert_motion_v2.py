#!/usr/bin/env python3
# ===================================================
# 昔の収録データ（motion_v2_*.json）を今の形式に取り込む
#
#   python tools/convert_motion_v2.py "C:\\Users\\shuna\\jsl backup"
#       そのフォルダの motion_v2_*.json を dataset/words/*.npz に取り込む
#
#   python tools/convert_motion_v2.py フォルダ --dry-run
#       取り込まずに、何がどれだけ入るかだけ見る
#
# 取り込んだあとは、いつも通り
#   python train.py
# を実行すれば、新しく録ったぶんと一緒に学習される。
#
# 昔の形式は landmarks / direction / normal / face をそのまま持っており、
# 今の168次元ベクトルと計算方法が一致していることを確認済み。
# したがって変換は近似ではなく、そのまま移し替えている。
# ===================================================

import argparse, glob, json, math, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

FRAMES, DIM = 64, 168
FACE_KEYS = ['nose', 'forehead', 'chin', 'left_eye', 'right_eye', 'mouth']
MIN_HAND_RATIO = 0.5      # 手がこれ未満しか写っていないサンプルは捨てる
MIN_FRAMES     = 12


def hand_block(h):
    """片手 → 69次元。features.js の handVec と同じ並び。"""
    v = []
    for p in h['landmarks']:
        v += [p['x'], p['y'], p['z']]
    d, n = h['direction'], h['normal']
    v += [d['x'], d['y'], d['z']]
    v += [n['x'], n['y'], n['z']]
    return v


def dist_block(h, face):
    """手首 → 顔6点の距離 → 6次元"""
    w = h['landmarks'][0]
    out = []
    for k in FACE_KEYS:
        if not face or k not in face:
            out.append(0.0); continue
        p = face[k]
        out.append(math.sqrt((w['x']-p['x'])**2 + (w['y']-p['y'])**2 + (w['z']-p['z'])**2))
    return out


def frame_vec(fr):
    """1フレーム → 168次元"""
    hands = fr.get('hands') or {}
    face  = fr.get('face') or None
    Z69, Z6 = [0.0]*69, [0.0]*6

    r = hand_block(hands['Right']) if 'Right' in hands else list(Z69)
    l = hand_block(hands['Left'])  if 'Left'  in hands else list(Z69)

    fv = [0.0]*18
    if face:
        for i, k in enumerate(FACE_KEYS):
            p = face.get(k)
            if p:
                fv[i*3], fv[i*3+1], fv[i*3+2] = p['x'], p['y'], p['z']

    rd = dist_block(hands['Right'], face) if 'Right' in hands else list(Z6)
    ld = dist_block(hands['Left'],  face) if 'Left'  in hands else list(Z6)
    return r + l + fv + rd + ld


def resample(seq, n=FRAMES):
    """任意の長さを n フレームに揃える（features.js の resample と同じ）"""
    a = np.asarray(seq, dtype=np.float32)
    if len(a) == n:
        return a
    pos = np.linspace(0, len(a) - 1, n)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, len(a) - 1)
    t = (pos - lo)[:, None].astype(np.float32)
    return a[lo] * (1 - t) + a[hi] * t


def convert_file(path):
    d = json.load(open(path, encoding='utf-8'))
    word = d['label']
    kept, dropped = [], 0
    for s in d['samples']:
        frames = s['frames']
        if len(frames) < MIN_FRAMES:
            dropped += 1; continue
        with_hands = sum(1 for f in frames if f.get('hands'))
        if with_hands / len(frames) < MIN_HAND_RATIO:
            dropped += 1; continue
        kept.append(resample([frame_vec(f) for f in frames]))
    return word, kept, dropped


def main():
    ap = argparse.ArgumentParser(description='昔の motion_v2 データを取り込む')
    ap.add_argument('folder', help='motion_v2_*.json が入っているフォルダ')
    ap.add_argument('--dry-run', action='store_true', help='取り込まずに内容だけ表示する')
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.folder, 'motion_v2_*.json')))
    if not files:
        sys.exit(f'{args.folder} に motion_v2_*.json が見つかりません')

    print(f'{len(files)} ファイルを読み込みます\n')
    buckets, total_dropped = {}, 0
    for p in files:
        word, samples, dropped = convert_file(p)
        buckets.setdefault(word, []).extend(samples)
        total_dropped += dropped
        print(f'  {os.path.basename(p)[:44]:46s} {len(samples):3d}件'
              + (f'（{dropped}件は手が写っておらず除外）' if dropped else ''))

    print(f'\n単語ごとの合計:')
    for w, arr in sorted(buckets.items(), key=lambda x: -len(x[1])):
        print(f'  {w:10s} {len(arr):4d} サンプル')
    print(f'\n使えるもの {sum(len(v) for v in buckets.values())} 件 / 除外 {total_dropped} 件')

    if args.dry_run:
        print('\n--dry-run のため取り込みませんでした。')
        return

    import train
    os.makedirs(train.WORDS_DIR, exist_ok=True)
    print('\n取り込み中...')
    for word, arr in buckets.items():
        incoming = np.stack(arr).astype(np.float32)
        cur = train.load_word(word)
        if cur is None:
            merged, meta = incoming, {}
        else:
            # 既に持っているテイクと重複しないものだけ足す
            seen = train._hashes(cur['samples'])
            import hashlib
            keep = np.array([hashlib.sha1(np.round(s, 5).tobytes()).hexdigest() not in seen
                             for s in incoming], dtype=bool)
            merged = np.concatenate([cur['samples'], incoming[keep]], 0) if keep.any() else cur['samples']
            meta = cur['meta']
        train.save_word(word, merged, meta)
        print(f'  {word:10s} → 合計 {len(merged)} サンプル')

    print('\n完了。次のコマンドで学習できます:')
    print('  python train.py')


if __name__ == '__main__':
    main()
