#!/usr/bin/env python3
# ===================================================
# HANDIT — Firestore との同期（GitHub Actions が使う）
#
#   python tools/sync_firestore.py pull
#       Firestore の contributions を dataset/words/*.npz に取り込む
#
#   python tools/sync_firestore.py push
#       学習後の dataset/prototypes.json を Firestore に書き戻す
#       （エンコーダが変わると古いプロトタイプは無効になるため、
#         学習のたびに作り直して上書きする）
#
# 認証は環境変数 GOOGLE_APPLICATION_CREDENTIALS か
# FIREBASE_SERVICE_ACCOUNT（JSON文字列）で行う。
# ===================================================

import base64, json, os, sys, tempfile
import numpy as np

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(ROOT, 'dataset')
PROTO_OUT = os.path.join(DATA_DIR, 'prototypes.json')
FRAMES, DIM = 64, 168

sys.path.insert(0, ROOT)


def client():
    from google.cloud import firestore
    raw = os.environ.get('FIREBASE_SERVICE_ACCOUNT')
    if raw:
        f = tempfile.NamedTemporaryFile('w', suffix='.json', delete=False)
        f.write(raw); f.close()
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = f.name
    if not os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'):
        sys.exit('サービスアカウントが設定されていません（FIREBASE_SERVICE_ACCOUNT）')
    return firestore.Client()


# ---------------------------------------------------
def pull():
    """contributions を全部取ってきて dataset/words/*.npz にする"""
    import train

    db = client()
    buckets = {}
    n = 0
    for doc in db.collection('contributions').stream():
        d = doc.to_dict()
        if d.get('dim') != DIM or d.get('frames') != FRAMES:
            print(f'  形が違うので飛ばす: {doc.id}')
            continue
        arr = np.frombuffer(base64.b64decode(d['data']), dtype=np.float32)
        if arr.size != FRAMES * DIM:
            print(f'  サイズが合わないので飛ばす: {doc.id}')
            continue
        buckets.setdefault(d['word'], []).append(arr.reshape(FRAMES, DIM))
        n += 1

    if not buckets:
        sys.exit('Firestore に学習データがありません')

    # 既に持っているテイクとハッシュで突き合わせて、増えた分だけ足す
    os.makedirs(train.WORDS_DIR, exist_ok=True)
    for word, samples in buckets.items():
        incoming = np.stack(samples).astype(np.float32)
        cur = train.load_word(word)
        if cur is None:
            merged, meta = incoming, {}
        else:
            seen = train._hashes(cur['samples'])
            keep = np.array([__import__('hashlib').sha1(np.round(s, 5).tobytes()).hexdigest() not in seen
                             for s in incoming], dtype=bool)
            merged = np.concatenate([cur['samples'], incoming[keep]], 0) if keep.any() else cur['samples']
            meta = cur['meta']
        train.save_word(word, merged, meta)
        print(f'  {word:14s} {len(merged):3d} テイク')

    print(f'\nFirestore から {n} テイク / {len(buckets)} 単語を取り込んだ')


# ---------------------------------------------------
def push():
    """学習後のプロトタイプを Firestore に書き戻す"""
    if not os.path.exists(PROTO_OUT):
        sys.exit(f'{PROTO_OUT} がありません。先に train.py を実行してください')
    payload = json.load(open(PROTO_OUT, encoding='utf-8'))

    db = client()
    db.collection('shared').document('prototypes').set({
        'encoderVersion': payload['encoderVersion'],
        'words': payload['words'],
    })
    print(f"プロトタイプ {len(payload['words'])} 単語を書き戻した "
          f"(encoderVersion={payload['encoderVersion']})")

    # 単語一覧にも「学習済み」を反映しておく
    ref = db.collection('shared').document('catalog')
    snap = ref.get()
    if snap.exists:
        cat = snap.to_dict()
        labels = set(json.load(open(os.path.join(DATA_DIR, 'labels.json'), encoding='utf-8'))['labels'])
        for w in cat.get('words', []):
            w['trained'] = w['word'] in labels
        ref.set(cat)
        print(f'単語一覧の学習済みフラグを更新した（{len(labels)} 単語）')


# ---------------------------------------------------
if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in ('pull', 'push'):
        sys.exit('使い方: python tools/sync_firestore.py [pull|push]')
    (pull if sys.argv[1] == 'pull' else push)()
