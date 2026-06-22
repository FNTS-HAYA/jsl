"""
Handit 前処理スクリプト v2
===========================
collect_motion.html v2（手 + 顔対応）で収集した
JSONファイルをLSTMに渡せるnpy形式に変換する。

特徴量（168次元）:
  右手 landmarks 21点 × 3 = 63
  右手 direction            =  3
  右手 normal               =  3
  左手 landmarks 21点 × 3 = 63
  左手 direction            =  3
  左手 normal               =  3
  顔6点 × 3               = 18
  右手首→顔6点の距離      =  6
  左手首→顔6点の距離      =  6
  合計: 168次元

手が片方しかない場合や顔が検出されない場合はゼロ埋め。
"""

import json
import numpy as np
from pathlib import Path
import glob

TARGET_FRAMES = 64
INPUT_DIR  = "."
OUTPUT_DIR = "dataset"

FACE_KEYS = ["nose", "forehead", "chin", "left_eye", "right_eye", "mouth"]
ZERO_HAND = [0.0] * 69   # 片手分のゼロ
ZERO_FACE = [0.0] * 18   # 顔分のゼロ
ZERO_DIST = [0.0] * 6    # 距離分のゼロ


def extract_hand_vec(hand_data):
    """手のデータを69次元に変換"""
    vec = []
    for lm in hand_data["landmarks"]:
        vec.extend([lm["x"], lm["y"], lm["z"]])
    d = hand_data["direction"]
    vec.extend([d["x"], d["y"], d["z"]])
    n = hand_data["normal"]
    vec.extend([n["x"], n["y"], n["z"]])
    return vec  # 69次元


def extract_face_vec(face_data):
    """顔データを18次元に変換"""
    vec = []
    for key in FACE_KEYS:
        if key in face_data:
            p = face_data[key]
            vec.extend([p["x"], p["y"], p["z"]])
        else:
            vec.extend([0.0, 0.0, 0.0])
    return vec  # 18次元


def calc_wrist_to_face_dist(hand_data, face_data):
    """手首から顔6点への距離を計算（6次元）"""
    wrist = hand_data["landmarks"][0]
    dists = []
    for key in FACE_KEYS:
        if key in face_data:
            p = face_data[key]
            dx = wrist["x"] - p["x"]
            dy = wrist["y"] - p["y"]
            dz = wrist["z"] - p["z"]
            dists.append(float(np.sqrt(dx*dx + dy*dy + dz*dz)))
        else:
            dists.append(0.0)
    return dists  # 6次元


def frame_to_vector(frame):
    """1フレームを168次元ベクトルに変換"""
    hands     = frame.get("hands", {})
    face_data = frame.get("face", None)

    # 右手
    if "Right" in hands:
        right_vec  = extract_hand_vec(hands["Right"])
        right_dist = calc_wrist_to_face_dist(hands["Right"], face_data) if face_data else ZERO_DIST
    else:
        right_vec  = ZERO_HAND[:]
        right_dist = ZERO_DIST[:]

    # 左手
    if "Left" in hands:
        left_vec  = extract_hand_vec(hands["Left"])
        left_dist = calc_wrist_to_face_dist(hands["Left"], face_data) if face_data else ZERO_DIST
    else:
        left_vec  = ZERO_HAND[:]
        left_dist = ZERO_DIST[:]

    # 顔
    face_vec = extract_face_vec(face_data) if face_data else ZERO_FACE[:]

    return right_vec + left_vec + face_vec + right_dist + left_dist
    # 69 + 69 + 18 + 6 + 6 = 168次元


def normalize_length(frames, target=TARGET_FRAMES):
    """フレーム列をtargetフレームに正規化"""
    n = len(frames)
    if n == target:
        return frames
    elif n > target:
        indices = np.linspace(0, n-1, target).astype(int)
        return [frames[i] for i in indices]
    else:
        padding   = target - n
        zero_frame = [0.0] * len(frames[0])
        return frames + [zero_frame] * padding


def process_sample(sample):
    """1サンプルを(64, 168)のnumpy配列に変換"""
    raw_frames = [frame_to_vector(f) for f in sample["frames"]]
    normalized = normalize_length(raw_frames)
    return np.array(normalized, dtype=np.float32)


def main():
    # v2ファイルを優先、v1も読めるが警告を出す
    json_files = glob.glob(f"{INPUT_DIR}/motion_v2*.json")
    if not json_files:
        print("motion_v2_*.jsonが見つかりません。")
        print("collect_motion.html v2で収集したファイルを置いてください。")
        return

    all_samples = []
    for path in json_files:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        samples = data.get("samples", [])
        all_samples.extend(samples)
        print(f"読み込み: {path}  ({len(samples)}サンプル)")

    print(f"\n合計: {len(all_samples)}サンプル")

    labels      = sorted(set(s["label"] for s in all_samples))
    label_to_id = {label: i for i, label in enumerate(labels)}

    print(f"\n単語一覧:")
    for label, idx in label_to_id.items():
        count = sum(1 for s in all_samples if s["label"] == label)
        print(f"  [{idx}] {label}: {count}サンプル")

    X, y = [], []
    for sample in all_samples:
        arr = process_sample(sample)
        X.append(arr)
        y.append(label_to_id[sample["label"]])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    print(f"\nデータ形状:")
    print(f"  X: {X.shape}  (サンプル数, フレーム数, 特徴量次元)")
    print(f"  y: {y.shape}")
    print(f"  特徴量: 168次元（右手69 + 左手69 + 顔18 + 距離12）")

    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    np.save(f"{OUTPUT_DIR}/X.npy", X)
    np.save(f"{OUTPUT_DIR}/y.npy", y)

    with open(f"{OUTPUT_DIR}/labels.json", "w", encoding="utf-8") as f:
        json.dump({"labels": labels, "label_to_id": label_to_id, "input_size": 168}, f, ensure_ascii=False, indent=2)

    print(f"\n保存完了 → {OUTPUT_DIR}/")
    print(f"  X.npy, y.npy, labels.json")
    print(f"\n次のステップ: python train.py")


if __name__ == "__main__":
    main()
