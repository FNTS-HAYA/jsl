"""
Handit ONNX変換スクリプト v2
"""

import json
import numpy as np
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort

DATASET_DIR = "dataset"


class HanditLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True, dropout=0.3
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),          nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.classifier(out[:, -1, :])


def main():
    print("model.pth を読み込み中...")
    ckpt = torch.load(f"{DATASET_DIR}/model.pth", map_location="cpu", weights_only=True)

    input_size  = ckpt["input_size"]
    hidden_size = ckpt["hidden_size"]
    num_layers  = ckpt["num_layers"]
    num_classes = ckpt["num_classes"]
    labels      = ckpt["labels"]

    print(f"  入力次元: {input_size}")
    print(f"  クラス数: {num_classes}  {labels}")

    model = HanditLSTM(input_size, hidden_size, num_layers, num_classes)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dummy       = torch.randn(1, 64, input_size)
    output_path = f"{DATASET_DIR}/model.onnx"

    print(f"\nONNX に変換中...")
    torch.onnx.export(
        model, dummy, output_path,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=17, dynamo=False
    )

    # 1ファイルにまとめる
    print("1ファイルに統合中...")
    m = onnx.load(output_path)
    single_path = f"{DATASET_DIR}/model_single.onnx"
    onnx.save(m, single_path, save_as_external_data=False)
    print(f"  保存: {single_path}")

    # 動作確認
    print(f"\n動作確認中...")
    session = ort.InferenceSession(single_path)
    X = np.load(f"{DATASET_DIR}/X.npy")
    y = np.load(f"{DATASET_DIR}/y.npy")

    correct = 0
    for i in range(len(X)):
        out  = session.run(["output"], {"input": X[i:i+1]})[0]
        pred = np.argmax(out)
        if pred == y[i]: correct += 1

    acc = correct / len(X)
    print(f"  全サンプル精度: {acc:.3f} ({acc*100:.1f}%)")
    print(f"  正解: {correct} / {len(X)}")
    print(f"\n変換完了！")
    print(f"  {single_path} をindex.htmlと同じフォルダのdataset/に置いてください")


if __name__ == "__main__":
    main()
