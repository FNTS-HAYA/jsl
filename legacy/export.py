"""
Handit ONNX変換スクリプト v3 — Transformer版
"""

import json
import math
import numpy as np
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort

DATASET_DIR = "dataset"


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=128, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class HanditTransformer(nn.Module):
    def __init__(self, input_size, num_classes, d_model, nhead, num_layers, dim_ff, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc    = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_ff, dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier  = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x)


def main():
    print("model.pth を読み込み中...")
    ckpt = torch.load(f"{DATASET_DIR}/model.pth", map_location="cpu", weights_only=True)

    input_size  = ckpt["input_size"]
    num_classes = ckpt["num_classes"]
    labels      = ckpt["labels"]
    d_model     = ckpt.get("d_model",   128)
    nhead       = ckpt.get("nhead",     4)
    num_layers  = ckpt.get("num_layers",3)
    dim_ff      = ckpt.get("dim_ff",    256)

    print(f"  入力次元: {input_size}")
    print(f"  クラス数: {num_classes}  {labels}")
    print(f"  モデル:   Transformer (d_model={d_model}, heads={nhead}, layers={num_layers})")

    model = HanditTransformer(input_size, num_classes, d_model, nhead, num_layers, dim_ff)
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

    print("1ファイルに統合中...")
    m = onnx.load(output_path)
    single_path = f"{DATASET_DIR}/model_single.onnx"
    onnx.save(m, single_path, save_as_external_data=False)
    print(f"  保存: {single_path}")

    print(f"\n動作確認中...")
    session = ort.InferenceSession(single_path)
    X = np.load(f"{DATASET_DIR}/X.npy")
    y = np.load(f"{DATASET_DIR}/y.npy")

    correct = 0
    from collections import defaultdict
    per_label = defaultdict(lambda: [0,0])
    for i in range(len(X)):
        out  = session.run(["output"], {"input": X[i:i+1]})[0]
        pred = np.argmax(out)
        per_label[y[i]][1] += 1
        if pred == y[i]:
            correct += 1
            per_label[y[i]][0] += 1

    acc = correct / len(X)
    print(f"  全サンプル精度: {acc:.3f} ({acc*100:.1f}%)")
    print(f"\n単語別精度:")
    for i, label in enumerate(labels):
        c, t = per_label[i]
        print(f"  {label}: {c}/{t} ({c/t*100:.0f}%)" if t > 0 else f"  {label}: -")

    print(f"\n変換完了！→ {single_path}")


if __name__ == "__main__":
    main()
