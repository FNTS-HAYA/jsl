"""
Handit トレーニングスクリプト v3 — Transformer版
==================================================
preprocess.py の出力（168次元）をそのまま使えます。
collect_motion.html や preprocess.py の変更は不要。

LSTMからTransformerに変更した点：
- 全フレームを一度に見てどのフレームが重要か学習（Attention）
- 動きの「どこが特徴的か」を自動で見つけられる
"""

import json
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

DATASET_DIR   = "dataset"
EPOCHS        = 150
BATCH_SIZE    = 16
LEARNING_RATE = 0.0005
VAL_SPLIT     = 0.2

# Transformer設定
D_MODEL    = 128   # 埋め込み次元
NHEAD      = 4     # Attentionのヘッド数
NUM_LAYERS = 3     # Transformerの層数
DIM_FF     = 256   # フィードフォワード層のサイズ
DROPOUT    = 0.2


class SignDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


class PositionalEncoding(nn.Module):
    """フレームの順番情報をモデルに教える"""
    def __init__(self, d_model, max_len=128, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class HanditTransformer(nn.Module):
    def __init__(self, input_size, num_classes, d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS, dim_ff=DIM_FF, dropout=DROPOUT):
        super().__init__()
        # 入力を d_model 次元に投影
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc    = PositionalEncoding(d_model, dropout=dropout)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_ff, dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 分類ヘッド
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: (batch, frames, features)
        x = self.input_proj(x)      # → (batch, frames, d_model)
        x = self.pos_enc(x)         # フレーム順番を付加
        x = self.transformer(x)     # → (batch, frames, d_model)
        x = x.mean(dim=1)           # 全フレームを平均（CLSトークンの代わり）
        return self.classifier(x)


def main():
    print("データを読み込み中...")
    X = np.load(f"{DATASET_DIR}/X.npy")
    y = np.load(f"{DATASET_DIR}/y.npy")
    with open(f"{DATASET_DIR}/labels.json", encoding="utf-8") as f:
        label_info = json.load(f)

    labels      = label_info["labels"]
    num_classes = len(labels)
    input_size  = X.shape[2]

    print(f"  サンプル数: {len(X)}")
    print(f"  フレーム数: {X.shape[1]}")
    print(f"  特徴量次元: {input_size}")
    print(f"  クラス数:   {num_classes}  {labels}")
    print(f"  モデル:     Transformer (d_model={D_MODEL}, heads={NHEAD}, layers={NUM_LAYERS})")

    dataset    = SignDataset(X, y)
    val_size   = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    print(f"\n学習: {train_size}サンプル  検証: {val_size}サンプル")

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"デバイス: {device}\n")

    model     = HanditTransformer(input_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_correct = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_correct += (logits.argmax(1) == y_batch).sum().item()
        scheduler.step()
        train_acc = train_correct / train_size

        model.eval()
        val_correct    = 0
        val_loss_total = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                val_loss_total += criterion(logits, y_batch).item()
                val_correct    += (logits.argmax(1) == y_batch).sum().item()

        val_acc = val_correct / val_size

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "input_size":  input_size,
                "num_classes": num_classes,
                "labels":      labels,
                "model_type":  "transformer",
                "d_model":     D_MODEL,
                "nhead":       NHEAD,
                "num_layers":  NUM_LAYERS,
                "dim_ff":      DIM_FF,
            }, f"{DATASET_DIR}/model.pth")

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS}  train: {train_acc:.3f}  val: {val_acc:.3f}  {'★ best' if val_acc == best_val_acc else ''}")

    print(f"\n学習完了！")
    print(f"  最高検証精度: {best_val_acc:.3f} ({best_val_acc*100:.1f}%)")
    print(f"  保存先: {DATASET_DIR}/model.pth")
    print(f"\n次のステップ: python export.py")


if __name__ == "__main__":
    main()
