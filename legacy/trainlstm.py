"""
Handit LSTMトレーニングスクリプト v2
======================================
168次元（手 + 顔）の入力に対応。
"""

import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

DATASET_DIR   = "dataset"
EPOCHS        = 150
BATCH_SIZE    = 16
LEARNING_RATE = 0.001
HIDDEN_SIZE   = 256
NUM_LAYERS    = 3
VAL_SPLIT     = 0.2


class SignDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


class HanditLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True, dropout=0.3
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.classifier(out[:, -1, :])


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

    dataset    = SignDataset(X, y)
    val_size   = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    print(f"\n学習: {train_size}サンプル  検証: {val_size}サンプル")

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"デバイス: {device}\n")

    model     = HanditLSTM(input_size, HIDDEN_SIZE, NUM_LAYERS, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)

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
            optimizer.step()
            train_correct += (logits.argmax(1) == y_batch).sum().item()

        train_acc = train_correct / train_size

        model.eval()
        val_correct = 0
        val_loss_total = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                val_loss_total += criterion(logits, y_batch).item()
                val_correct    += (logits.argmax(1) == y_batch).sum().item()

        val_acc = val_correct / val_size
        scheduler.step(val_loss_total)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "input_size":  input_size,
                "hidden_size": HIDDEN_SIZE,
                "num_layers":  NUM_LAYERS,
                "num_classes": num_classes,
                "labels":      labels,
            }, f"{DATASET_DIR}/model.pth")

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS}  train: {train_acc:.3f}  val: {val_acc:.3f}  {'★ best' if val_acc == best_val_acc else ''}")

    print(f"\n学習完了！")
    print(f"  最高検証精度: {best_val_acc:.3f} ({best_val_acc*100:.1f}%)")
    print(f"  保存先: {DATASET_DIR}/model.pth")
    print(f"\n次のステップ: python export.py")


if __name__ == "__main__":
    main()
