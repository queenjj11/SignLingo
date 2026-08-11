"""
Train a word/gloss classifier for one language from the landmark
sequences produced by scripts/extract_landmarks_words.py.

Architecture: a small 2-layer bidirectional LSTM over per-frame
(hands + pose) landmark vectors, followed by a linear classifier head.
This is the "words move through space and time" counterpart to the
static-pose MLP used for letters.

Usage:
    python training/train_words_model.py --lang asl --epochs 40
"""
import argparse
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset


class SignLSTM(nn.Module):
    def __init__(self, n_features, n_classes, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        out, _ = self.lstm(x)
        last_step = out[:, -1, :]  # final timestep, both directions
        return self.head(last_step)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", required=True, choices=["asl", "bsl", "isl"])
    parser.add_argument("--input_prefix", default=None, help="Defaults to data/processed/<lang>_words")
    parser.add_argument("--model_dir", default="models")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--test_size", type=float, default=0.2)
    args = parser.parse_args()

    prefix = args.input_prefix or f"data/processed/{args.lang}_words"
    x_path, y_path = f"{prefix}_X.npy", f"{prefix}_y.csv"
    if not (os.path.exists(x_path) and os.path.exists(y_path)):
        raise FileNotFoundError(
            f"{x_path} / {y_path} not found. Run "
            f"scripts/extract_landmarks_words.py --lang {args.lang} first."
        )

    X = np.load(x_path)  # (N, seq_len, n_features)
    import pandas as pd
    y_df = pd.read_csv(y_path)

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_df["label"])

    print(f"[{args.lang}] X shape: {X.shape}, classes: {len(encoder.classes_)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y if len(set(y)) > 1 else None
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)
    )
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    model = SignLSTM(
        n_features=X.shape[2], n_classes=len(encoder.classes_), hidden_size=args.hidden_size
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    print(f"[{args.lang}] Training LSTM for {args.epochs} epochs on {device}...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)

        if epoch % 5 == 0 or epoch == args.epochs:
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for xb, yb in test_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    preds = model(xb).argmax(dim=1)
                    correct += (preds == yb).sum().item()
                    total += yb.size(0)
            acc = correct / max(total, 1)
            print(f"  epoch {epoch:3d} | train loss {total_loss / len(train_ds):.4f} | test acc {acc:.4f}")

    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, f"{args.lang}_words_model.pt")
    encoder_path = os.path.join(args.model_dir, f"{args.lang}_words_encoder.pkl")
    meta_path = os.path.join(args.model_dir, f"{args.lang}_words_meta.pkl")

    torch.save(model.state_dict(), model_path)
    with open(encoder_path, "wb") as f:
        pickle.dump(encoder, f)
    with open(meta_path, "wb") as f:
        pickle.dump(
            {"n_features": X.shape[2], "seq_len": X.shape[1], "hidden_size": args.hidden_size},
            f,
        )

    print(f"[{args.lang}] Saved model -> {model_path}")
    print(f"[{args.lang}] Saved encoder -> {encoder_path}")
    print(f"[{args.lang}] Saved meta -> {meta_path}")


if __name__ == "__main__":
    main()
