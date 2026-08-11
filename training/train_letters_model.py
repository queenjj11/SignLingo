"""
Train a fingerspelling letter classifier using landmark CSV files.

Examples:

Train multilingual model:
    python training/train_letters_model.py --lang all

Train only ISL:
    python training/train_letters_model.py --lang isl

Train only BSL:
    python training/train_letters_model.py --lang bsl
"""

import argparse
import os
import pickle

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--lang",
        default="all",
        choices=["all", "asl", "isl", "bsl"],
        help="Dataset to train on",
    )

    parser.add_argument(
        "--input",
        default=None,
        help="Optional CSV path"
    )

    parser.add_argument(
        "--model_dir",
        default="models"
    )

    parser.add_argument(
        "--hidden",
        type=int,
        nargs="+",
        default=[128, 64]
    )

    parser.add_argument(
        "--max_iter",
        type=int,
        default=500
    )

    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2
    )

    args = parser.parse_args()

    # -----------------------------
    # Choose input CSV
    # -----------------------------

    if args.input:

        input_path = args.input

    elif args.lang == "all":

        input_path = "data/processed/multilingual_letters.csv"

    else:

        input_path = f"data/processed/{args.lang}_letters.csv"

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"{input_path} not found.")

    print(f"\nLoading: {input_path}")

    df = pd.read_csv(input_path)

    print("Dataset shape:", df.shape)

    # ---------------------------------
    # Remove non-feature columns
    # ---------------------------------

    X = df.drop(columns=["label"], errors="ignore")

    if "language" in X.columns:
        X = X.drop(columns=["language"])

    y = df["label"]

    encoder = LabelEncoder()

    y_encoded = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=args.test_size,
        random_state=42,
        stratify=y_encoded,
    )

    print("\nTraining MLP...")

    model = MLPClassifier(
        hidden_layer_sizes=tuple(args.hidden),
        max_iter=args.max_iter,
        random_state=42,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\nAccuracy:", round(accuracy, 4))
    print()
    print(classification_report(
        y_test,
        predictions,
        target_names=encoder.classes_
    ))

    os.makedirs(args.model_dir, exist_ok=True)

    model_name = f"{args.lang}_letters_model.pkl"
    encoder_name = f"{args.lang}_letters_encoder.pkl"

    with open(os.path.join(args.model_dir, model_name), "wb") as f:
        pickle.dump(model, f)

    with open(os.path.join(args.model_dir, encoder_name), "wb") as f:
        pickle.dump(encoder, f)

    print("\nModel saved.")

    print(os.path.join(args.model_dir, model_name))
    print(os.path.join(args.model_dir, encoder_name))


if __name__ == "__main__":
    main()