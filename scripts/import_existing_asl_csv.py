"""
Fold your existing dataset/asl_landmarks_final.csv into the new unified
TWO-HAND schema used across languages, so ASL (one-handed) and BSL
(two-handed) letter data can share the same training code.

Unified schema for letter-level data (one row = one static hand pose):
    h0_x0,h0_y0,h0_z0, ..., h0_x20,h0_y20,h0_z20, h0_present,
    h1_x0,h1_y0,h1_z0, ..., h1_x20,h1_y20,h1_z20, h1_present,
    label

  - h0 = first detected hand, h1 = second detected hand (BSL only).
  - *_present is 1 if that hand's landmarks are real, 0 if zero-filled padding.

Your original CSV only has one hand's 21 landmarks (63 features) + label.
We place that into the h0 slot, mark h0_present=1, and zero-fill h1 with
h1_present=0 -- so ASL rows and BSL rows can sit in the same DataFrame /
be trained with the same MLP code (see training/train_letters_model.py).

Usage:
    python scripts/import_existing_asl_csv.py \
        --input ../asl_landmarks_final.csv \
        --output data/processed/asl_letters.csv
"""
import argparse
import os
import numpy as np
import pandas as pd

N_LANDMARKS = 21
COORDS = ["x", "y", "z"]


def hand_columns(prefix):
    return [f"{prefix}_{c}{i}" for i in range(N_LANDMARKS) for c in COORDS]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="dataset/asl_landmarks_final.csv",
        help="Path to your existing ASL landmarks CSV",
    )
    parser.add_argument(
        "--output",
        default="data/processed/asl_letters.csv",
        help="Where to write the unified-schema CSV",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(
            f"Could not find {args.input}. Pass --input path/to/asl_landmarks_final.csv"
        )

    src = pd.read_csv(args.input)
    if "label" not in src.columns:
        raise ValueError("Expected a 'label' column in the source CSV.")

    old_coord_cols = [c for c in src.columns if c != "label"]
    if len(old_coord_cols) != 63:
        raise ValueError(
            f"Expected 63 coordinate columns (21 landmarks x xyz), found {len(old_coord_cols)}"
        )

    h0_cols = hand_columns("h0")
    h1_cols = hand_columns("h1")
    n = len(src)

    h0_df = pd.DataFrame(src[old_coord_cols].values, columns=h0_cols)
    h1_df = pd.DataFrame(np.zeros((n, 63)), columns=h1_cols)
    out = pd.concat(
        [
            h0_df,
            pd.Series(np.ones(n, dtype=int), name="h0_present"),
            h1_df,
            pd.Series(np.zeros(n, dtype=int), name="h1_present"),
            src["label"].reset_index(drop=True),
        ],
        axis=1,
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out.to_csv(args.output, index=False)

    print(f"Imported {len(out)} rows, {out['label'].nunique()} classes -> {args.output}")
    print(out["label"].value_counts())


if __name__ == "__main__":
    main()
