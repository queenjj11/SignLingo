import argparse
import glob
import os

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

N_LANDMARKS = 21
COORDS = ["x", "y", "z"]
IMG_EXTS = ("*.jpg", "*.jpeg", "*.png")

# Ignore non-alphabet classes
IGNORE_CLASSES = {
    "space",
    "nothing",
    "del",
    "{",
    ".DS_Store",
}


def hand_columns(prefix):
    return [f"{prefix}_{c}{i}" for i in range(N_LANDMARKS) for c in COORDS]


def flatten_landmarks(hand_landmarks):
    vals = []
    for lm in hand_landmarks.landmark:
        vals.extend([lm.x, lm.y, lm.z])
    return vals


def order_hands_left_to_right(multi_hand_landmarks):
    scored = []

    for hand in multi_hand_landmarks:
        mean_x = np.mean([lm.x for lm in hand.landmark])
        scored.append((mean_x, hand))

    scored.sort(key=lambda x: x[0])

    return [h for _, h in scored]


def process_image(hands_model, image_path):
    img = cv2.imread(image_path)

    if img is None:
        return None

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands_model.process(rgb)

    if not results.multi_hand_landmarks:
        return None

    detected = order_hands_left_to_right(results.multi_hand_landmarks)

    h0 = flatten_landmarks(detected[0])
    h0_present = 1

    if len(detected) >= 2:
        h1 = flatten_landmarks(detected[1])
        h1_present = 1
    else:
        h1 = [0.0] * 63
        h1_present = 0

    return h0, h0_present, h1, h1_present


# ---------------------------------------------------------
# DATASET PATHS (YOUR PROJECT STRUCTURE)
# ---------------------------------------------------------

DEFAULT_FOLDERS = {

    # NOTE:
    # Your ASL dataset has an extra nested folder.
    "asl": [
        "dataset/asl/asl_alphabet_train/asl_alphabet_train"
    ],

    "isl": [
        "dataset/isl"
    ],

    "bsl": [
        "dataset/bsl/train",
        "dataset/bsl/test"
    ]
}


def get_class_directories(input_dirs):
    class_to_dirs = {}

    for root in input_dirs:

        if not os.path.isdir(root):
            continue

        for folder in sorted(os.listdir(root)):

            if folder in IGNORE_CLASSES:
                continue

            full = os.path.join(root, folder)

            if os.path.isdir(full):
                class_to_dirs.setdefault(folder.upper(), []).append(full)

    return class_to_dirs


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--lang",
        required=True,
        choices=["asl", "isl", "bsl"]
    )

    parser.add_argument(
        "--output",
        default=None
    )

    parser.add_argument(
        "--max_per_class",
        type=int,
        default=None
    )

    args = parser.parse_args()

    input_dirs = DEFAULT_FOLDERS[args.lang]

    output_path = args.output or f"data/processed/{args.lang}_letters.csv"

    class_to_dirs = get_class_directories(input_dirs)

    if not class_to_dirs:
        raise RuntimeError(
            f"No class folders found.\nChecked:\n{input_dirs}"
        )

    print(f"\n[{args.lang.upper()}]")
    print(f"Folders merged : {len(input_dirs)}")
    print(f"Classes found  : {len(class_to_dirs)}\n")

    mp_hands = mp.solutions.hands

    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5,
    )

    rows = []

    skipped = 0

    for label in sorted(class_to_dirs.keys()):

        image_paths = []

        for folder in class_to_dirs[label]:

            for ext in IMG_EXTS:
                image_paths.extend(
                    glob.glob(os.path.join(folder, ext))
                )

        if args.max_per_class:
            image_paths = image_paths[:args.max_per_class]

        print(f"{label}: {len(image_paths)} images")

        for image in image_paths:

            result = process_image(hands, image)

            if result is None:
                skipped += 1
                continue

            h0, h0_present, h1, h1_present = result

            rows.append(
                h0 +
                [h0_present] +
                h1 +
                [h1_present] +
                [label]
            )

    hands.close()

    if len(rows) == 0:
        raise RuntimeError(
            "No hands detected in any image."
        )

    columns = (
        hand_columns("h0")
        + ["h0_present"]
        + hand_columns("h1")
        + ["h1_present"]
        + ["label"]
    )

    df = pd.DataFrame(rows, columns=columns)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)

    print("\n---------------------------------------")
    print(f"Rows written : {len(df)}")
    print(f"Skipped      : {skipped}")
    print(f"Classes      : {df['label'].nunique()}")
    print(f"Saved to     : {output_path}")
    print("---------------------------------------")


if __name__ == "__main__":
    main()