
import argparse
import glob
import os

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

VIDEO_EXTS = ("*.mp4", "*.avi", "*.mov", "*.mkv")

# 2 hands x 21 landmarks x 3 coords = 126, + pose (upper body: 25 landmarks x 3) = 75
N_HAND_FEATURES = 21 * 3 * 2
N_POSE_FEATURES = 25 * 3
N_FEATURES = N_HAND_FEATURES + N_POSE_FEATURES  # 201


def extract_frame_features(holistic_result):
    """Flatten one frame's hands + pose into a fixed-size feature vector,
    zero-filling anything not detected in that frame."""
    left = holistic_result.left_hand_landmarks
    right = holistic_result.right_hand_landmarks
    pose = holistic_result.pose_landmarks

    def flat(landmark_list, n_points):
        if landmark_list is None:
            return [0.0] * (n_points * 3)
        vals = []
        for lm in landmark_list.landmark[:n_points]:
            vals.extend([lm.x, lm.y, lm.z])
        return vals

    left_vec = flat(left, 21)
    right_vec = flat(right, 21)
    pose_vec = flat(pose, 25)  # upper-body subset is enough; skip legs/face

    return left_vec + right_vec + pose_vec


def sample_or_pad(frames, seq_len):
    """Uniformly sample `seq_len` frames if the clip is longer, or pad with
    the last frame repeated if shorter -- keeps every training example the
    same length for the LSTM."""
    n = len(frames)
    if n == 0:
        return None
    if n >= seq_len:
        idx = np.linspace(0, n - 1, seq_len).astype(int)
        return [frames[i] for i in idx]
    padded = frames + [frames[-1]] * (seq_len - n)
    return padded


def process_video(holistic_model, video_path, seq_len, frame_stride=1):
    cap = cv2.VideoCapture(video_path)
    frames = []
    i = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        if i % frame_stride == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = holistic_model.process(rgb)
            if result.left_hand_landmarks or result.right_hand_landmarks:
                frames.append(extract_frame_features(result))
        i += 1
    cap.release()
    return sample_or_pad(frames, seq_len)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", required=True, choices=["asl", "bsl", "isl"])
    parser.add_argument("--input_dir", default=None, help="Defaults to data/raw/<lang>/words")
    parser.add_argument("--output_prefix", default=None, help="Defaults to data/processed/<lang>_words")
    parser.add_argument("--seq_len", type=int, default=30, help="Frames per sample after resampling")
    parser.add_argument("--frame_stride", type=int, default=1, help="Read every Nth frame (speed up long videos)")
    parser.add_argument("--max_per_class", type=int, default=None)
    args = parser.parse_args()

    input_dir = args.input_dir or f"data/raw/{args.lang}/words"
    output_prefix = args.output_prefix or f"data/processed/{args.lang}_words"

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(
            f"{input_dir} not found. Download a word-level dataset for '{args.lang}' "
            f"(see README.md) and reorganize it to {input_dir}/<GLOSS>/*.mp4"
        )

    class_dirs = sorted(
        d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))
    )
    if not class_dirs:
        raise ValueError(f"No class subfolders found under {input_dir}")

    mp_holistic = mp.solutions.holistic
    holistic_model = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    sequences, labels = [], []
    skipped = 0
    for label in class_dirs:
        class_dir = os.path.join(input_dir, label)
        paths = []
        for ext in VIDEO_EXTS:
            paths.extend(glob.glob(os.path.join(class_dir, ext)))
        if args.max_per_class:
            paths = paths[: args.max_per_class]

        print(f"[{args.lang}] {label}: {len(paths)} videos")
        for path in paths:
            seq = process_video(holistic_model, path, args.seq_len, args.frame_stride)
            if seq is None:
                skipped += 1
                continue
            sequences.append(seq)
            labels.append(label)

    holistic_model.close()

    if not sequences:
        raise RuntimeError("No usable clips found -- check input_dir contents.")

    X = np.array(sequences, dtype=np.float32)  # (N, seq_len, N_FEATURES)
    y = pd.DataFrame({"label": labels})

    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
    np.save(f"{output_prefix}_X.npy", X)
    y.to_csv(f"{output_prefix}_y.csv", index=False)

    print(f"\nDone. {X.shape[0]} clips, shape {X.shape}, {skipped} skipped.")
    print(f"Classes: {y['label'].nunique()} -> {output_prefix}_X.npy / _y.csv")


if __name__ == "__main__":
    main()
