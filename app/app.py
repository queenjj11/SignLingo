import os
import pickle
import sys
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nlp.sentence_builder import SentenceBuilder, speak  # noqa: E402

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

st.set_page_config(page_title="SignLingo", layout="wide")


@st.cache_resource
def load_letters_model():
    model_path = os.path.join(MODEL_DIR, "all_letters_model.pkl")
    encoder_path = os.path.join(MODEL_DIR, "all_letters_encoder.pkl")

    if not (os.path.exists(model_path) and os.path.exists(encoder_path)):
        return None, None

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(encoder_path, "rb") as f:
        encoder = pickle.load(f)

    return model, encoder


@st.cache_resource
def load_words_model(lang):
    import torch
    from training.train_words_model import SignLSTM  # noqa: E402

    model_path = os.path.join(MODEL_DIR, f"{lang}_words_model.pt")
    encoder_path = os.path.join(MODEL_DIR, f"{lang}_words_encoder.pkl")
    meta_path = os.path.join(MODEL_DIR, f"{lang}_words_meta.pkl")
    if not all(os.path.exists(p) for p in (model_path, encoder_path, meta_path)):
        return None, None, None

    with open(encoder_path, "rb") as f:
        encoder = pickle.load(f)
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    model = SignLSTM(
        n_features=meta["n_features"],
        n_classes=len(encoder.classes_),
        hidden_size=meta["hidden_size"],
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model, encoder, meta


def hand_columns(prefix, n=21):
    return [f"{prefix}_{c}{i}" for i in range(n) for c in ["x", "y", "z"]]


LETTER_FEATURE_COLUMNS = hand_columns("h0") + ["h0_present"] + hand_columns("h1") + ["h1_present"]


def predict_letter(model, encoder, hands_result):
    """hands_result: MediaPipe Hands result for one frame."""
    if not hands_result.multi_hand_landmarks:
        return None, None

    detected = hands_result.multi_hand_landmarks
    scored = sorted(detected, key=lambda hl: float(np.mean([lm.x for lm in hl.landmark])))

    def flat(hl):
        vals = []
        for lm in hl.landmark:
            vals.extend([lm.x, lm.y, lm.z])
        return vals

    h0 = flat(scored[0])
    h0_present = 1
    if len(scored) >= 2:
        h1 = flat(scored[1])
        h1_present = 1
    else:
        h1 = [0.0] * 63
        h1_present = 0

    row = h0 + [h0_present] + h1 + [h1_present]
    X = pd.DataFrame([row], columns=LETTER_FEATURE_COLUMNS)
    pred = model.predict(X)
    letter = encoder.inverse_transform(pred)[0]
    confidence = float(np.max(model.predict_proba(X)))
    return letter, confidence


def main():
    st.title("SignLingo: ASL / BSL / ISL Sign-to-Speech")

    with st.sidebar:
        st.header("Settings")
        lang = st.selectbox("Language", ["asl", "bsl", "isl"], format_func=str.upper)
        mode = st.radio("Mode", ["Letters (fingerspelling)", "Words (gloss)"])
        run_camera = st.checkbox("Enable webcam", value=False)
        nlp_backend = st.selectbox("Sentence smoothing", ["rules", "transformer"], index=0)

    if "builder" not in st.session_state or st.session_state.get("builder_lang") != lang:
        st.session_state.builder = SentenceBuilder(backend=nlp_backend)
        st.session_state.builder_lang = lang
    else:
        st.session_state.builder.backend = nlp_backend

    builder: SentenceBuilder = st.session_state.builder

    col_video, col_controls = st.columns([2, 1])

    letters_model, letters_encoder = load_letters_model()
    words_model, words_encoder, words_meta = load_words_model(lang)

    with col_video:
        frame_slot = st.empty()
        pred_slot = st.empty()

        if run_camera:
            if mode.startswith("Letters") and letters_model is None:
                st.warning(
                    f"No letters model found for {lang.upper()}. Train one with:\n\n"
                    f"`python training/train_letters_model.py --lang {lang}`"
                )
            elif mode.startswith("Words") and words_model is None:
                st.warning(
                    f"No words model found for {lang.upper()}. Train one with:\n\n"
                    f"`python training/train_words_model.py --lang {lang}`"
                )
            else:
                mp_hands = mp.solutions.hands
                mp_draw = mp.solutions.drawing_utils
                hands_model = mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.7,
                )
                cap = cv2.VideoCapture(0)
                last_letter = None

                # Streamlit re-runs top-to-bottom; this loop yields to the
                # button widgets below via session_state, single-frame per
                # script run is intentional (see README for the "Run" loop
                # note if you want a persistent capture loop instead).
                success, frame = cap.read()
                if success:
                    frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands_model.process(rgb)
                    if results.multi_hand_landmarks:
                        for hl in results.multi_hand_landmarks:
                            mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)

                    if mode.startswith("Letters") and letters_model is not None:
                        letter, conf = predict_letter(letters_model, letters_encoder, results)
                        if letter:
                            last_letter = letter
                            pred_slot.markdown(f"**Prediction:** {letter}  ({conf*100:.1f}%)")
                            st.session_state.current_letter = letter

                    frame_slot.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                cap.release()
                hands_model.close()

    with col_controls:
        st.subheader("Sentence")
        st.text_area("Built sentence", value=builder.render(), height=100, key="sentence_display")

        if mode.startswith("Letters"):
            current_letter = st.session_state.get("current_letter", "")
            st.write(f"Current letter: **{current_letter or '-'}**")

            suggestions = builder.suggest()
            if suggestions:
                st.write("Suggestions:", ", ".join(suggestions))

            b1, b2, b3, b4 = st.columns(4)
            if b1.button("Add letter"):
                if current_letter:
                    builder.add_letter(current_letter)
            if b2.button("Space"):
                builder.add_space()
            if b3.button("Delete"):
                builder.delete_last()
            if b4.button("Clear"):
                builder.clear()
        else:
            gloss_input = st.text_input("Simulated recognized gloss (wire this to live LSTM output)")
            if st.button("Add gloss") and gloss_input:
                builder.add_gloss(gloss_input)
            if st.button("Clear glosses"):
                builder.clear()

        if st.button("🔊 Speak sentence"):
            text = builder.render()
            if text.strip():
                speak(text)

        st.caption(
            "Note: Streamlit reruns the script per interaction, so the webcam "
            "block above grabs one frame per rerun. For a smoother continuous "
            "feed, wrap the capture loop with `st.session_state` frame caching "
            "or use `streamlit-webrtc` (see requirements suggestion in README)."
        )


if __name__ == "__main__":
    main()
