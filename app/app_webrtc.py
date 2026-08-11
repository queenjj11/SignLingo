"""
SignLingo -- live webcam fingerspelling recognition via streamlit-webrtc.

WHAT THIS APP ACTUALLY DOES (read before demoing / writing this up):
    Loads a single multilingual letter classifier, models/all_letters_model.pkl,
    trained on the union of ASL + BSL + ISL letter samples with the *language
    column dropped* -- i.e. the label space is plain A-Z (+ del/space), not
    asl_A / bsl_A / isl_A. In practice this means the model recognizes a
    fingerspelled letter's hand shape regardless of which of the three
    languages it came from, but it does NOT tell you which language a given
    sign belongs to, and it cannot distinguish two languages' letters that
    happen to share the same label but look different (it just learns
    whichever shape dominates that label in training data, or an average
    decision boundary across them). If you need language-specific behavior,
    train separate per-language models instead (see train_letters_model.py
    --lang {asl,bsl,isl}) and swap the loader in load_model() below.
"""
import os
import pickle
import sys
import time
from collections import Counter, deque

import av
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, WebRtcMode, webrtc_streamer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nlp.sentence_builder import SentenceBuilder, speak  # noqa: E402

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "all_letters_model.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "all_letters_encoder.pkl")

CONFIDENCE_THRESHOLD = 0.60          # below this -> shown as "Unknown sign"
SMOOTHING_WINDOW = 8                 # frames of prediction history to vote over
SMOOTHING_MIN_AGREEMENT = 0.5        # fraction of the window that must agree

N_LANDMARKS = 21
HAND_FEATURE_COLUMNS = (
    [f"h0_{c}{i}" for i in range(N_LANDMARKS) for c in ["x", "y", "z"]] + ["h0_present"]
    + [f"h1_{c}{i}" for i in range(N_LANDMARKS) for c in ["x", "y", "z"]] + ["h1_present"]
)

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

st.set_page_config(page_title="SignLingo Live", layout="wide")


# --------------------------------------------------------------------------
# Model loading -- single multilingual model, no per-language fallback
# --------------------------------------------------------------------------
@st.cache_resource
def load_model():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH)):
        return None, None
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f:
        encoder = pickle.load(f)
    return model, encoder


# --------------------------------------------------------------------------
# Landmark -> feature row helpers
# --------------------------------------------------------------------------
def order_hands_left_to_right(multi_hand_landmarks):
    scored = [
        (float(np.mean([lm.x for lm in hl.landmark])), hl) for hl in multi_hand_landmarks
    ]
    scored.sort(key=lambda t: t[0])
    return [hl for _, hl in scored]


def flatten(hand_landmarks):
    vals = []
    for lm in hand_landmarks.landmark:
        vals.extend([lm.x, lm.y, lm.z])
    return vals


def build_feature_row(hands_result):
    if not hands_result.multi_hand_landmarks:
        return None
    detected = order_hands_left_to_right(hands_result.multi_hand_landmarks)

    h0 = flatten(detected[0])
    h0_present = 1
    if len(detected) >= 2:
        h1 = flatten(detected[1])
        h1_present = 1
    else:
        h1 = [0.0] * 63
        h1_present = 0

    row = h0 + [h0_present] + h1 + [h1_present]
    return pd.DataFrame([row], columns=HAND_FEATURE_COLUMNS)


# --------------------------------------------------------------------------
# Video processor -- runs continuously on streamlit-webrtc's own media
# thread, independent of Streamlit script reruns.
# --------------------------------------------------------------------------
class SignPredictor(VideoProcessorBase):
    def __init__(self):
        self.model, self.encoder = load_model()
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )
        self.draw_utils = mp.solutions.drawing_utils
        self.hand_connections = mp.solutions.hands.HAND_CONNECTIONS

        self.history = deque(maxlen=SMOOTHING_WINDOW)
        self.latest_letter = None           # smoothed + thresholded output
        self.latest_confidence = 0.0
        self.latest_raw_letter = None       # unsmoothed, single-frame prediction
        self.latest_inference_ms = 0.0
        self.latest_fps = 0.0
        self._last_frame_time = time.time()

    def _smoothed_prediction(self):
        """Majority vote over the last SMOOTHING_WINDOW frames -- reduces
        flicker between visually similar letters (e.g. M/N/S) that a
        single frame can misclassify for an instant."""
        if not self.history:
            return None, 0.0
        letters = [l for l, _ in self.history]
        counts = Counter(letters)
        best_letter, best_count = counts.most_common(1)[0]
        agreement = best_count / len(letters)
        if agreement < SMOOTHING_MIN_AGREEMENT:
            return None, 0.0
        avg_conf = float(np.mean([c for l, c in self.history if l == best_letter]))
        return best_letter, avg_conf

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        now = time.time()
        dt = now - self._last_frame_time
        self._last_frame_time = now
        self.latest_fps = (1.0 / dt) if dt > 0 else 0.0

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        raw_letter, raw_conf = None, 0.0

        if results.multi_hand_landmarks:
            for hl in results.multi_hand_landmarks:
                self.draw_utils.draw_landmarks(img, hl, self.hand_connections)

            if self.model is not None:
                X = build_feature_row(results)
                if X is not None:
                    t0 = time.perf_counter()
                    pred = self.model.predict(X)
                    proba = self.model.predict_proba(X)
                    self.latest_inference_ms = (time.perf_counter() - t0) * 1000

                    raw_letter = self.encoder.inverse_transform(pred)[0]
                    raw_conf = float(np.max(proba))

        self.latest_raw_letter = raw_letter
        if raw_letter is not None:
            self.history.append((raw_letter, raw_conf))
        else:
            # No hand this frame -- drain the window one frame at a time
            # rather than resetting instantly, so a brief tracking
            # drop-out doesn't wipe out an otherwise-stable prediction.
            if self.history:
                self.history.popleft()

        smoothed_letter, smoothed_conf = self._smoothed_prediction()
        if smoothed_letter is not None and smoothed_conf >= CONFIDENCE_THRESHOLD:
            self.latest_letter = smoothed_letter
            self.latest_confidence = smoothed_conf
            display_text = smoothed_letter
            color = (0, 200, 0)
        elif smoothed_letter is not None:
            self.latest_letter = None
            self.latest_confidence = smoothed_conf
            display_text = "Unknown sign"
            color = (0, 165, 255)
        else:
            self.latest_letter = None
            self.latest_confidence = 0.0
            display_text = "No hand detected" if not results.multi_hand_landmarks else "..."
            color = (0, 0, 255)

        # --- overlay ---
        cv2.putText(img, f"Prediction: {display_text}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.putText(img, f"Confidence: {self.latest_confidence * 100:.1f}%", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(img, f"Inference: {self.latest_inference_ms:.1f} ms | FPS: {self.latest_fps:.1f}",
                    (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# --------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------
def main():
    st.title("SignLingo Live: Multilingual Fingerspelling -> Speech")

    model, encoder = load_model()
    if model is None:
        st.error(
            f"Model not found at `{MODEL_PATH}`. Train the combined multilingual "
            f"model first: concat data/processed/{{asl,bsl,isl}}_letters.csv, drop "
            f"any language prefix from the label so it's plain A-Z, then train + "
            f"save to models/all_letters_model.pkl / all_letters_encoder.pkl."
        )

    with st.expander("About this model", expanded=False):
        st.markdown(
            "This app uses **one multilingual letter classifier** trained on the "
            "combined ASL + BSL + ISL letter data, with plain `A-Z` labels (the "
            "language column was dropped during training). It recognizes a "
            "fingerspelled hand shape regardless of which of the three languages "
            "it came from -- it does **not** identify or distinguish which "
            "language you're signing in."
        )

    if "builder" not in st.session_state:
        st.session_state.builder = SentenceBuilder(backend="rules")
    if "tts_engine" not in st.session_state:
        st.session_state.tts_engine = None

    builder: SentenceBuilder = st.session_state.builder

    col_video, col_controls = st.columns([2, 1])

    with col_video:
        ctx = webrtc_streamer(
            key="signlingo-live",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=SignPredictor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        st.caption(
            "Live overlay shows the smoothed prediction, confidence, and "
            "per-frame inference time / FPS. Predictions below the "
            f"{CONFIDENCE_THRESHOLD:.0%} confidence threshold display as "
            "'Unknown sign' rather than a guess."
        )

    with col_controls:
        st.subheader("Sentence")
        st.text_area("Built sentence", value=builder.render(), height=100, key="sentence_display")

        current_letter, current_conf = None, 0.0
        latest_inference_ms, latest_fps = 0.0, 0.0
        if ctx.video_processor:
            current_letter = ctx.video_processor.latest_letter
            current_conf = ctx.video_processor.latest_confidence
            latest_inference_ms = ctx.video_processor.latest_inference_ms
            latest_fps = ctx.video_processor.latest_fps

        st.metric("Current letter", current_letter or "-", f"{current_conf * 100:.1f}% confidence")

        suggestions = builder.suggest()
        if suggestions:
            st.write("Suggestions:", ", ".join(suggestions))

        b1, b2, b3, b4 = st.columns(4)
        if b1.button("Add letter", disabled=current_letter is None):
            if current_letter:
                builder.add_letter(current_letter)
        if b2.button("Space"):
            builder.add_space()
        if b3.button("Delete"):
            builder.delete_last()
        if b4.button("Clear"):
            builder.clear()

        if st.button("🔊 Speak sentence"):
            text = builder.render()
            if text.strip():
                st.session_state.tts_engine = speak(text, engine=st.session_state.tts_engine)

        st.divider()
        st.caption(f"Last inference: {latest_inference_ms:.1f} ms  ({latest_fps:.1f} FPS)")
        st.caption(
            "The panels above refresh on your next click/interaction, not "
            "continuously -- the webcam feed itself keeps running smoothly "
            "regardless, since it's driven by streamlit-webrtc's own media "
            "thread rather than a Streamlit script rerun loop."
        )


if __name__ == "__main__":
    main()