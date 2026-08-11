# SignLingo — Multi-Language Sign Language Translator
### ASL · BSL · ISL &nbsp;|&nbsp; Letters · Words · Sentences &nbsp;|&nbsp; Real-time webcam → speech

SignLingo is an end-to-end sign language recognition system that turns webcam video into
text and speech across **three sign languages** (American, British, Indian) and **three
linguistic levels** (fingerspelled letters, isolated words, continuous sentences).

This started from a single-language, single-letter MLP classifier (`legacy/`). It has been
rebuilt into a modular pipeline: landmark extraction → per-language models → sentence
construction with NLP smoothing → speech output, wrapped in a Streamlit app.

---

## 1. Why this is hard (and what makes it a real project, not a toy)

| | ASL | BSL | ISL |
|---|---|---|---|
| Hands used for fingerspelling | One | **Two** (totally different alphabet shape) | Mixed (one for some letters, two for others) |
| Grammar | Distinct sentence order (topic-comment) | Distinct order, mouthing plays a grammatical role | SOV order, differs from spoken Hindi/English |
| Words vs. letters | Words are full-arm/body motions, not static poses | Same | Same |

Because of this, a single model can't cover everything. SignLingo trains **two model
families per language**:

1. **Static pose classifiers** (MLP, like your original) for fingerspelled letters —
   one frame of hand landmarks in → one letter out.
2. **Sequence classifiers** (LSTM) for words and common sentence chunks — a few seconds
   of landmark sequences in → a gloss (word) out. Sentences are then assembled from
   glosses + grammar smoothing (see §5), the same way real sign-to-text systems work,
   because there is no dataset of "every possible sentence" — sentences are composed.

---

## 3. Project structure

```
signlingo/
├── data/
│   ├── raw/{asl,bsl,isl}/         ← drop downloaded datasets here (see script docstrings)
│   └── processed/                 ← landmark CSVs / .npy sequences (generated)
├── scripts/
│   ├── import_existing_asl_csv.py     # folds your current CSV into the new schema
│   ├── extract_landmarks_letters.py   # image datasets -> per-language letter CSV
│   └── extract_landmarks_words.py     # video datasets -> per-language word/sentence sequences
├── training/
│   ├── train_letters_model.py     # MLP per language (extends your original approach)
│   └── train_words_model.py       # LSTM per language, sequence classification
├── nlp/
│   └── sentence_builder.py        # gloss -> grammatical sentence, autocomplete, TTS
├── models/                        # saved models per language+level (generated)
├── app/
│   └── app.py                     # Streamlit app: language + mode switch, live webcam
└── legacy/
    ├── predict.py                 # your original script, kept for reference
    └── train_model.py
```

## 4. Pipeline

```
raw images/videos  →  MediaPipe landmark extraction  →  unified CSV/.npy per language
                                                              │
                                        ┌─────────────────────┴─────────────────────┐
                                letters: MLPClassifier                    words: LSTM sequence classifier
                                (scikit-learn, per language)              (PyTorch, per language)
                                        │                                             │
                                        └─────────────────────┬─────────────────────┘
                                                              ▼
                                          nlp/sentence_builder.py (gloss buffer →
                                          autocomplete + grammar smoothing)
                                                              ▼
                                                  pyttsx3 text-to-speech
                                                              ▼
                                                    app/app_webrtc.py (Streamlit UI)
```

## 5. Sentence construction (the "impressive" part)

Sign languages don't map 1:1 onto English word order (ASL/ISL are topic-comment / SOV-ish;
fingerspelling is letter-by-letter). Rather than pretend a labeled dataset exists for every
sentence a user might sign, SignLingo:

1. Recognizes a stream of glosses/letters (e.g. `ME NAME J-O-H-N GO SCHOOL`).
2. Buffers them into a rolling window.
3. Runs a lightweight grammar-smoothing pass (`nlp/sentence_builder.py`) — starting with a
   rule-based reordering + a pluggable slot for a HuggingFace grammar-correction model
   (`pipeline("text2text-generation", model="grammarly/coedit-large")` or similar) — to
   turn that into fluent English/Hindi text: `"My name is John, I go to school."`
4. Speaks the result with `pyttsx3`.

This mirrors how real research systems (e.g. the ISL-CSLTR and INCLUDE papers) frame the
translation problem: recognition gives glosses, a separate language model gives grammar.


