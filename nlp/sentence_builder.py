from difflib import get_close_matches

# A tiny illustrative gloss->grammar rule table. Extend this as you add
# more glosses from WLASL / INCLUDE / FS23K. Keys are uppercase glosses,
# same casing convention as the folder names used in
# scripts/extract_landmarks_words.py.
PRONOUN_MAP = {"ME": "I", "MY": "my", "YOU": "you", "YOUR": "your"}

# A few common phrase-level rewrites so glosses come out fluent rather
# than a flat word list. This is intentionally a small, extensible seed --
# swap in a real seq2seq grammar model (see `backend="transformer"`) once
# you have enough gloss vocabulary for it to matter.
PHRASE_RULES = [
    (["ME", "NAME"], "My name is"),
    (["YOU", "NAME", "WHAT"], "What is your name?"),
    (["ME", "GO", "SCHOOL"], "I am going to school."),
    (["ME", "WANT"], "I want"),
    (["THANK", "YOU"], "Thank you."),
]

COMMON_WORDS = [
    "HELLO", "THANK", "YOU", "PLEASE", "SORRY", "NAME", "SCHOOL", "WATER",
    "FOOD", "HELP", "FRIEND", "FAMILY", "GOOD", "BAD", "YES", "NO",
    "WANT", "GO", "COME", "STOP", "MORE", "FINISH",
]


class SentenceBuilder:
    def __init__(self, backend="rules", transformer_model="pszemraj/flan-t5-large-grammar-synthesis"):
        self.backend = backend
        self.letters = ""       # current in-progress fingerspelled word
        self.words = []         # completed fingerspelled words
        self.glosses = []       # completed word-level sign glosses
        self._transformer_pipeline = None
        self.transformer_model = transformer_model

    # ---- letter-level controls (mirrors your original predict.py keys) ----
    def add_letter(self, letter):
        self.letters += letter

    def add_space(self):
        if self.letters:
            self.words.append(self.letters)
            self.letters = ""

    def delete_last(self):
        if self.letters:
            self.letters = self.letters[:-1]
        elif self.words:
            self.words.pop()

    def clear(self):
        self.letters, self.words, self.glosses = "", [], []

    # ---- word-level (gloss) controls, for the LSTM word model ----
    def add_gloss(self, gloss):
        self.glosses.append(gloss.upper())

    # ---- autocomplete over partially fingerspelled words ----
    def suggest(self, n=3):
        partial = self.letters.upper()
        if not partial:
            return []
        matches = [w for w in COMMON_WORDS if w.startswith(partial)]
        if not matches:
            matches = get_close_matches(partial, COMMON_WORDS, n=n, cutoff=0.5)
        return matches[:n]

    # ---- render final fluent text ----
    def render(self):
        raw_words = self.words + ([self.letters] if self.letters else [])
        gloss_sentence = self._smooth_glosses(self.glosses) if self.glosses else ""
        letter_sentence = " ".join(raw_words)
        parts = [p for p in [gloss_sentence, letter_sentence] if p]
        return " ".join(parts).strip()

    def _smooth_glosses(self, glosses):
        if self.backend == "transformer":
            return self._smooth_with_transformer(glosses)
        return self._smooth_with_rules(glosses)

    def _smooth_with_rules(self, glosses):
        remaining = list(glosses)
        output = []

        # Greedy phrase-rule matching over the gloss stream.
        i = 0
        while i < len(remaining):
            matched = False
            for pattern, replacement in PHRASE_RULES:
                span = len(pattern)
                if remaining[i:i + span] == pattern:
                    output.append(replacement)
                    i += span
                    matched = True
                    break
            if not matched:
                word = remaining[i]
                output.append(PRONOUN_MAP.get(word, word.capitalize()))
                i += 1

        sentence = " ".join(output)
        if not sentence.endswith((".", "?", "!")):
            sentence += "."
        return sentence

    def _smooth_with_transformer(self, glosses):
        if self._transformer_pipeline is None:
            from transformers import pipeline  # imported lazily -- heavy dependency

            self._transformer_pipeline = pipeline(
                "text2text-generation", model=self.transformer_model
            )
        rough = " ".join(g.capitalize() for g in glosses)
        result = self._transformer_pipeline(
            f"Fix the grammar and make this fluent: {rough}", max_length=64
        )
        return result[0]["generated_text"]


def speak(text, engine=None):
    """Speak `text` aloud via pyttsx3. Pass an existing engine to avoid
    re-initializing it every call (re-init is slow)."""
    import pyttsx3

    own_engine = engine is None
    if own_engine:
        engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
    return engine if not own_engine else None
