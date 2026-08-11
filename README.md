# SignLingo
### Real-Time Multilingual Sign Language Translator (ASL • BSL • ISL)

SignLingo is an AI-powered sign language recognition system that translates hand gestures into **text and speech** in real time using computer vision and machine learning.

The project supports **American Sign Language (ASL)**, **British Sign Language (BSL)**, and **Indian Sign Language (ISL)** through a unified landmark-based pipeline built with **MediaPipe**, **Scikit-learn**, **PyTorch**, and **Streamlit WebRTC**.

---

# ✨ Features

- 🌍 Multilingual support
  - American Sign Language (ASL)
  - British Sign Language (BSL)
  - Indian Sign Language (ISL)

- ✋ Real-time webcam recognition

- 🤖 MediaPipe hand landmark extraction

- 🧠 Machine Learning based classification
  - MLP classifier for fingerspelled letters
  - Bidirectional LSTM for word recognition

- 📝 Sentence construction with NLP

- 🔊 Text-to-Speech output

- 📹 Browser-based webcam using Streamlit WebRTC

- 🏗 Modular training pipeline for adding new languages and datasets

---

# 🛠 Technology Stack

| Category | Technologies |
|----------|--------------|
| Programming | Python |
| Computer Vision | OpenCV, MediaPipe |
| Machine Learning | Scikit-learn, PyTorch |
| NLP | Rule-based Sentence Builder |
| UI | Streamlit, Streamlit-WebRTC |
| Data Processing | NumPy, Pandas |

---

# 📂 Project Structure

```text
signlingo/
│
├── app/
│   ├── app.py
│   └── app_webrtc.py
│
├── models/
│
├── nlp/
│
├── scripts/
│
├── training/
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# ⚙️ System Pipeline

```text
Raw Images / Videos
        │
        ▼
MediaPipe Hand Landmark Extraction
        │
        ▼
Unified Landmark Dataset
        │
        ├────────► Letter Recognition (MLP)
        │
        └────────► Word Recognition (Bi-LSTM)
                        │
                        ▼
            Sentence Builder (NLP)
                        │
                        ▼
               Text-to-Speech Engine
                        │
                        ▼
              Streamlit Web Application
```

---

# Model Architecture

### Letter Recognition

- MediaPipe Hands
- 21 hand landmarks
- Two-hand unified landmark schema
- MLPClassifier (Scikit-learn)

### Word Recognition

- Landmark sequence extraction
- Bidirectional LSTM
- Sequence classification

### Sentence Generation

Predicted letters and glosses are accumulated into a rolling buffer before being transformed into natural language using a lightweight NLP sentence builder. The generated sentence is then spoken using a text-to-speech engine.

---

# 📊 Performance

| Model | Accuracy |
|---------|---------:|
| Multilingual Letter Classifier | **98.8%** |

---

# 📁 Datasets

The project uses publicly available datasets for:

- American Sign Language (ASL)
- British Sign Language (BSL)
- Indian Sign Language (ISL)

The datasets are **not included** in this repository due to size and licensing restrictions.

After downloading them, place them inside:

```text
dataset/
```

Then generate landmarks using:

```bash
python scripts/extract_landmarks_letters.py
```

---

# 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/signlingo.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch the application:

```bash
streamlit run app/app_webrtc.py
```

---

# 📈 Future Improvements

- Continuous sentence recognition
- Transformer-based language translation
- Mobile application
- TensorFlow Lite deployment
- Additional sign language support
- Cloud deployment
