---
title: YouTube Comment Sentiment Analyzer
emoji: 🎬
colorFrom: red
colorTo: indigo
sdk: streamlit
sdk_version: 1.35.0
app_file: streamlit_app.py
pinned: false
license: mit
---

<div align="center">

# 🎬 YouTube Comment Sentiment Analyzer & AI Creator Studio

An end-to-end, high-performance Machine Learning web application and MLOps system that extracts live YouTube comments in real-time and classifies audience feedback into **Positive**, **Neutral**, and **Negative** sentiments with interactive creator analytics.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![Python Version](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM-blue?logo=lightning)](https://lightgbm.readthedocs.io/)
[![Scikit-Learn](https://img.shields.io/badge/NLP-TF--IDF%20Trigram-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![DVC Tracked](https://img.shields.io/badge/Data%20Version%20Control-DVC-945DD6?logo=dvc&logoColor=white)](https://dvc.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[**Live Demo**](https://share.streamlit.io) • [**Report Bug**](https://github.com/sanjuDiatm-design/yt-comment-sentiment-analysis/issues) • [**Request Feature**](https://github.com/sanjuDiatm-design/yt-comment-sentiment-analysis/issues)

</div>

---

## 🌟 Key Features

### ⚡ 1. Real-Time Scraping Without API Keys
- Direct, multi-threaded InnerTube scraping pipeline capable of pulling hundreds of comments from any public YouTube video URL in seconds.
- **No Google Cloud API keys, quotas, or credentials required.**
- Embedded video player preview alongside instant extraction telemetry.

### 🧠 2. Dual-Engine Sentiment Intelligence
- **Trained Machine Learning Model:** High-speed **LightGBM** classifier trained on diverse social media comment datasets.
- **Sublinear Trigram TF-IDF Vectorization:** Captures 1-to-3 word context combinations for nuanced understanding.
- **Hinglish & Regional Slang Normalization:** Built-in linguistic mapping for Hindi-English transliterations (e.g., distinguishing colloquial *"mera / mere"* and *"ke baad"* from English negative false-friends).

### 📊 3. Creator Studio Dashboard & Visualizations
- **Executive Sentiment Snapshot:** Net Sentiment Score (NSS) verdict card with positive/neutral/negative percentage split.
- **Sentiment Gauge & Donut Charts:** Dynamic gauge ranging from -100 (Hostile) to +100 (Acclaimed).
- **Interactive Word Clouds:** High-contrast word clouds visualizing dominant positive praise and critical feedback themes.
- **Top Comment Spotlight:** Automatically detects and spotlights top praising comments and frequent viewer critiques.
- **Interactive Data Explorer:** Filter, search, and inspect cleaned tokens with 1-click **CSV Export**.

### 🧪 4. Flexible Input Modes
1. **Live YouTube URL Analysis:** Paste any link (standard, short, or mobile URLs).
2. **Single Comment Sandbox:** Test raw custom sentences and inspect TF-IDF token outputs.
3. **Bulk CSV Upload:** Upload any CSV containing a `Comment` column for instant batch classification.

---

## 🏗️ System Architecture

```text
       ┌────────────────────────┐
       │   YouTube Video URL    │  / CSV Batch / Single Text
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │  Extraction Engine     │ (Multi-tier InnerTube Scraper)
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │  NLP Preprocessing     │ (Regex, Hinglish Normalizer,
       │  & Lemmatization       │  WordNet Lemmatizer, Stopwords)
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │ Trigram TF-IDF (pkl)   │ (Feature Extraction)
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │   LightGBM Classifier  │ (Inference & Confidence Scoring)
       └───────────┬────────────┘
                   │
                   ▼
 ┌─────────────────────────────────────────────────────────┐
 │            Streamlit Creator Studio UI                  │
 │  • Video Player Preview    • Executive Sentiment Card   │
 │  • Net Sentiment Gauge     • Dual Word Clouds           │
 │  • Distribution Analytics  • Filterable Data & CSV Export│
 └─────────────────────────────────────────────────────────┘
```

---

## 🚀 One-Click Deployment to Streamlit Cloud

This repository is pre-configured and 100% ready for deployment on **[Streamlit Community Cloud](https://share.streamlit.io)**:

1. **Fork or Push** this repository to your GitHub account (`sanjuDiatm-design/yt-comment-sentiment-analysis`).
2. Log in at **[share.streamlit.io](https://share.streamlit.io)** using your GitHub credentials.
3. Click **New app** and configure:
   - **Repository:** `sanjuDiatm-design/yt-comment-sentiment-analysis`
   - **Branch:** `master`
   - **Main file path:** `streamlit_app.py`
4. Click **Deploy!**

> **Note:** The serialized model artifacts (`lgbm_model.pkl` ~4.1 MB and `tfidf_vectorizer.pkl` ~340 KB) are included directly in the repository, so no external AWS S3 bucket is required for the web application to run.

---

## 💻 Local Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/sanjuDiatm-design/yt-comment-sentiment-analysis.git
cd yt-comment-sentiment-analysis
```

### 2. Create and Activate Virtual Environment
```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Required Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Web Application
```bash
streamlit run streamlit_app.py
```
Open your browser at `http://localhost:8501`.

---

## 🐳 Docker Deployment

You can also run the application using Docker:

```bash
# Build the Docker image
docker build -t yt-sentiment-studio .

# Run the container
docker run -p 8501:8501 yt-sentiment-studio
```
Access the application at `http://localhost:8501`.

---

## 📁 Repository Structure

```text
yt-comment-sentiment-analysis/
├── .streamlit/
│   └── config.toml               # Streamlit custom dark theme configuration
├── src/                          # Model training & pipeline modules
│   ├── data/
│   │   ├── data_ingestion.py     # Data ingestion & train/test splitting
│   │   └── data_preprocessing.py # Text preprocessing & Hinglish regex mapping
│   └── model/
│       ├── model_building.py     # LightGBM training & vectorizer export
│       ├── model_evaluation.py   # Metrics computation & confusion matrix
│       └── register_model.py     # Model registry management
├── lgbm_model.pkl                # Trained LightGBM classifier (~4.1 MB)
├── tfidf_vectorizer.pkl          # Serialized TF-IDF feature extractor (~340 KB)
├── streamlit_app.py              # Main interactive Streamlit application
├── requirements.txt              # Production Python dependencies
├── runtime.txt                   # Target Python runtime (python-3.11)
├── params.yaml                   # Model & ingestion hyperparameters
├── dvc.yaml                      # DVC pipeline stages definition
├── Dockerfile                    # Containerization configuration
├── .gitignore                    # Git ignore specifications
└── README.md                     # Project documentation
```

---

## 🛠️ Technology Stack

| Domain | Technology / Library |
|---|---|
| **Frontend Framework** | [Streamlit](https://streamlit.io/) |
| **Machine Learning** | [LightGBM](https://lightgbm.readthedocs.io/), [scikit-learn](https://scikit-learn.org/) |
| **Natural Language Processing** | [NLTK](https://www.nltk.org/), Regular Expressions |
| **Interactive Visualizations** | [Plotly Express & Graph Objects](https://plotly.com/), [Matplotlib](https://matplotlib.org/), [WordCloud](https://github.com/amueller/word_cloud) |
| **Web Scraping** | InnerTube API Scraper + [youtube-comment-downloader](https://github.com/egbertbouman/youtube-comment-downloader) |
| **Data Manipulation** | [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/) |
| **MLOps & Versioning** | [DVC (Data Version Control)](https://dvc.org/) |

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!
1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'feat: Add AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

---

<div align="center">
Made with ❤️ by <a href="https://github.com/sanjuDiatm-design">sanjuDiatm-design</a>
</div>
