# 🎬 YouTube Comment Sentiment Analyzer & Creator Studio

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-blue)
![Scikit--Learn](https://img.shields.io/badge/Vectorization-TF--IDF%20Trigram-orange?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

An end-to-end, high-performance Machine Learning web application built with **Streamlit** that fetches live YouTube comments and classifies audience feedback into **Positive**, **Neutral**, and **Negative** sentiments with real-time analytics.

---

## 🌟 Key Highlights

- ⚡ **Zero-API-Key Live Scraping:** Analyze comments from any public YouTube video URL directly using a resilient multi-tier extraction engine (no Google Cloud API quota or credentials needed).
- 🧠 **Trained ML Architecture:** High-speed **LightGBM** classifier paired with sublinear **Trigram TF-IDF** vectorization for nuanced sentiment prediction.
- 🇮🇳 **Hinglish & Slang Normalization:** Built-in false-friend disambiguation for mixed Hindi-English phrases (e.g., distinguishing colloquial Hindi *"mere/mera"* and *"ke baad"* from English negative false-friends).
- 📊 **Executive Analytics & Visualizations:**
  - Sentiment breakdown doughnut charts and sentiment health score gauges
  - Dual positive/negative interactive **Word Clouds**
  - High-confidence classification telemetry & prediction distributions
  - Keyword and entity extraction for audience voice analysis
- 📥 **Interactive Data Export:** Inspect preprocessed tokens and filter comments with one-click **CSV export**.

---

## 📸 App Preview & Workflow

```text
  [ YouTube URL / Manual Text / CSV ]
                   │
                   ▼
  [ Preprocessing & Hinglish Normalizer ]
                   │
                   ▼
  [ Trigram TF-IDF Vectorizer (pkl) ]
                   │
                   ▼
       [ LightGBM Model (pkl) ]
                   │
                   ▼
 [ Streamlit Dashboard: Metrics | WordClouds | Filterable Data ]
```

---

## 🚀 Deployment to Streamlit Community Cloud

This repository is pre-configured and 100% ready for one-click deployment on **[Streamlit Community Cloud](https://share.streamlit.io)**:

1. **Fork or Push** this repository to your GitHub account:
   ```bash
   git add .
   git commit -m "Deploy: Prepare repository for Streamlit Community Cloud"
   git push origin master
   ```
2. Go to **[share.streamlit.io](https://share.streamlit.io)** and log in with GitHub.
3. Click **New app** and configure:
   - **Repository:** `your-username/yt-comment-sentiment-analysis`
   - **Branch:** `master`
   - **Main file path:** `streamlit_app.py`
4. Click **Deploy!** — Streamlit will automatically install dependencies from `requirements.txt` and launch the app in seconds.

> **Note:** The model artifacts (`lgbm_model.pkl` and `tfidf_vectorizer.pkl`, ~4.5 MB total) are included directly in the repository, so no external storage bucket setup is required.

---

## 💻 Local Development Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/yt-comment-sentiment-analysis.git
cd yt-comment-sentiment-analysis
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
```bash
streamlit run streamlit_app.py
```
Open your browser at `http://localhost:8501`.

---

## 📂 Project Structure

```text
yt-comment-sentiment-analysis/
├── .streamlit/
│   └── config.toml               # Custom dark theme & server configuration
├── src/                          # Model training & preprocessing pipelines
│   ├── data/
│   │   ├── data_ingestion.py
│   │   └── data_preprocessing.py
│   └── model/
│       ├── model_building.py
│       ├── model_evaluation.py
│       └── register_model.py
├── lgbm_model.pkl                # Trained LightGBM sentiment model (~4.1 MB)
├── tfidf_vectorizer.pkl          # Serialized TF-IDF feature extractor (~340 KB)
├── streamlit_app.py              # Main interactive Streamlit application
├── requirements.txt              # Production deployment dependencies
├── runtime.txt                   # Target Python runtime (python-3.11)
├── .gitignore                    # Deployment-optimized ignore rules
└── README.md                     # Project documentation
```

---

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| **Frontend & UI** | [Streamlit](https://streamlit.io/) |
| **Machine Learning** | [LightGBM](https://lightgbm.readthedocs.io/), [scikit-learn](https://scikit-learn.org/) |
| **Natural Language Processing** | [NLTK](https://www.nltk.org/), Regular Expressions |
| **Interactive Charts** | [Plotly](https://plotly.com/), [Matplotlib](https://matplotlib.org/), [WordCloud](https://github.com/amueller/word_cloud) |
| **YouTube Scraper** | Custom InnerTube Scraper + [youtube-comment-downloader](https://github.com/egbertbouman/youtube-comment-downloader) |
| **Data Handling** | [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/) |

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
