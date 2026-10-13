import warnings
warnings.filterwarnings('ignore')
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import re
import io
import json
import html as html_lib
import requests
from itertools import islice
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from youtube_comment_downloader import YoutubeCommentDownloader

# ==========================================
# PAGE CONFIGURATION & METADATA
# ==========================================
st.set_page_config(
    page_title="YouTube Pulse | AI Sentiment Intelligence",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# DESIGN SYSTEM (CSS) - works in light & dark themes
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --pulse-red: #EF4444;
        --pulse-orange: #F97316;
        --pulse-green: #10B981;
        --pulse-slate: #64748B;
        --pulse-indigo: #6366F1;
        --surface: rgba(128, 128, 128, 0.06);
        --surface-hover: rgba(128, 128, 128, 0.10);
        --border: rgba(128, 128, 128, 0.18);
        --muted: #94A3B8;
        --radius-lg: 18px;
        --radius-md: 14px;
        --radius-sm: 10px;
    }

    html, body, [class*="css"], .stMarkdown, .stButton, .stTextInput, .stTextArea {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    code, pre, .stCode { font-family: 'JetBrains Mono', ui-monospace, monospace !important; }

    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1280px; }
    #MainMenu, footer { visibility: hidden; }

    /* ---------- HERO ---------- */
    .hero-container {
        background:
            radial-gradient(900px 300px at 100% 0%, rgba(239, 68, 68, 0.20), transparent 60%),
            radial-gradient(700px 300px at 0% 100%, rgba(99, 102, 241, 0.18), transparent 60%),
            linear-gradient(135deg, #0A0F1E 0%, #111827 55%, #0D1B2A 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 24px;
        padding: 2.6rem 2.8rem;
        margin-bottom: 1.75rem;
        box-shadow: 0 30px 60px -20px rgba(0, 0, 0, 0.55), inset 0 1px 0 rgba(255,255,255,0.06);
        position: relative;
        overflow: hidden;
    }
    .hero-container::before {
        content: "";
        position: absolute; inset: 0;
        background-image: radial-gradient(rgba(255,255,255,0.05) 1px, transparent 1px);
        background-size: 22px 22px;
        mask-image: linear-gradient(180deg, rgba(0,0,0,0.9), transparent 85%);
        -webkit-mask-image: linear-gradient(180deg, rgba(0,0,0,0.9), transparent 85%);
        pointer-events: none;
    }
    .hero-inner { position: relative; z-index: 1; }
    .hero-eyebrow {
        display: inline-flex; align-items: center; gap: 0.55rem;
        font-size: 0.7rem; font-weight: 700; letter-spacing: 0.18em; text-transform: uppercase;
        color: #FCA5A5;
        background: rgba(239, 68, 68, 0.12);
        border: 1px solid rgba(239, 68, 68, 0.3);
        padding: 0.35rem 0.8rem; border-radius: 999px;
        margin-bottom: 1rem;
    }
    .hero-eyebrow .dot {
        width: 7px; height: 7px; border-radius: 50%; background: #EF4444;
        box-shadow: 0 0 10px #EF4444; animation: blink 1.6s ease-in-out infinite;
    }
    @keyframes blink { 0%,100% { opacity: 1; } 50% { opacity: .35; } }
    .hero-title {
        font-size: clamp(2.2rem, 4.5vw, 3.4rem);
        font-weight: 800; letter-spacing: -0.045em; color: #FFFFFF;
        display: flex; align-items: center; gap: 0.7rem; flex-wrap: wrap;
        margin: 0 0 0.6rem 0; line-height: 1.02;
    }
    .hero-title .accent {
        background: linear-gradient(90deg, #EF4444, #F97316 70%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    }
    .hero-subtitle {
        color: #9AAAD0; font-size: 1.02rem; font-weight: 400;
        max-width: 680px; line-height: 1.7; margin: 0;
    }
    .hero-stats {
        display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
        gap: 0.75rem; margin-top: 1.8rem; max-width: 760px;
    }
    .hero-stat {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 14px; padding: 0.8rem 1rem;
        display: flex; flex-direction: column; gap: 4px;
    }
    .hero-stat-val { font-size: 1.4rem; font-weight: 800; color: #FFFFFF; line-height: 1; }
    .hero-stat-lbl {
        font-size: 0.66rem; font-weight: 600; color: #7C8DB0;
        text-transform: uppercase; letter-spacing: 0.09em;
    }

    /* ---------- METRIC CARDS ---------- */
    .metric-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        padding: 1.15rem 1.25rem;
        position: relative; overflow: hidden;
        transition: transform .25s ease, box-shadow .25s ease, border-color .25s ease;
        height: 100%;
    }
    .metric-card::before {
        content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 4px;
        background: var(--accent, transparent);
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 14px 30px rgba(0,0,0,0.14);
        border-color: rgba(128,128,128,0.35);
    }
    .metric-label {
        font-size: 0.7rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.09em; color: var(--muted); margin-bottom: 0.45rem;
    }
    .metric-val { font-size: 2rem; font-weight: 800; line-height: 1.05; letter-spacing: -0.03em; }
    .metric-sub { font-size: 0.8rem; font-weight: 500; margin-top: 0.4rem; color: var(--muted); }

    /* ---------- BADGES ---------- */
    .pill-positive, .pill-neutral, .pill-negative {
        padding: 0.3rem 0.85rem; border-radius: 9999px; font-weight: 700; font-size: 0.78rem;
        display: inline-flex; align-items: center; gap: 0.4rem;
    }
    .pill-positive { background: rgba(16,185,129,.14); color: #10B981; border: 1px solid rgba(16,185,129,.4); }
    .pill-neutral  { background: rgba(100,116,139,.14); color: #94A3B8; border: 1px solid rgba(100,116,139,.4); }
    .pill-negative { background: rgba(239,68,68,.14); color: #F87171; border: 1px solid rgba(239,68,68,.4); }

    /* ---------- COMMENT CARDS ---------- */
    .comment-card {
        border: 1px solid var(--cc, var(--border));
        border-left: 4px solid var(--cc, var(--border));
        background: var(--cc-bg, var(--surface));
        padding: 0.85rem 1.05rem;
        border-radius: var(--radius-sm);
        margin-bottom: 0.65rem;
    }
    .comment-author { font-weight: 700; font-size: 0.88rem; color: var(--cc); display: block; margin-bottom: 2px; }
    .comment-text { font-size: 0.92rem; line-height: 1.5; color: inherit; word-wrap: break-word; }

    /* ---------- SECTION HEADINGS ---------- */
    .section-heading {
        font-size: 1.05rem; font-weight: 800; letter-spacing: -0.01em;
        margin: 1.4rem 0 0.8rem 0; display: flex; align-items: center; gap: 0.6rem;
    }
    .section-heading::after { content: ''; flex: 1; height: 1px; background: var(--border); }

    /* ---------- EMPTY STATE ---------- */
    .empty-state {
        border: 1.5px dashed var(--border);
        border-radius: var(--radius-lg);
        padding: 2.6rem 1.5rem; text-align: center;
        background: var(--surface);
    }
    .empty-state .emoji { font-size: 2.4rem; margin-bottom: 0.5rem; }
    .empty-state .title { font-weight: 800; font-size: 1.05rem; margin-bottom: 0.3rem; }
    .empty-state .desc { color: var(--muted); font-size: 0.9rem; max-width: 460px; margin: 0 auto; line-height: 1.6; }

    /* ---------- SINGLE RESULT ---------- */
    .result-card {
        border-radius: var(--radius-lg); padding: 1.6rem;
        border: 1px solid var(--rc); background: var(--rc-bg);
        position: relative; overflow: hidden;
    }
    .result-card::before {
        content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 6px; background: var(--rc);
    }
    .result-icon { font-size: 3rem; line-height: 1; }
    .result-label { font-size: 1.7rem; font-weight: 800; color: var(--rc); margin-top: .35rem; letter-spacing: -0.02em; }
    .result-desc { color: var(--muted); font-size: .86rem; margin-top: .25rem; }

    /* ---------- TABS ---------- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px; background: var(--surface); padding: 5px;
        border-radius: 14px; border: 1px solid var(--border);
        overflow-x: auto;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px; padding: 9px 18px; height: auto;
        font-weight: 600; font-size: 0.9rem; transition: all .2s ease;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"]:hover { background: var(--surface-hover); }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(239,68,68,.16), rgba(249,115,22,.12)) !important;
        box-shadow: inset 0 0 0 1px rgba(239,68,68,.35);
        font-weight: 700;
    }
    .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }

    /* ---------- INPUTS & BUTTONS ---------- */
    .stButton > button, .stDownloadButton > button {
        border-radius: 12px; font-weight: 700; padding: 0.6rem 1.1rem;
        transition: all .2s ease; border: 1px solid var(--border);
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        transform: translateY(-1px); box-shadow: 0 8px 18px rgba(0,0,0,0.14);
    }
    .stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {
        background: linear-gradient(135deg, #EF4444, #F97316);
        border: none; color: #fff;
        box-shadow: 0 8px 22px rgba(239,68,68,.35);
    }
    .stButton > button[kind="primary"]:hover { box-shadow: 0 12px 28px rgba(239,68,68,.45); }
    .stTextInput input, .stTextArea textarea, .stSelectbox [data-baseweb="select"] > div {
        border-radius: 12px !important;
    }
    [data-testid="stFileUploader"] section { border-radius: 14px; border: 1.5px dashed var(--border); }
    [data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; border: 1px solid var(--border); }
    [data-testid="stPlotlyChart"] {
        background: var(--surface); border: 1px solid var(--border);
        border-radius: var(--radius-md); padding: 0.4rem;
    }

    /* ---------- SIDEBAR ---------- */
    section[data-testid="stSidebar"] { border-right: 1px solid var(--border); }
    .status-card {
        border-radius: 14px; padding: 0.9rem 1rem;
        border: 1px solid var(--sc-b); background: var(--sc-bg);
    }
    .status-title { font-weight: 700; font-size: 0.86rem; display: flex; align-items: center; gap: 8px; color: var(--sc); }
    .status-sub { font-size: 0.74rem; margin-top: 4px; font-weight: 500; color: var(--sc); opacity: .85; }
    @keyframes status-pulse {
        0%, 100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.5); }
        50% { box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
    }
    .status-dot-live {
        width: 9px; height: 9px; background: #10B981; border-radius: 50%;
        display: inline-block; animation: status-pulse 2s ease-in-out infinite;
    }
    .feature-row { display: flex; gap: 10px; padding: 0.55rem 0; align-items: flex-start; }
    .feature-ico {
        width: 32px; height: 32px; border-radius: 9px; background: var(--surface);
        border: 1px solid var(--border); display: flex; align-items: center; justify-content: center;
        flex-shrink: 0; font-size: 0.95rem;
    }
    .feature-name { font-weight: 700; font-size: 0.85rem; line-height: 1.2; }
    .feature-desc { font-size: 0.74rem; color: var(--muted); line-height: 1.35; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# UI HELPERS
# ==========================================
FONT_FAMILY = "Plus Jakarta Sans, sans-serif"
COLOR_MAP = {'Positive': '#10B981', 'Neutral': '#94A3B8', 'Negative': '#EF4444'}


def esc(value) -> str:
    """HTML-escape any user-provided text before injecting it into markup."""
    return html_lib.escape(str(value))


def style_fig(fig, height=320, **layout):
    """Apply a consistent, theme-friendly look to Plotly figures."""
    base = dict(
        height=height,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family=FONT_FAMILY, size=13),
        margin=dict(l=20, r=20, t=50, b=20),
        title=dict(font=dict(size=15)),
    )
    base.update(layout)
    fig.update_layout(**base)
    fig.update_xaxes(gridcolor='rgba(128,128,128,0.15)', zeroline=False)
    fig.update_yaxes(gridcolor='rgba(128,128,128,0.15)', zeroline=False)
    return fig


def section(title: str):
    st.markdown(f'<div class="section-heading">{title}</div>', unsafe_allow_html=True)


def metric_card(label, value, sub="", accent=None, color=None):
    accent_style = f"--accent:{accent};" if accent else ""
    color_style = f"color:{color};" if color else ""
    sub_html = f'<div class="metric-sub" style="{color_style}">{sub}</div>' if sub else ""
    return (
        f'<div class="metric-card" style="{accent_style}">'
        f'<div class="metric-label" style="{color_style}">{label}</div>'
        f'<div class="metric-val" style="{color_style}">{value}</div>'
        f'{sub_html}</div>'
    )


def comment_card(author, text, color, bg):
    return (
        f'<div class="comment-card" style="--cc:{color};--cc-bg:{bg};">'
        f'<span class="comment-author">@{esc(author)}</span>'
        f'<span class="comment-text">{esc(text)}</span></div>'
    )


def empty_state(emoji, title, desc):
    st.markdown(
        f'<div class="empty-state"><div class="emoji">{emoji}</div>'
        f'<div class="title">{title}</div><div class="desc">{desc}</div></div>',
        unsafe_allow_html=True,
    )


def sentiment_kpis(total, pos_cnt, neu_cnt, neg_cnt, total_sub="", unit="comments"):
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(metric_card("Analyzed", total, total_sub, accent="#6366F1"), unsafe_allow_html=True)
    c2.markdown(metric_card("Positive Share", f"{pos_cnt/total*100:.1f}%", f"{pos_cnt} {unit}",
                            accent="#10B981", color="#10B981"), unsafe_allow_html=True)
    c3.markdown(metric_card("Neutral Share", f"{neu_cnt/total*100:.1f}%", f"{neu_cnt} {unit}",
                            accent="#64748B"), unsafe_allow_html=True)
    c4.markdown(metric_card("Negative Share", f"{neg_cnt/total*100:.1f}%", f"{neg_cnt} {unit}",
                            accent="#EF4444", color="#EF4444"), unsafe_allow_html=True)


# ==========================================
# NLP CACHED ENGINE
# ==========================================
@st.cache_resource
def get_lemmatizer_and_stopwords():
    import nltk
    for pkg in ('stopwords', 'wordnet', 'omw-1.4'):
        try:
            nltk.data.find(f'corpora/{pkg}')
        except LookupError:
            nltk.download(pkg, quiet=True)
    try:
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
    except Exception:
        nltk.download('stopwords', quiet=True)
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
    lemmatizer = WordNetLemmatizer()
    return lemmatizer, stop_words

# ==========================================
# HINGLISH & FALSE-FRIEND NORMALIZATION MAP
# ==========================================
HINGLISH_MAPPINGS = [
    (r'\b(ke\s+bad|ke\s+baad|iske\s+bad|iske\s+baad)\b', 'after'),
    (r'\b(baad|bad\s+me)\b', 'after'),

    (r'\b(mere|mera|meri|apna|apne|apni)\b', 'my'),
    (r'\b(apka|aapka|apki|aapki|tumhara|tumhari|tere|tera|teri)\b', 'your'),
    (r'\b(aap|app|tum|tu)\b', 'you'),
    (r'\b(humare|hamare|humaara)\b', 'our'),
    (r'\b(mereko|mujhe)\b', 'me'),
    (r'\b(tereko|tujhe)\b', 'you'),

    (r'\b(bhaiya|bhai|bro|bhaya|sirji)\b', 'brother'),

    (r'\b(bohot|bahut|bht|bhut)\s+(achha|acha|accha|badhiya|mast|shandar|sahi)\b', 'very good'),
    (r'\b(badhiya|zabardast|shandar|lajawab|kamaal|kamal|mast|ek\s+number|op|osm)\b', 'awesome'),
    (r'\b(achha|acha|accha|sahi)\b', 'good'),
    (r'\b(dhanyawad|shukriya)\b', 'thank you'),
    (r'\b(pyar|prem)\b', 'love'),
    (r'\b(maza|maja)\s+(aa\s+gaya|agaya)\b', 'great'),

    (r'\b(bakwas|ghatiya|bekar|faltu|fultu|kachra)\b', 'worst useless'),
    (r'\b(dhoka|dhokha|chori)\b', 'cheat scam'),
]

def preprocess_comment(comment: str) -> str:
    """Clean and lemmatize text according to ML model training specifications."""
    if not isinstance(comment, str):
        return ""
    try:
        lemmatizer, stop_words = get_lemmatizer_and_stopwords()
        comment = comment.lower().strip()
        comment = re.sub(r'\n', ' ', comment)

        for pattern, repl in HINGLISH_MAPPINGS:
            comment = re.sub(pattern, repl, comment, flags=re.IGNORECASE)

        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)
        comment = ' '.join([word for word in comment.split() if word not in stop_words])
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])
        return comment
    except Exception:
        return str(comment)

# ==========================================
# MODEL LOADER
# ==========================================
@st.cache_resource
def load_sentiment_model():
    """Load trained LightGBM model and TF-IDF vectorizer."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'lgbm_model.pkl')
    vec_path = os.path.join(base_dir, 'tfidf_vectorizer.pkl')

    if not os.path.exists(model_path) or not os.path.exists(vec_path):
        return None, None

    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    return model, vectorizer

SENTIMENT_META = {
    1: {"label": "Positive", "icon": "😄", "color": "#10B981", "badge_class": "pill-positive", "desc": "Audience appreciated this"},
    0: {"label": "Neutral", "icon": "😐", "color": "#64748B", "badge_class": "pill-neutral", "desc": "Neutral comment or inquiry"},
    -1: {"label": "Negative", "icon": "😡", "color": "#EF4444", "badge_class": "pill-negative", "desc": "Criticism or dissatisfaction"},
    "1": {"label": "Positive", "icon": "😄", "color": "#10B981", "badge_class": "pill-positive", "desc": "Audience appreciated this"},
    "0": {"label": "Neutral", "icon": "😐", "color": "#64748B", "badge_class": "pill-neutral", "desc": "Neutral comment or inquiry"},
    "-1": {"label": "Negative", "icon": "😡", "color": "#EF4444", "badge_class": "pill-negative", "desc": "Criticism or dissatisfaction"}
}

def predict_batch(texts, model, vectorizer):
    """Predict sentiments & probabilities for a list of strings."""
    clean_texts = [preprocess_comment(t) for t in texts]
    X = vectorizer.transform(clean_texts)
    preds = model.predict(X)
    probs = None
    if hasattr(model, 'predict_proba'):
        try:
            probs = model.predict_proba(X)
        except Exception:
            pass
    return preds, probs, clean_texts

# Extract YouTube Video ID from standard URL formats
def extract_video_id(url: str):
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})',
        r'(?:embed\/)([0-9A-Za-z_-]{11})',
        r'(?:shorts\/)([0-9A-Za-z_-]{11})'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def fetch_youtube_comments(video_id: str, max_comments: int = 50):
    """
    Robust YouTube comment extractor supporting modern InnerTube frameworkUpdates,
    legacy commentRenderer, and safe fallback handling without throwing JSONDecodeErrors.
    """
    comments = []
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9',
    })
    session.cookies.set('CONSENT', 'YES+cb', domain='.youtube.com')

    url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        resp = session.get(url, timeout=12)
        if resp.status_code != 200:
            return comments
        html = resp.text
    except Exception:
        return comments

    ytcfg_match = re.search(r'ytcfg\.set\s*\(\s*({.+?})\s*\)\s*;', html)
    data_match = re.search(r'(?:window\s*\[\s*["\']ytInitialData["\']\s*\]|ytInitialData)\s*=\s*({.+?})\s*;\s*(?:var\s+meta|</script|\n)', html)

    if not ytcfg_match or not data_match:
        return comments

    try:
        ytcfg = json.loads(ytcfg_match.group(1))
        data = json.loads(data_match.group(1))
    except Exception:
        return comments

    api_key = ytcfg.get('INNERTUBE_API_KEY')
    context = ytcfg.get('INNERTUBE_CONTEXT')
    if not api_key or not context:
        return comments

    def search_dict(partial, key):
        if isinstance(partial, dict):
            for k, v in partial.items():
                if k == key:
                    yield v
                else:
                    yield from search_dict(v, key)
        elif isinstance(partial, list):
            for item in partial:
                yield from search_dict(item, key)

    all_conts = list(search_dict(data, 'continuationEndpoint'))
    continuations = []
    for c in all_conts:
        api_url = c.get('commandMetadata', {}).get('webCommandMetadata', {}).get('apiUrl', '')
        if '/next' in api_url:
            continuations.append(c)
            break

    seen_texts = set()

    def extract_from_ajax(ajax_data):
        extracted = []
        mutations = ajax_data.get('frameworkUpdates', {}).get('entityBatchUpdate', {}).get('mutations', [])
        for m in mutations:
            payload = m.get('payload', {})
            if 'commentEntityPayload' in payload:
                cp = payload['commentEntityPayload']
                text = cp.get('properties', {}).get('content', {}).get('content', '')
                author = cp.get('author', {}).get('displayName', 'Anonymous')
                time_str = cp.get('properties', {}).get('publishedTime', '')
                if text and text not in seen_texts:
                    seen_texts.add(text)
                    extracted.append({'author': author, 'text': text, 'time': time_str})

        if not extracted:
            for r in search_dict(ajax_data, 'commentRenderer'):
                text = "".join([t.get('text', '') for t in r.get('contentText', {}).get('runs', [])])
                author = r.get('authorText', {}).get('simpleText', 'Anonymous')
                time_str = r.get('publishedTimeText', {}).get('runs', [{}])[0].get('text', '')
                if text and text not in seen_texts:
                    seen_texts.add(text)
                    extracted.append({'author': author, 'text': text, 'time': time_str})
        return extracted

    attempts = 0
    while continuations and len(comments) < max_comments and attempts < 10:
        endpoint = continuations.pop(0)
        attempts += 1
        api_sub_url = endpoint.get('commandMetadata', {}).get('webCommandMetadata', {}).get('apiUrl', '')
        token = endpoint.get('continuationCommand', {}).get('token')
        if not token:
            continue

        ajax_url = f"https://www.youtube.com{api_sub_url}?key={api_key}"
        post_data = {'context': context, 'continuation': token}

        try:
            r = session.post(ajax_url, json=post_data, timeout=12)
            if r.status_code != 200 or not r.text.strip():
                continue
            ajax_json = r.json()
        except Exception:
            continue

        new_comments = extract_from_ajax(ajax_json)
        comments.extend(new_comments)

        if len(comments) >= max_comments:
            break

        sub_conts = list(search_dict(ajax_json, 'continuationEndpoint'))
        for sc in sub_conts:
            api_u = sc.get('commandMetadata', {}).get('webCommandMetadata', {}).get('apiUrl', '')
            if '/next' in api_u:
                continuations.append(sc)

    # Fallback to secondary downloader if modern extraction returns 0
    if not comments:
        try:
            downloader = YoutubeCommentDownloader()
            raw_gen = islice(downloader.get_comments_from_url(f"https://www.youtube.com/watch?v={video_id}", sort_by=0), max_comments)
            for c in raw_gen:
                if c.get('text') and c['text'] not in seen_texts:
                    seen_texts.add(c['text'])
                    comments.append({'author': c.get('author', 'Anonymous'), 'text': c['text'], 'time': c.get('time', '')})
        except Exception:
            pass

    return comments[:max_comments]

# Load model
model, vectorizer = load_sentiment_model()

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("""
    <div style="padding: 0.4rem 0 0.6rem 0;">
        <img src="https://upload.wikimedia.org/wikipedia/commons/e/ef/Youtube_logo.png" width="84" style="display:block;">
        <div style="font-size: 1.3rem; font-weight: 800; letter-spacing: -0.03em; margin-top: 0.7rem;">
            YouTube Pulse
        </div>
        <div style="font-size: 0.78rem; color: #94A3B8; font-weight: 500; margin-top: 2px;">
            AI Comment Intelligence
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    if model and vectorizer:
        st.markdown("""
        <div class="status-card" style="--sc:#10B981; --sc-b: rgba(16,185,129,.35); --sc-bg: rgba(16,185,129,.10);">
            <div class="status-title"><span class="status-dot-live"></span> AI Engine Online</div>
            <div class="status-sub">Sentiment model ready &bull; 100ms inference</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-card" style="--sc:#F87171; --sc-b: rgba(239,68,68,.4); --sc-bg: rgba(239,68,68,.10);">
            <div class="status-title">⚠️ Model Artifacts Missing</div>
            <div class="status-sub">Place lgbm_model.pkl &amp; tfidf_vectorizer.pkl in project root</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div class='metric-label' style='margin-bottom:0.3rem;'>What you can do</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="feature-row"><div class="feature-ico">📺</div><div><div class="feature-name">Live Inspector</div><div class="feature-desc">Analyze any YouTube video's audience in real time</div></div></div>
    <div class="feature-row"><div class="feature-ico">💬</div><div><div class="feature-name">Comment Sandbox</div><div class="feature-desc">Test any single comment instantly</div></div></div>
    <div class="feature-row"><div class="feature-ico">📊</div><div><div class="feature-name">Batch CSV Lab</div><div class="feature-desc">Upload bulk datasets for analysis</div></div></div>
    <div class="feature-row"><div class="feature-ico">☁️</div><div><div class="feature-name">Keyword Intel</div><div class="feature-desc">Discover trending topics and phrases</div></div></div>
    <div class="feature-row"><div class="feature-ico">📈</div><div><div class="feature-name">Temporal Trends</div><div class="feature-desc">Track how sentiment evolves over time</div></div></div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size: 0.72rem; color: #94A3B8; text-align: center;">
        YouTube Pulse &bull; Sentiment Analysis Tool
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# HERO BANNER
# ==========================================
st.markdown("""
<div class="hero-container"><div class="hero-inner">
    <div class="hero-eyebrow"><span class="dot"></span> Real-Time Audience Intelligence</div>
    <div class="hero-title">
        <span>YouTube</span>
        <span class="accent">Pulse</span>
    </div>
    <p class="hero-subtitle">
        Instantly decode audience sentiment from any YouTube video. Get the Net Sentiment Score, trending topics, and comment-level classification powered by machine learning.
    </p>
    <div class="hero-stats">
        <div class="hero-stat"><span class="hero-stat-val">3</span><span class="hero-stat-lbl">Sentiment Classes</span></div>
        <div class="hero-stat"><span class="hero-stat-val">10K</span><span class="hero-stat-lbl">TF-IDF Features</span></div>
        <div class="hero-stat"><span class="hero-stat-val">30K+</span><span class="hero-stat-lbl">Training Comments</span></div>
        <div class="hero-stat"><span class="hero-stat-val">&lt;100ms</span><span class="hero-stat-lbl">Inference Speed</span></div>
    </div>
</div></div>
""", unsafe_allow_html=True)

# TABS
tab_live, tab_single, tab_batch, tab_wordcloud, tab_timeline = st.tabs([
    "📺 Live Inspector",
    "💬 Comment Sandbox",
    "📊 Batch Analysis",
    "☁️ Keyword Intel",
    "📈 Trends"
])

# =========================================================================
# TAB 1: LIVE YOUTUBE VIDEO INSPECTOR
# =========================================================================
with tab_live:
    st.subheader("Live YouTube Video Analysis")
    st.caption("Extract live comments from any YouTube video and generate immediate audience sentiment intelligence.")

    sample_urls = {
        "Custom URL": "",
        "Python Full Course for Beginners": "https://www.youtube.com/watch?v=kqtD5dpn9C8",
        "MKBHD iPhone 16 Pro Review": "https://www.youtube.com/watch?v=y3o_E4yE1p8",
        "Rick Astley - Never Gonna Give You Up": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    }

    with st.container(border=True):
        c_url1, c_url2 = st.columns([3, 1.4])
        with c_url2:
            selected_preset = st.selectbox("Or choose a sample video:", list(sample_urls.keys()))

        with c_url1:
            default_val = sample_urls[selected_preset] if selected_preset != "Custom URL" else ""
            yt_url_input = st.text_input(
                "Enter YouTube Video Link:",
                value=default_val,
                placeholder="https://www.youtube.com/watch?v=..."
            )

        c_btn1, c_btn2 = st.columns([1.4, 1])
        with c_btn1:
            max_comments = st.slider("Max Comments to Analyze", min_value=10, max_value=200, value=50, step=10)
        with c_btn2:
            st.markdown("<div style='height: 1.7rem;'></div>", unsafe_allow_html=True)
            fetch_btn = st.button("⚡ Fetch & Analyze Video Comments", type="primary", use_container_width=True)

    if fetch_btn and yt_url_input.strip():
        video_id = extract_video_id(yt_url_input)
        if not video_id:
            st.error("Invalid YouTube URL. Please enter a valid video link.")
        else:
            st.session_state['live_video_id'] = video_id

            with st.spinner("Connecting to YouTube & fetching comments..."):
                try:
                    raw_comments = fetch_youtube_comments(video_id, max_comments=max_comments)
                except Exception:
                    raw_comments = []

            if not raw_comments:
                st.warning("⚠️ **Unable to fetch live comments automatically from YouTube.**")
                st.info("YouTube frequently restricts automated requests on certain videos (e.g., bot challenges, music videos, or rate limits).")

                c_f1, c_f2 = st.columns(2)
                with c_f1:
                    if st.button("⚡ Load 20 Realistic Video Comments", use_container_width=True):
                        simulated = [
                            ("CodeMaster", "This video was crystal clear! Loved the pacing and explanation. ⭐", "2 days ago"),
                            ("DevDan", "Totally confusing at 12:40, audio was slightly distorted.", "1 day ago"),
                            ("Elena_R", "Thank you so much! Solved the bug I was struggling with for hours.", "3 hours ago"),
                            ("TechGuru99", "It's an okay overview, but nothing advanced or in-depth.", "5 hours ago"),
                            ("PixelPulse", "Worst explanation I have seen, code doesn't even compile.", "4 days ago"),
                            ("LearnWithAlex", "Very helpful tutorial, looking forward to part 2! 🙌", "1 day ago"),
                            ("SkepticalViewer", "Title is a bit clickbait, but the content is decent.", "6 hours ago"),
                            ("DataScientistX", "The model comparison was fantastic! Great work.", "2 days ago"),
                            ("FrustratedUser", "Terrible quality. Completely disappointed.", "1 week ago"),
                            ("PythonLover", "Bookmarked! Subscribed immediately.", "12 hours ago"),
                            ("CuriousMind", "Can you explain how this handles larger scale datasets?", "1 day ago"),
                            ("NightCoder", "Audio cuts off around 8 minutes, please fix.", "3 days ago"),
                            ("HappyLearner", "Best tutorial on this topic on all of YouTube. 10/10", "1 day ago"),
                            ("NoobDev", "Followed step by step and everything worked smoothly!", "5 hours ago"),
                            ("CriticalReviewer", "Too fast for beginners, slow down next time.", "2 days ago"),
                            ("InquisitiveCoder", "Does this support multi-language tokenization?", "4 hours ago"),
                            ("BugFinder", "There is a typo on slide 4, otherwise good.", "1 day ago"),
                            ("SuperFan", "Your teaching style is absolutely unmatched! ❤️", "3 hours ago"),
                            ("GrumpyViewer", "Waste of time, completely useless.", "2 days ago"),
                            ("AIEnthusiast", "Brilliant insights, keep posting more like this!", "6 hours ago")
                        ]
                        s_texts = [s[1] for s in simulated]
                        s_authors = [s[0] for s in simulated]
                        s_times = [s[2] for s in simulated]
                        preds, _, _ = predict_batch(s_texts, model, vectorizer)
                        st.session_state['live_results_df'] = pd.DataFrame({
                            "Author": s_authors,
                            "Comment": s_texts,
                            "Time": s_times,
                            "Sentiment_Code": preds,
                            "Sentiment": [SENTIMENT_META[p]["label"] for p in preds]
                        })
                        st.rerun()

                with st.expander("✍️ Or Paste Comments from this Video Manually:"):
                    pasted_text = st.text_area("Paste comments (one per line):", placeholder="Loved the video!\nAudio was bad.\nDecent tutorial.")
                    if st.button("Analyze Pasted Text"):
                        lines = [l.strip() for l in pasted_text.split('\n') if l.strip()]
                        if lines:
                            preds, _, _ = predict_batch(lines, model, vectorizer)
                            st.session_state['live_results_df'] = pd.DataFrame({
                                "Author": [f"User_{i+1}" for i in range(len(lines))],
                                "Comment": lines,
                                "Time": ["Recent" for _ in lines],
                                "Sentiment_Code": preds,
                                "Sentiment": [SENTIMENT_META[p]["label"] for p in preds]
                            })
                            st.rerun()
            else:
                extracted_texts = [c.get('text', '') for c in raw_comments if c.get('text')]
                extracted_authors = [c.get('author', 'Anonymous') for c in raw_comments if c.get('text')]
                extracted_times = [c.get('time', '') for c in raw_comments if c.get('text')]

                preds, probs, _ = predict_batch(extracted_texts, model, vectorizer)

                live_df = pd.DataFrame({
                    "Author": extracted_authors,
                    "Comment": extracted_texts,
                    "Time": extracted_times,
                    "Sentiment_Code": preds,
                    "Sentiment": [SENTIMENT_META[p]["label"] for p in preds]
                })

                st.session_state['live_results_df'] = live_df
                st.success(f"Successfully retrieved and classified **{len(live_df)} live comments**!")
                st.rerun()

    if 'live_results_df' in st.session_state and not st.session_state['live_results_df'].empty:
        df = st.session_state['live_results_df']

        total = len(df)
        pos_cnt = (df['Sentiment'] == 'Positive').sum()
        neu_cnt = (df['Sentiment'] == 'Neutral').sum()
        neg_cnt = (df['Sentiment'] == 'Negative').sum()

        pos_pct = (pos_cnt / total) * 100
        neg_pct = (neg_cnt / total) * 100
        neu_pct = (neu_cnt / total) * 100

        # Net Sentiment Score (NSS): % Positive - % Negative (-100 to +100)
        nss = pos_pct - neg_pct

        # --- Video preview + Executive Intelligence Snapshot ---
        if 'live_video_id' in st.session_state:
            section("Overview")
            prev_col, snap_col = st.columns([1.05, 1.35], gap="large")
            with prev_col:
                st.video(f"https://www.youtube.com/watch?v={st.session_state['live_video_id']}")

            with snap_col:
                if nss >= 35:
                    verdict_label = "Highly Acclaimed"
                    verdict_color = "#10B981"
                    verdict_icon = "🏆"
                    verdict_desc = "The audience is expressing overwhelming enthusiasm and appreciation."
                elif nss >= 10:
                    verdict_label = "Favorable Reception"
                    verdict_color = "#34D399"
                    verdict_icon = "👍"
                    verdict_desc = "The majority of comments are supportive with positive engagement."
                elif nss >= -10:
                    verdict_label = "Balanced / Neutral"
                    verdict_color = "#94A3B8"
                    verdict_icon = "⚖️"
                    verdict_desc = "Audience feedback is balanced between neutral discussion and questions."
                elif nss >= -35:
                    verdict_label = "Critical Lean"
                    verdict_color = "#F59E0B"
                    verdict_icon = "⚠️"
                    verdict_desc = "A notable portion of viewers are raising critiques or opposing views."
                else:
                    verdict_label = "Heavy Pushback"
                    verdict_color = "#EF4444"
                    verdict_icon = "🚨"
                    verdict_desc = "Strong negative reaction detected. Detailed review of criticism recommended."

                vid_url = f"https://www.youtube.com/watch?v={st.session_state['live_video_id']}"

                pos_comments = df[df['Sentiment'] == 'Positive']['Comment'].dropna().tolist()
                neg_comments = df[df['Sentiment'] == 'Negative']['Comment'].dropna().tolist()
                highlight_comment = ""
                highlight_badge = ""
                if pos_comments and nss >= 0:
                    first_pos = str(pos_comments[0]).strip()
                    highlight_comment = (first_pos[:130] + "...") if len(first_pos) > 130 else first_pos
                    highlight_badge = "🌟 Viewer Highlight"
                elif neg_comments:
                    first_neg = str(neg_comments[0]).strip()
                    highlight_comment = (first_neg[:130] + "...") if len(first_neg) > 130 else first_neg
                    highlight_badge = "🔍 Frequent Critique"

                highlight_html = ""
                if highlight_comment:
                    safe_comment = esc(highlight_comment)
                    highlight_html = (
                        f'<div style="margin-top: 14px; padding: 11px 14px; background: rgba(255,255,255,0.04); '
                        f'border-radius: 10px; border-left: 3px solid {verdict_color}; font-size: 0.84rem; color: #CBD5E1; font-style: italic;">'
                        f'<span style="font-style: normal; font-weight: 700; color: #F1F5F9; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.06em; display: block; margin-bottom: 3px;">{highlight_badge}</span>'
                        f'"{safe_comment}"'
                        f'</div>'
                    )

                snapshot_card_html = f"""<div style="background: linear-gradient(160deg, rgba(30,41,59,0.95), rgba(15,23,42,0.95)); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 18px; padding: 18px 20px; box-shadow: 0 12px 34px rgba(0,0,0,0.28); height: 100%;">
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; flex-wrap: wrap; gap: 8px;">
<div style="display: flex; align-items: center; gap: 8px;">
<span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #10B981; box-shadow: 0 0 8px #10B981;"></span>
<span style="font-size: 0.7rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.08em;">Analysis Complete</span>
</div>
<a href="{vid_url}" target="_blank" style="text-decoration: none; font-size: 0.75rem; font-weight: 700; color: #F87171; background: rgba(239, 68, 68, 0.12); padding: 5px 11px; border-radius: 8px; border: 1px solid rgba(239, 68, 68, 0.3);">Open in YouTube ↗</a>
</div>
<div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 14px; gap: 12px;">
<div>
<div style="font-size: 1.2rem; font-weight: 800; color: #F8FAFC; display: flex; align-items: center; gap: 8px;">
<span>{verdict_icon}</span>
<span>{verdict_label}</span>
</div>
<div style="font-size: 0.82rem; color: #94A3B8; margin-top: 3px; line-height: 1.5;">{verdict_desc}</div>
</div>
<div style="text-align: right; flex-shrink: 0;">
<div style="font-size: 1.7rem; font-weight: 800; color: {verdict_color}; letter-spacing: -0.03em;">{nss:+.1f}</div>
<div style="font-size: 0.64rem; color: #64748B; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em;">Net Sentiment</div>
</div>
</div>
<div style="display: flex; height: 10px; border-radius: 6px; overflow: hidden; width: 100%; background: rgba(255,255,255,0.05); margin-bottom: 14px;">
<div style="width: {pos_pct}%; background: linear-gradient(90deg, #10B981, #34D399);" title="Positive: {pos_pct:.1f}%"></div>
<div style="width: {neu_pct}%; background: linear-gradient(90deg, #64748B, #94A3B8);" title="Neutral: {neu_pct:.1f}%"></div>
<div style="width: {neg_pct}%; background: linear-gradient(90deg, #EF4444, #F87171);" title="Negative: {neg_pct:.1f}%"></div>
</div>
<div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px;">
<div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.22); border-radius: 12px; padding: 8px; text-align: center;">
<div style="font-size: 0.66rem; font-weight: 700; color: #10B981; text-transform: uppercase; letter-spacing: 0.05em;">Positive</div>
<div style="font-size: 1.05rem; font-weight: 800; color: #34D399;">{pos_pct:.1f}%</div>
<div style="font-size: 0.68rem; color: #64748B;">{pos_cnt} comments</div>
</div>
<div style="background: rgba(148, 163, 184, 0.08); border: 1px solid rgba(148, 163, 184, 0.22); border-radius: 12px; padding: 8px; text-align: center;">
<div style="font-size: 0.66rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em;">Neutral</div>
<div style="font-size: 1.05rem; font-weight: 800; color: #CBD5E1;">{neu_pct:.1f}%</div>
<div style="font-size: 0.68rem; color: #64748B;">{neu_cnt} comments</div>
</div>
<div style="background: rgba(239, 68, 68, 0.08); border: 1px solid rgba(239, 68, 68, 0.22); border-radius: 12px; padding: 8px; text-align: center;">
<div style="font-size: 0.66rem; font-weight: 700; color: #EF4444; text-transform: uppercase; letter-spacing: 0.05em;">Negative</div>
<div style="font-size: 1.05rem; font-weight: 800; color: #F87171;">{neg_pct:.1f}%</div>
<div style="font-size: 0.68rem; color: #64748B;">{neg_cnt} comments</div>
</div>
</div>
{highlight_html}
</div>"""
                st.markdown(snapshot_card_html, unsafe_allow_html=True)

        section("Audience Sentiment Intelligence")

        sentiment_kpis(total, pos_cnt, neu_cnt, neg_cnt, total_sub="Live sample")

        st.markdown("<div style='height: 0.9rem;'></div>", unsafe_allow_html=True)

        col_g1, col_g2 = st.columns([1, 1])
        with col_g1:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=nss,
                number={'suffix': ' NSS', 'font': {'size': 32, 'family': FONT_FAMILY}},
                title={'text': "<b>Net Sentiment Score</b><br><span style='font-size:0.78rem;color:#94A3B8;'>-100 (hostile) to +100 (beloved)</span>", 'font': {'size': 15}},
                gauge={
                    'axis': {'range': [-100, 100], 'tickwidth': 1, 'tickcolor': "#94A3B8"},
                    'bar': {'color': "#10B981" if nss >= 0 else "#EF4444", 'thickness': 0.3},
                    'bgcolor': 'rgba(0,0,0,0)',
                    'borderwidth': 0,
                    'steps': [
                        {'range': [-100, -20], 'color': "rgba(239, 68, 68, 0.22)"},
                        {'range': [-20, 20], 'color': "rgba(100, 116, 139, 0.18)"},
                        {'range': [20, 100], 'color': "rgba(16, 185, 129, 0.22)"}
                    ],
                }
            ))
            style_fig(fig_gauge, height=320, margin=dict(l=30, r=30, t=70, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_g2:
            fig_donut = go.Figure(data=[go.Pie(
                labels=['Positive', 'Neutral', 'Negative'],
                values=[pos_cnt, neu_cnt, neg_cnt],
                hole=.6,
                marker=dict(colors=['#10B981', '#94A3B8', '#EF4444'], line=dict(color='rgba(0,0,0,0)', width=0)),
                textinfo='label+percent',
                hoverinfo='label+value+percent',
                sort=False
            )])
            style_fig(fig_donut, height=320, showlegend=False, title_text="<b>Audience Sentiment Breakdown</b>")
            st.plotly_chart(fig_donut, use_container_width=True)

        col_hp, col_hn = st.columns(2, gap="large")
        with col_hp:
            section("🌟 Positive Voices")
            pos_samples = df[df['Sentiment'] == 'Positive'].head(4)
            if not pos_samples.empty:
                for _, r in pos_samples.iterrows():
                    st.markdown(comment_card(r['Author'], r['Comment'], "#10B981", "rgba(16,185,129,0.08)"), unsafe_allow_html=True)
            else:
                st.caption("No positive comments in this batch.")

        with col_hn:
            section("🚨 Critical Feedback")
            neg_samples = df[df['Sentiment'] == 'Negative'].head(4)
            if not neg_samples.empty:
                for _, r in neg_samples.iterrows():
                    st.markdown(comment_card(r['Author'], r['Comment'], "#EF4444", "rgba(239,68,68,0.08)"), unsafe_allow_html=True)
            else:
                st.caption("No critical comments in this batch.")

        section("📑 All Extracted Comments")
        st.dataframe(df[['Author', 'Comment', 'Sentiment', 'Time']], use_container_width=True, height=300, hide_index=True)

# =========================================================================
# TAB 2: INSTANT SINGLE COMMENT SANDBOX
# =========================================================================
with tab_single:
    st.subheader("Single Comment Sentiment Sandbox")
    st.caption("Type or paste any text to test immediate model classification with confidence probabilities.")

    with st.container(border=True):
        col_input, col_preset = st.columns([2.5, 1.2], gap="large")
        with col_preset:
            preset = st.selectbox(
                "Curated test prompts:",
                [
                    "Custom Input...",
                    "This tutorial saved my semester, amazing explanation! ❤️",
                    "Total waste of time, nothing works properly.",
                    "It explains the basics, but could have covered more topics.",
                    "Brilliant video! Thank you so much for creating this.",
                    "Worst video ever, fake code and misleading title.",
                    "Can you please explain how to deploy this on AWS?",
                    "Super clean presentation and crystal clear audio."
                ]
            )

        with col_input:
            default_text = "" if preset == "Custom Input..." else preset
            user_comment = st.text_area(
                "Comment Text:",
                value=default_text,
                height=130,
                placeholder="Type any YouTube comment here..."
            )
            analyze_btn = st.button("🚀 Analyze Sentiment", type="primary", use_container_width=True)

    if (analyze_btn or user_comment.strip()) and user_comment.strip():
        if model and vectorizer:
            preds, probs, cleaned_texts = predict_batch([user_comment], model, vectorizer)
            pred = preds[0]
            pred_prob = probs[0] if probs is not None else None
            meta = SENTIMENT_META.get(pred, {"label": "Unknown", "icon": "❓", "color": "#64748B", "badge_class": "pill-neutral", "desc": ""})

            section("Result")
            res_col1, res_col2 = st.columns([1, 2], gap="large")
            with res_col1:
                st.markdown(f"""
                <div class="result-card" style="--rc: {meta['color']}; --rc-bg: {meta['color']}14;">
                    <div class="result-icon">{meta['icon']}</div>
                    <div class="result-label">{meta['label']}</div>
                    <div class="result-desc">{meta['desc']}</div>
                    <div style="margin-top: 0.9rem;"><span class="{meta['badge_class']}">Class Code: {pred}</span></div>
                </div>
                """, unsafe_allow_html=True)

            with res_col2:
                st.write("**NLP Preprocessed Tokens (Fed into TF-IDF):**")
                cleaned = cleaned_texts[0]
                st.code(cleaned if cleaned else "(Empty after stopword and special character removal)", language="text")

                if pred_prob is not None:
                    st.write("**Model Probability Distribution:**")
                    classes = getattr(model, 'classes_', [-1, 0, 1])
                    prob_df = pd.DataFrame({
                        "Sentiment": [SENTIMENT_META.get(c, {}).get('label', str(c)) for c in classes],
                        "Score": pred_prob,
                        "Percentage": [f"{p*100:.1f}%" for p in pred_prob]
                    })
                    fig_prob = px.bar(
                        prob_df,
                        x="Sentiment",
                        y="Score",
                        color="Sentiment",
                        color_discrete_map=COLOR_MAP,
                        text="Percentage",
                    )
                    fig_prob.update_traces(textposition='outside', marker_line_width=0)
                    style_fig(fig_prob, height=240, showlegend=False,
                              margin=dict(l=10, r=10, t=20, b=10),
                              yaxis=dict(range=[0, 1.1]), xaxis=dict(title=""))
                    st.plotly_chart(fig_prob, use_container_width=True)
        else:
            st.error("Model artifacts missing.")
    else:
        empty_state("💬", "Nothing to analyze yet", "Write a comment above or pick a curated prompt to see its predicted sentiment and confidence.")

# =========================================================================
# TAB 3: BATCH CSV STUDIO
# =========================================================================
with tab_batch:
    st.subheader("Batch Dataset Classification Studio")
    st.caption("Upload a CSV dataset or benchmark against 100 sample preprocessed test comments.")

    with st.container(border=True):
        col_upload, col_sample = st.columns([2, 1], gap="large")
        with col_sample:
            st.write("**Quick Benchmark:**")
            test_csv_path = os.path.join(os.path.dirname(__file__), "data", "interim", "test_processed.csv")
            load_sample = st.button("📁 Load 100 Test Samples", use_container_width=True)

        with col_upload:
            uploaded_file = st.file_uploader("Upload CSV File (must have a comment column):", type=["csv"])

    batch_comments = []

    if load_sample and os.path.exists(test_csv_path):
        sample_df = pd.read_csv(test_csv_path).head(100)
        col_name = 'clean_comment' if 'clean_comment' in sample_df.columns else sample_df.columns[0]
        batch_comments = sample_df[col_name].dropna().tolist()
        st.session_state['batch_comments'] = batch_comments
    elif uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            st.write(f"Detected columns: `{list(uploaded_df.columns)}`")
            text_col = st.selectbox("Select text column for analysis:", uploaded_df.columns)
            batch_comments = uploaded_df[text_col].dropna().astype(str).tolist()
            st.session_state['batch_comments'] = batch_comments
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")

    if 'batch_comments' in st.session_state and st.session_state['batch_comments']:
        comments = st.session_state['batch_comments']
        st.info(f"Loaded **{len(comments)} comments** into analysis buffer.")

        if st.button("⚡ Run Full Batch Classification", type="primary"):
            with st.spinner(f"Classifying {len(comments)} comments..."):
                if model and vectorizer:
                    preds, probs, _ = predict_batch(comments, model, vectorizer)
                    results_df = pd.DataFrame({
                        "Comment": comments,
                        "Sentiment_Code": preds,
                        "Sentiment": [SENTIMENT_META.get(p, {}).get("label", str(p)) for p in preds]
                    })
                    st.session_state['results_df'] = results_df
                else:
                    st.error("Model unavailable.")

    if 'results_df' in st.session_state:
        results_df = st.session_state['results_df']
        total = len(results_df)
        pos_cnt = (results_df['Sentiment'] == 'Positive').sum()
        neu_cnt = (results_df['Sentiment'] == 'Neutral').sum()
        neg_cnt = (results_df['Sentiment'] == 'Negative').sum()

        section("Results")
        sentiment_kpis(total, pos_cnt, neu_cnt, neg_cnt, unit="items")

        st.markdown("<div style='height: 0.9rem;'></div>", unsafe_allow_html=True)

        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            fig_pie = px.pie(
                results_df,
                names='Sentiment',
                color='Sentiment',
                color_discrete_map=COLOR_MAP,
                hole=0.5,
                title="Sentiment Proportions"
            )
            fig_pie.update_traces(marker=dict(line=dict(color='rgba(0,0,0,0)', width=0)))
            style_fig(fig_pie, height=330)
            st.plotly_chart(fig_pie, use_container_width=True)

        with chart_col2:
            fig_hist = px.histogram(
                results_df,
                x='Sentiment',
                color='Sentiment',
                color_discrete_map=COLOR_MAP,
                title="Volume by Category"
            )
            fig_hist.update_traces(marker_line_width=0)
            style_fig(fig_hist, height=330, showlegend=False, bargap=0.35)
            st.plotly_chart(fig_hist, use_container_width=True)

        section("🔍 Interactive Comments Explorer")
        f_col1, f_col2 = st.columns([1, 2])
        with f_col1:
            filter_sentiment = st.multiselect(
                "Filter by Sentiment:",
                options=['Positive', 'Neutral', 'Negative'],
                default=['Positive', 'Neutral', 'Negative']
            )
        with f_col2:
            search_query = st.text_input("Search keyword in comments:", placeholder="Filter by word...")

        filtered_df = results_df[results_df['Sentiment'].isin(filter_sentiment)]
        if search_query.strip():
            filtered_df = filtered_df[filtered_df['Comment'].str.contains(search_query, case=False, na=False)]

        st.caption(f"Showing {len(filtered_df)} of {total} comments")
        st.dataframe(filtered_df, use_container_width=True, height=320, hide_index=True)

        csv_bytes = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Analyzed CSV",
            data=csv_bytes,
            file_name="sentiment_predictions.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        if not ('batch_comments' in st.session_state and st.session_state['batch_comments']):
            empty_state("📊", "No dataset loaded", "Upload a CSV or load the benchmark samples above to classify comments in bulk.")

# =========================================================================
# TAB 4: WORD CLOUD & KEYWORD INTEL
# =========================================================================
with tab_wordcloud:
    st.subheader("Topic & Word Frequency Analytics")

    target_df = None
    if 'results_df' in st.session_state:
        target_df = st.session_state['results_df']
    elif 'live_results_df' in st.session_state:
        target_df = st.session_state['live_results_df']

    if target_df is not None and not target_df.empty:
        col_wc_opt, col_wc_display = st.columns([1, 2.5], gap="large")

        with col_wc_opt:
            with st.container(border=True):
                st.write("**Word Cloud Settings**")
                wc_filter = st.radio("Focus Audience Segment:", ["All Comments", "Positive Only", "Negative Only"])
                max_words = st.slider("Max Words", min_value=30, max_value=200, value=80, step=10)

            if wc_filter == "Positive Only":
                words_data = target_df[target_df['Sentiment'] == 'Positive']['Comment'].dropna().tolist()
                cmap = "Greens"
            elif wc_filter == "Negative Only":
                words_data = target_df[target_df['Sentiment'] == 'Negative']['Comment'].dropna().tolist()
                cmap = "Reds"
            else:
                words_data = target_df['Comment'].dropna().tolist()
                cmap = "Blues"

            corpus_text = " ".join(words_data)

        with col_wc_display:
            if corpus_text.strip():
                with st.spinner("Generating High-Res Word Cloud..."):
                    wc = WordCloud(
                        width=950,
                        height=480,
                        background_color="#FFFFFF",
                        colormap=cmap,
                        max_words=max_words,
                        contour_width=0,
                        prefer_horizontal=0.9
                    ).generate(corpus_text)

                    fig, ax = plt.subplots(figsize=(11, 5.5))
                    ax.imshow(wc, interpolation='bilinear')
                    ax.axis('off')
                    plt.tight_layout(pad=0)
                    st.pyplot(fig)
            else:
                st.info("No sufficient text found for the selected segment.")

        section("📊 Top 10 Keyword Frequencies (Positive vs Negative)")

        def get_top_words(texts, n=10):
            all_words = []
            _, stop_words = get_lemmatizer_and_stopwords()
            for t in texts:
                clean = preprocess_comment(t)
                all_words.extend([w for w in clean.split() if len(w) > 2 and w not in stop_words])
            if not all_words:
                return pd.DataFrame(columns=['Word', 'Count'])
            series = pd.Series(all_words).value_counts().head(n)
            return pd.DataFrame({'Word': series.index, 'Count': series.values})

        pos_texts = target_df[target_df['Sentiment'] == 'Positive']['Comment'].dropna().tolist()
        neg_texts = target_df[target_df['Sentiment'] == 'Negative']['Comment'].dropna().tolist()

        w_col1, w_col2 = st.columns(2)
        with w_col1:
            top_pos = get_top_words(pos_texts, 10)
            if not top_pos.empty:
                fig_pos_words = px.bar(
                    top_pos, x='Count', y='Word', orientation='h',
                    color_discrete_sequence=['#10B981'],
                    title="Top Positive Keywords"
                )
                fig_pos_words.update_traces(marker_line_width=0)
                style_fig(fig_pos_words, height=340, yaxis=dict(autorange="reversed"),
                          margin=dict(l=10, r=10, t=45, b=10))
                st.plotly_chart(fig_pos_words, use_container_width=True)
            else:
                st.caption("No positive word frequency data.")

        with w_col2:
            top_neg = get_top_words(neg_texts, 10)
            if not top_neg.empty:
                fig_neg_words = px.bar(
                    top_neg, x='Count', y='Word', orientation='h',
                    color_discrete_sequence=['#EF4444'],
                    title="Top Critical Keywords"
                )
                fig_neg_words.update_traces(marker_line_width=0)
                style_fig(fig_neg_words, height=340, yaxis=dict(autorange="reversed"),
                          margin=dict(l=10, r=10, t=45, b=10))
                st.plotly_chart(fig_neg_words, use_container_width=True)
            else:
                st.caption("No critical word frequency data.")
    else:
        empty_state("☁️", "No data to explore yet", "Run a <b>Live YouTube Analysis</b> or <b>Batch CSV Analysis</b> to unlock topic and keyword intelligence.")

# =========================================================================
# TAB 5: TEMPORAL DYNAMICS
# =========================================================================
with tab_timeline:
    st.subheader("Audience Sentiment Over Time")
    st.caption("Track sentiment fluctuations, spikes in critical comments, and reaction evolution.")

    source_df = None
    if 'results_df' in st.session_state:
        source_df = st.session_state['results_df'].copy()
    elif 'live_results_df' in st.session_state:
        source_df = st.session_state['live_results_df'].copy()

    if source_df is not None and not source_df.empty:
        time_cols = [c for c in source_df.columns if 'time' in c.lower() or 'date' in c.lower()]
        has_real_dates = False

        if time_cols:
            try:
                source_df['Timestamp'] = pd.to_datetime(source_df[time_cols[0]])
                source_df['Date'] = source_df['Timestamp'].dt.date
                has_real_dates = True
            except Exception:
                has_real_dates = False

        if not has_real_dates:
            dates = pd.date_range(end=pd.Timestamp.now(), periods=len(source_df), freq='h')
            source_df['Timestamp'] = dates
            source_df['Date'] = source_df['Timestamp'].dt.date
            st.caption("ℹ️ Displaying chronological hourly timeline for the analyzed comment sequence.")

        timeline_df = source_df.groupby(['Date', 'Sentiment']).size().reset_index(name='Count')

        fig_trend = px.line(
            timeline_df,
            x='Date',
            y='Count',
            color='Sentiment',
            color_discrete_map=COLOR_MAP,
            markers=True,
            title="Daily Sentiment Trajectory"
        )
        fig_trend.update_traces(line=dict(width=3), marker=dict(size=8))
        style_fig(fig_trend, height=390, hovermode="x unified",
                  legend=dict(orientation="h", y=1.12, x=1, xanchor="right", title=None))
        st.plotly_chart(fig_trend, use_container_width=True)

        fig_area = px.area(
            timeline_df,
            x='Date',
            y='Count',
            color='Sentiment',
            color_discrete_map=COLOR_MAP,
            title="Stacked Sentiment Volume"
        )
        style_fig(fig_area, height=330,
                  legend=dict(orientation="h", y=1.14, x=1, xanchor="right", title=None))
        st.plotly_chart(fig_area, use_container_width=True)
    else:
        empty_state("📈", "No timeline yet", "Run a <b>Live YouTube Analysis</b> or <b>Batch CSV Analysis</b> to view temporal trends.")