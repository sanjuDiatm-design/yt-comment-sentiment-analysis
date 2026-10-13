FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

COPY . /app/

RUN pip install --no-cache-dir streamlit plotly joblib lightgbm matplotlib nltk "numpy<2" pandas wordcloud scikit-learn pyyaml

RUN python -m nltk.downloader stopwords wordnet

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]