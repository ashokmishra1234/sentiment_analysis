# Dockerfile for Smart NLP Text Analyzer
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
# 🔥 Download NLP resources (IMPORTANT)
RUN python -m textblob.download_corpora
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
