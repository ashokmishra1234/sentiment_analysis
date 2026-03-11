"""
Smart NLP Text Analyzer API
============================
A FastAPI application that provides NLP features:
  - Sentiment Analysis
  - Keyword Extraction
  - Text Summarization (extractive)

Tech Stack: FastAPI + TextBlob + NLTK + Scikit-learn
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import re

# ─── Download required NLTK data ────────────────────────────────────────────
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("punkt_tab", quiet=True)

# ─── FastAPI App Setup ───────────────────────────────────────────────────────
app = FastAPI(
    title="Smart NLP Text Analyzer",
    description="An API that analyzes text using NLP techniques like Sentiment Analysis, Keyword Extraction, and Summarization.",
    version="1.0.0",
)


# ─── Pydantic Models (Request / Response Schemas) ────────────────────────────
class TextInput(BaseModel):
    text: str

    class Config:
        json_schema_extra = {
            "example": {
                "text": "FastAPI is an amazing framework for building APIs quickly and efficiently."
            }
        }


class SentimentResponse(BaseModel):
    text: str
    sentiment: str          # Positive / Negative / Neutral
    polarity: float         # -1.0 (most negative) to +1.0 (most positive)
    subjectivity: float     # 0.0 (objective) to 1.0 (subjective)


class KeywordResponse(BaseModel):
    text: str
    keywords: list[str]


class SummaryResponse(BaseModel):
    original_text: str
    summary: str
    original_word_count: int
    summary_word_count: int


# ─── Helper Functions ────────────────────────────────────────────────────────

def get_sentiment_label(polarity: float) -> str:
    """Convert polarity score to human-readable label."""
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"


def extract_keywords(text: str, top_n: int = 5) -> list[str]:
    """
    Extract top N keywords from text using TF-IDF.
    TF-IDF = Term Frequency × Inverse Document Frequency
    It finds words that are important in this text but rare overall.
    """
    # Clean text
    sentences = sent_tokenize(text)
    if len(sentences) < 2:
        # For short text, use simple word frequency
        stop_words = set(stopwords.words("english"))
        words = word_tokenize(text.lower())
        filtered = [w for w in words if w.isalpha() and w not in stop_words]
        freq = {}
        for word in filtered:
            freq[word] = freq.get(word, 0) + 1
        sorted_words = sorted(freq, key=freq.get, reverse=True)
        return sorted_words[:top_n]

    # Use TF-IDF for longer text
    vectorizer = TfidfVectorizer(stop_words="english", max_features=20)
    try:
        tfidf_matrix = vectorizer.fit_transform(sentences)
        feature_names = vectorizer.get_feature_names_out()
        # Sum TF-IDF scores across all sentences
        scores = np.array(tfidf_matrix.sum(axis=0)).flatten()
        top_indices = scores.argsort()[::-1][:top_n]
        return [feature_names[i] for i in top_indices]
    except Exception:
        return []


def extractive_summarize(text: str, num_sentences: int = 2) -> str:
    """
    Extractive summarization: pick the most important sentences.
    Uses TF-IDF scores to rank each sentence by importance.
    """
    sentences = sent_tokenize(text)
    if len(sentences) <= num_sentences:
        return text  # Already short enough

    # Score each sentence using TF-IDF
    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        tfidf_matrix = vectorizer.fit_transform(sentences)
        # Sentence score = sum of TF-IDF weights of its words
        sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
        # Pick top sentences (keep original order for readability)
        top_indices = sorted(
            np.argsort(sentence_scores)[::-1][:num_sentences]
        )
        summary = " ".join([sentences[i] for i in top_indices])
        return summary
    except Exception:
        return sentences[0]  # Fallback: return first sentence


# ─── API Endpoints ───────────────────────────────────────────────────────────

@app.get("/")
def root():
    """Health check and welcome message."""
    return {
        "message": "Welcome to Smart NLP Text Analyzer API 🚀",
        "docs": "/docs",
        "endpoints": ["/sentiment", "/keywords", "/summarize"],
    }


@app.post("/sentiment", response_model=SentimentResponse)
def analyze_sentiment(input_data: TextInput):
    """
    Analyzes the sentiment of the given text.

    - **Polarity**: Score from -1.0 (very negative) to +1.0 (very positive)
    - **Subjectivity**: Score from 0.0 (factual/objective) to 1.0 (opinion/subjective)
    - **Sentiment**: Positive / Negative / Neutral label
    """
    text = input_data.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    blob = TextBlob(text)
    polarity = round(blob.sentiment.polarity, 4)
    subjectivity = round(blob.sentiment.subjectivity, 4)
    sentiment_label = get_sentiment_label(polarity)

    return SentimentResponse(
        text=text,
        sentiment=sentiment_label,
        polarity=polarity,
        subjectivity=subjectivity,
    )


@app.post("/keywords", response_model=KeywordResponse)
def extract_keywords_endpoint(input_data: TextInput):
    """
    Extracts the top 5 most relevant keywords from the text using TF-IDF.

    Useful for: resume parsing, topic detection, content tagging.
    """
    text = input_data.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    keywords = extract_keywords(text, top_n=5)

    if not keywords:
        raise HTTPException(
            status_code=422,
            detail="Could not extract keywords. Please provide more meaningful text.",
        )

    return KeywordResponse(text=text, keywords=keywords)


@app.post("/summarize", response_model=SummaryResponse)
def summarize_text(input_data: TextInput):
    """
    Generates an extractive summary of the given text.

    Picks the 2 most important sentences using TF-IDF scoring.
    Best results with paragraphs (3+ sentences).
    """
    text = input_data.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    original_word_count = len(text.split())
    summary = extractive_summarize(text, num_sentences=2)
    summary_word_count = len(summary.split())

    return SummaryResponse(
        original_text=text,
        summary=summary,
        original_word_count=original_word_count,
        summary_word_count=summary_word_count,
    )
