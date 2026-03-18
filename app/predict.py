"""
predict.py
==========
All NLP prediction/inference functions.
Each function takes raw text and returns a result.

Functions:
  - predict_sentiment()   → uses TextBlob
  - predict_keywords()    → uses TF-IDF (scikit-learn)
  - predict_summary()     → uses TF-IDF sentence scoring (extractive)
"""

import numpy as np
from textblob import TextBlob
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from train import get_tfidf_vectorizer


# ─── SENTIMENT PREDICTION ─────────────────────────────────────────────────────

def get_sentiment_label(polarity: float) -> str:
    """
    Map polarity score → human-readable label.
      > 0.1  → Positive
      < -0.1 → Negative
      else   → Neutral
    """
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"


def predict_sentiment(text: str) -> dict:
    blob = TextBlob(text)
    polarity     = round(blob.sentiment.polarity,     4)
    subjectivity = round(blob.sentiment.subjectivity, 4)

    return {
        "sentiment":    get_sentiment_label(polarity),
        "polarity":     polarity,
        "subjectivity": subjectivity,
    }


# ─── KEYWORD PREDICTION ───────────────────────────────────────────────────────

def predict_keywords(text: str, top_n: int = 5) -> list[str]:
   
  
    sentences = sent_tokenize(text)

    # Short text fallback — use word frequency
    if len(sentences) < 2:
        stop_words = set(stopwords.words("english"))
        words      = word_tokenize(text.lower())
        filtered   = [w for w in words if w.isalpha() and w not in stop_words]
        freq       = {}
        for word in filtered:
            freq[word] = freq.get(word, 0) + 1
        return sorted(freq, key=freq.get, reverse=True)[:top_n]

    # TF-IDF for longer text
    try:
        vectorizer   = get_tfidf_vectorizer()
        tfidf_matrix = vectorizer.fit_transform(sentences)
        feature_names = vectorizer.get_feature_names_out()
        scores        = np.array(tfidf_matrix.sum(axis=0)).flatten()
        top_indices   = scores.argsort()[::-1][:top_n]
        return [feature_names[i] for i in top_indices]
    except Exception:
        return []


# ─── SUMMARIZATION PREDICTION ─────────────────────────────────────────────────

def predict_summary(text: str, num_sentences: int = 2) -> str:
    sentences = sent_tokenize(text)

    # Already short enough
    if len(sentences) <= num_sentences:
        return text

    try:
        vectorizer      = get_tfidf_vectorizer()
        tfidf_matrix    = vectorizer.fit_transform(sentences)
        sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()

        # Sort by score, pick top N, restore original order
        top_indices = sorted(
            np.argsort(sentence_scores)[::-1][:num_sentences]
        )
        return " ".join([sentences[i] for i in top_indices])

    except Exception:
        return sentences[0]  # Fallback: return first sentence
