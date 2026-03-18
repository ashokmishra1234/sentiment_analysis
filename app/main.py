"""
main.py
=======
FastAPI application entry point.
Only contains:
  - App initialization
  - Route definitions
  - Request/response handling

All logic is separated into:
  - schema.py   → Pydantic models
  - train.py    → Model/NLP setup
  - predict.py  → Prediction functions
"""

from fastapi import FastAPI, HTTPException
from app.schema import TextInput, SentimentResponse, KeywordResponse, SummaryResponse
from app.predict import predict_sentiment, predict_keywords, predict_summary
from app.train import download_nltk_resources

# ─── Download NLTK resources on startup ──────────────────────────────────────
@app.on_event("startup")
def startup_event():
    download_nltk_resources()

# ─── FastAPI App ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Smart NLP Text Analyzer",
    description="Analyze text using Sentiment Analysis, Keyword Extraction, and Summarization.",
    version="1.0.0",
)


# ─── ROUTES ───────────────────────────────────────────────────────────────────

@app.get("/")
@app.get("/root")
@app.get("/Root")
def root():
    """Health check."""
    return {
        "message": "Welcome to Smart NLP Text Analyzer API 🚀",
        "docs": "/docs",
        "endpoints": ["/sentiment", "/keywords", "/summarize"],
    }


@app.post("/sentiment", response_model=SentimentResponse)
@app.post("/Sentiment", response_model=SentimentResponse)
def analyze_sentiment(input_data: TextInput):
    """
    Analyze sentiment of the given text.
    Returns: sentiment label, polarity score, subjectivity score.
    """
    text = input_data.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    result = predict_sentiment(text)

    return SentimentResponse(text=text, **result)


@app.post("/keywords", response_model=KeywordResponse)
@app.post("/Keywords", response_model=KeywordResponse)
def extract_keywords(input_data: TextInput):
    """
    Extract top 5 keywords from text using TF-IDF.
    """
    text = input_data.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    keywords = predict_keywords(text, top_n=5)

    if not keywords:
        raise HTTPException(
            status_code=422,
            detail="Could not extract keywords. Please provide more meaningful text.",
        )

    return KeywordResponse(text=text, keywords=keywords)


@app.post("/summarize", response_model=SummaryResponse)
@app.post("/Summarize", response_model=SummaryResponse)
def summarize_text(input_data: TextInput):
    """
    Extractive summarization — returns the 2 most important sentences.
    Best results with 3+ sentence paragraphs.
    """
    text = input_data.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    original_word_count = len(text.split())
    summary             = predict_summary(text, num_sentences=2)
    summary_word_count  = len(summary.split())

    return SummaryResponse(
        original_text=text,
        summary=summary,
        original_word_count=original_word_count,
        summary_word_count=summary_word_count,
    )
