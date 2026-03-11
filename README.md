# 🧠 Smart NLP Text Analyzer API

A simple REST API built with **FastAPI** and **NLP** techniques to analyze text — sentiment, keywords, and summarization, complete with a **Streamlit** frontend interface.

---

## 🌐 Live Streamlit Application
**[Click Here to view the deployed Streamlit App](https://sentimentanalysis-fastapi.streamlit.app/)** 
*(Update this URL with your actual deployed Streamlit Community Cloud link)*

---

## 🚀 Tech Stack

| Technology      | Purpose                              |
|----------------|--------------------------------------|
| FastAPI         | Web framework for building REST API  |
| TextBlob        | Sentiment analysis (polarity score)  |
| NLTK            | Tokenization, stopwords              |
| Scikit-learn    | TF-IDF for keywords & summarization  |
| Pydantic        | Request/response data validation     |
| Uvicorn         | ASGI server to run FastAPI           |

---

## 📁 Project Structure

```
nlp_fastapi_project/
├── main.py            # All FastAPI routes and NLP logic
├── requirements.txt   # Python dependencies
└── README.md          # Project documentation
```

---

## ⚙️ Setup & Run

### 1. Clone / Navigate to project
```bash
cd nlp_fastapi_project
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate       # Mac/Linux
venv\Scripts\activate          # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Server
```bash
uvicorn main:app --reload
```

### 5. Open API Docs
Visit → [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 📡 API Endpoints

### `GET /`
Health check.

---

### `POST /sentiment`
Analyze sentiment of text.

**Request:**
```json
{ "text": "I love this amazing product!" }
```
**Response:**
```json
{
  "text": "I love this amazing product!",
  "sentiment": "Positive",
  "polarity": 0.625,
  "subjectivity": 0.6
}
```

---

### `POST /keywords`
Extract top 5 keywords using TF-IDF.

**Request:**
```json
{ "text": "Machine learning models are trained on large datasets to make predictions." }
```
**Response:**
```json
{
  "text": "...",
  "keywords": ["machine", "learning", "models", "predictions", "trained"]
}
```

---

### `POST /summarize`
Extractive summarization — picks the 2 most important sentences.

**Request:**
```json
{ "text": "A long paragraph with multiple sentences..." }
```
**Response:**
```json
{
  "original_text": "...",
  "summary": "Two most important sentences.",
  "original_word_count": 80,
  "summary_word_count": 25
}
```

---

## 🧪 Test with curl

```bash
curl -X POST "http://127.0.0.1:8000/sentiment" \
  -H "Content-Type: application/json" \
  -d '{"text": "FastAPI is really great for building APIs fast!"}'
```
