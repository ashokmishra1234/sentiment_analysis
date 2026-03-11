import streamlit as st
from textblob import TextBlob
from main import extract_keywords, extractive_summarize, get_sentiment_label

# --- Page Configuration ---
st.set_page_config(
    page_title="Smart NLP Text Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Aesthetics ---
st.markdown("""
<style>
    /* Styling for the metrics */
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        color: #4CAF50;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 1.1rem;
        font-weight: 600;
    }
    /* Simple headers */
    h1, h2, h3 {
        color: #1E88E5;
        font-family: 'Inter', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# --- Main App ---
st.title("🧠 Smart NLP Text Analyzer")
st.markdown(
    "Analyze your text using advanced **NLP techniques**. Extract keywords, "
    "determine sentiment, and generate extractive text summaries seamlessly!"
)

st.divider()

# Input area
text_input = st.text_area(
    "Enter text to analyze:", 
    height=250, 
    placeholder="Type or paste your text here to begin the magic..."
)

# Analyze Button
if st.button("🚀 Analyze Text", use_container_width=True, type="primary"):
    if not text_input.strip():
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        st.success("Analysis complete!")
        st.divider()
        
        # Columns layout
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("📊 Sentiment Analysis")
            st.info("Measures the emotion and tone of the text.")
            
            blob = TextBlob(text_input)
            polarity = round(blob.sentiment.polarity, 4)
            subjectivity = round(blob.sentiment.subjectivity, 4)
            sentiment_label = get_sentiment_label(polarity)
            
            st.metric(label="Overall Sentiment", value=sentiment_label)
            st.metric(label="Polarity (Tone)", value=f"{polarity}", delta="Positive > 0 > Negative", delta_color="off")
            st.metric(label="Subjectivity", value=f"{subjectivity}", delta="0 (Objective) to 1 (Subjective)", delta_color="off")
            
        with col2:
            st.subheader("🔑 Top Keywords")
            st.info("Extracts the most relevant terms using TF-IDF.")
            
            keywords = extract_keywords(text_input, top_n=5)
            if keywords:
                for i, kw in enumerate(keywords, 1):
                    st.markdown(f"**{i}.** `{kw}`")
            else:
                st.write("Could not extract meaningful keywords from the text.")
                
        with col3:
            st.subheader("📝 Extractive Summary")
            st.info("Highlights the most important sentences.")
            
            summary = extractive_summarize(text_input, num_sentences=2)
            st.write(summary)
            
            # Word Count Info
            st.markdown("---")
            original_words = len(text_input.split())
            summary_words = len(summary.split())
            st.caption(f"**Original Word Count**: {original_words}")
            st.caption(f"**Summary Word Count**: {summary_words}")
            if original_words > 0:
                reduction = round((1 - summary_words/original_words) * 100)
                st.caption(f"**Reduction**: {reduction}% smaller")
