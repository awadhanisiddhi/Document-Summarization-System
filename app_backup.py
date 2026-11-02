
import re
import heapq
import streamlit as st
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize

# Download sentence tokenizer data if needed
@st.cache_resource
def setup_nltk():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)

setup_nltk()


def clean_text(text):
    """Remove unnecessary whitespace and clean the document."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def summarize_document(text, summary_percent=30):
    """Generate an extractive summary using TF-IDF sentence ranking."""

    text = clean_text(text)

    if not text:
        return "Please enter some text to summarize."

    sentences = sent_tokenize(text)

    if len(sentences) < 2:
        return "Please enter at least two complete sentences."

    # Convert each sentence into numerical TF-IDF features
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(sentences)

    # Score sentences using their average TF-IDF weight
    scores = tfidf_matrix.mean(axis=1).A1

    # Select the highest-ranked sentences
    summary_count = max(
        1,
        min(len(sentences) - 1,
            round(len(sentences) * summary_percent / 100))
    )

    ranked_indices = heapq.nlargest(
        summary_count,
        range(len(sentences)),
        key=lambda i: scores[i]
    )

    # Preserve original sentence order in the final summary
    selected_indices = sorted(ranked_indices)

    summary = " ".join(sentences[i] for i in selected_indices)

    return summary


# ---------------- User Interface ----------------

st.set_page_config(
    page_title="Document Summarization System",
    page_icon="📄",
    layout="wide"
)

st.title("Document Summarization System")
st.write(
    "An NLP-based application that extracts important sentences "
    "from a document using TF-IDF ranking."
)

st.sidebar.header("Summary Settings")

summary_percent = st.sidebar.slider(
    "Summary length (%)",
    min_value=10,
    max_value=60,
    value=30,
    step=10
)

uploaded_file = st.file_uploader(
    "Upload a text file (.txt)",
    type=["txt"]
)

document_text = ""

if uploaded_file is not None:
    document_text = uploaded_file.getvalue().decode(
        "utf-8", errors="replace"
    )

document_text = st.text_area(
    "Enter or paste your document here",
    value=document_text,
    height=300,
    placeholder="Paste a document containing several sentences..."
)

if st.button("Summarize Document", type="primary"):
    if not document_text.strip():
        st.warning("Please enter text or upload a text file.")
    else:
        try:
            summary = summarize_document(
                document_text,
                summary_percent
            )

            st.subheader("Generated Summary")
            st.write(summary)

            original_words = len(document_text.split())
            summary_words = len(summary.split())

            col1, col2, col3 = st.columns(3)

            col1.metric("Original Words", original_words)
            col2.metric("Summary Words", summary_words)

            reduction = (
                (1 - summary_words / original_words) * 100
                if original_words else 0
            )

            col3.metric(
                "Word Reduction",
                f"{reduction:.1f}%"
            )

            st.download_button(
                "Download Summary",
                data=summary,
                file_name="summary.txt",
                mime="text/plain"
            )

        except ValueError:
            st.error(
                "The document could not be processed. "
                "Please enter a longer document with meaningful text."
            )
        except Exception as error:
            st.error(f"An error occurred: {error}")

st.divider()
st.caption(
    "Mini Project | Information Retrieval | "
    "Python, NLP, TF-IDF and Streamlit"
)
