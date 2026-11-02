
import re
import heapq
import time
from io import BytesIO

import streamlit as st
import nltk
import numpy as np

from docx import Document
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from xml.sax.saxutils import escape

from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pypdf import PdfReader
from docx import Document


# ---------------- NLTK SETUP ----------------

@st.cache_resource
def setup_nltk():
    for package in ["punkt", "punkt_tab"]:
        try:
            nltk.data.find(f"tokenizers/{package}")
        except LookupError:
            nltk.download(package, quiet=True)

setup_nltk()


# ---------------- TEXT EXTRACTION ----------------

def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_text(uploaded_file):
    file_name = uploaded_file.name.lower()
    file_bytes = uploaded_file.getvalue()

    if file_name.endswith(".txt"):
        return file_bytes.decode("utf-8", errors="replace")

    if file_name.endswith(".pdf"):
        reader = PdfReader(BytesIO(file_bytes))
        return " ".join(
            page.extract_text() or "" for page in reader.pages
        )

    if file_name.endswith(".docx"):
        document = Document(BytesIO(file_bytes))
        return " ".join(
            paragraph.text for paragraph in document.paragraphs
        )

    raise ValueError("Unsupported file format.")


# ---------------- TF-IDF SUMMARIZATION ----------------

def tfidf_summary(sentences, summary_percent):
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(sentences)

    scores = matrix.mean(axis=1).A1

    count = max(
        1,
        min(
            len(sentences),
            round(len(sentences) * summary_percent / 100)
        )
    )

    selected = heapq.nlargest(
        count,
        range(len(sentences)),
        key=lambda i: scores[i]
    )

    return sorted(selected)


# ---------------- TEXTRANK SUMMARIZATION ----------------

def textrank_summary(sentences, summary_percent):
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(sentences)

    similarity = cosine_similarity(matrix)
    np.fill_diagonal(similarity, 0)

    row_sums = similarity.sum(axis=1, keepdims=True)

    transition = np.divide(
        similarity,
        row_sums,
        out=np.zeros_like(similarity),
        where=row_sums != 0
    )

    scores = np.ones(len(sentences)) / len(sentences)

    for _ in range(100):
        new_scores = (
            0.15 / len(sentences)
            + 0.85 * transition.T.dot(scores)
        )

        if np.allclose(scores, new_scores):
            scores = new_scores
            break

        scores = new_scores

    count = max(
        1,
        min(
            len(sentences),
            round(len(sentences) * summary_percent / 100)
        )
    )

    selected = heapq.nlargest(
        count,
        range(len(sentences)),
        key=lambda i: scores[i]
    )

    return sorted(selected)


# ---------------- KEYWORD EXTRACTION ----------------

def extract_keywords(text, number=10):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=100
    )

    matrix = vectorizer.fit_transform([text])
    scores = matrix.toarray()[0]
    terms = vectorizer.get_feature_names_out()

    ranked = sorted(
        zip(terms, scores),
        key=lambda item: item[1],
        reverse=True
    )

    return [term for term, score in ranked[:number]]

# ---------------- EXPORT FUNCTIONS ----------------

def create_docx(summary):
    buffer = BytesIO()

    document = Document()
    document.add_heading("SmartDoc - Generated Summary", 0)
    document.add_paragraph(summary)
    document.add_paragraph(
        "Generated using SmartDoc Document Summarization System."
    )

    document.save(buffer)
    buffer.seek(0)

    return buffer.getvalue()


def create_pdf(summary):
    buffer = BytesIO()

    pdf = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=0.7 * inch,
        leftMargin=0.7 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch
    )

    styles = getSampleStyleSheet()

    content = [
        Paragraph(
            "SmartDoc - Generated Summary",
            styles["Title"]
        ),
        Spacer(1, 20),
        Paragraph(
            escape(summary).replace("\n", "<br/>"),
            styles["BodyText"]
        ),
        Spacer(1, 20),
        Paragraph(
            "Generated using SmartDoc Document Summarization System.",
            styles["Italic"]
        )
    ]

    pdf.build(content)

    buffer.seek(0)
    return buffer.getvalue()


# ---------------- SUMMARIZATION ENGINE ----------------

def summarize_document(text, summary_percent, algorithm):
    text = clean_text(text)

    if not text:
        raise ValueError("Please provide a document.")

    sentences = sent_tokenize(text)

    if len(sentences) < 2:
        raise ValueError(
            "Please provide at least two complete sentences."
        )

    start = time.perf_counter()

    if algorithm == "TF-IDF":
        selected = tfidf_summary(sentences, summary_percent)
    else:
        selected = textrank_summary(sentences, summary_percent)

    elapsed = time.perf_counter() - start

    summary = " ".join(sentences[i] for i in selected)

    return summary, sentences, elapsed


# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="SmartDoc | Document Summarizer",
    page_icon="📄",
    layout="wide"
)


# ---------------- CUSTOM DESIGN ----------------

st.markdown("""
<style>
.main {
    background-color: #f7f8fc;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
.hero {
    padding: 25px;
    border-radius: 15px;
    background: linear-gradient(120deg, #243b55, #141e30);
    color: white;
    margin-bottom: 25px;
}
.hero h1 {
    color: white;
    font-size: 36px;
}
.hero p {
    color: #e0e6ed;
    font-size: 16px;
}
div[data-testid="stMetric"] {
    background: white;
    padding: 16px;
    border-radius: 12px;
    border: 1px solid #e6e8ef;
}
</style>
""", unsafe_allow_html=True)


# ---------------- SIDEBAR ----------------

st.sidebar.title("SmartDoc")
st.sidebar.caption("Document Intelligence System")

st.sidebar.divider()

algorithm = st.sidebar.selectbox(
    "Summarization Algorithm",
    ["TF-IDF", "TextRank"]
)

summary_percent = st.sidebar.slider(
    "Summary Length",
    min_value=10,
    max_value=60,
    value=30,
    step=10
)

st.sidebar.divider()

st.sidebar.info(
    "Upload a document or paste text to generate "
    "an extractive summary."
)


# ---------------- HEADER ----------------

st.markdown("""
<div class="hero">
    <h1>SmartDoc</h1>
    <p>AI-powered Document Summarization and Analysis</p>
    <p>Extract important information from lengthy documents.</p>
</div>
""", unsafe_allow_html=True)


# ---------------- INPUT SECTION ----------------

# ---------------- INPUT SECTION ----------------

st.subheader("1. Upload or Enter Your Document")

uploaded_file = st.file_uploader(
    "Upload a document",
    type=["txt", "pdf", "docx"]
)

document_text = ""

if uploaded_file is not None:
    try:
        document_text = extract_text(uploaded_file)
        st.success(f"Document uploaded successfully: {uploaded_file.name}")

    except Exception as error:
        st.error(f"Unable to read the file: {error}")

else:
    document_text = st.text_area(
        "Document Content",
        height=250,
        placeholder="Paste your document text here..."
    )


# ---------------- SUMMARIZATION ----------------

if st.button("Generate Summary", type="primary", use_container_width=True):

    if not document_text.strip():
        st.warning("Please upload a document or enter some text.")

    else:
        try:
            with st.spinner("Analysing your document..."):
                summary, sentences, processing_time = summarize_document(
                    document_text,
                    summary_percent,
                    algorithm
                )

                keywords = extract_keywords(document_text)

            st.session_state["summary"] = summary
            st.session_state["keywords"] = keywords
            st.session_state["sentences"] = sentences
            st.session_state["processing_time"] = processing_time
            st.session_state["algorithm"] = algorithm
            st.session_state["original_text"] = document_text

        except ValueError as error:
            st.error(str(error))
        except Exception as error:
            st.error(f"An error occurred: {error}")


# ---------------- RESULTS SECTION ----------------

if "summary" in st.session_state:

    summary = st.session_state["summary"]
    original = st.session_state["original_text"]

    original_words = len(original.split())
    summary_words = len(summary.split())

    reduction = (
        (1 - summary_words / original_words) * 100
        if original_words else 0
    )

    original_sentences = len(st.session_state["sentences"])

    summary_sentences = len(sent_tokenize(summary))

    reading_time = max(1, round(original_words / 200))

    st.divider()
    st.subheader("2. Document Analytics")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Original Words", original_words)
    col2.metric("Summary Words", summary_words)
    col3.metric("Compression", f"{reduction:.1f}%")
    col4.metric("Reading Time", f"{reading_time} min")

    col5, col6, col7 = st.columns(3)

    col5.metric("Total Sentences", original_sentences)
    col6.metric("Summary Sentences", summary_sentences)
    col7.metric(
        "Processing Time",
        f"{st.session_state['processing_time']:.3f} sec"
    )

    st.divider()
    st.subheader("3. Generated Summary")

    st.caption(
        f"Algorithm used: {st.session_state['algorithm']}"
    )

    st.write(summary)

        # Generate downloadable files
    docx_file = create_docx(summary)
    pdf_file = create_pdf(summary)

    st.markdown("### Download Your Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.download_button(
            label="Download TXT",
            data=summary,
            file_name="SmartDoc_Summary.txt",
            mime="text/plain",
            use_container_width=True
        )

    with col2:
        st.download_button(
            label="Download PDF",
            data=pdf_file,
            file_name="SmartDoc_Summary.pdf",
            mime="application/pdf",
            use_container_width=True
        )

    with col3:
        st.download_button(
            label="Download DOCX",
            data=docx_file,
            file_name="SmartDoc_Summary.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True
        )

    st.divider()
    st.subheader("4. Important Keywords")

    st.write("Keywords extracted from the original document:")

    st.write(
        "  •  ".join(st.session_state["keywords"])
    )

    


# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "SmartDoc | Information Retrieval Mini Project | "
    "Python • NLTK • Scikit-learn • Streamlit"
)
