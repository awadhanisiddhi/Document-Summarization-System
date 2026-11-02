# SmartDoc – Document Summarization System

**An NLP-based document summarization and analysis application built using Python and Streamlit.**

SmartDoc is a web-based application that generates concise summaries from lengthy documents using Natural Language Processing (NLP) techniques. It implements two extractive summarization algorithms — **TF-IDF and TextRank** — to identify and extract the most important sentences from a document.

The application also provides document analytics, keyword extraction, and downloadable summaries in multiple formats.

## Features

- **Multiple File Formats:** Upload documents in TXT, PDF, and DOCX formats.
- **Text Input:** Paste document content directly into the application.
- **Dual Summarization Algorithms:** Choose between TF-IDF and TextRank.
- **Customizable Summary Length:** Adjust the summary length from 10% to 60%.
- **Document Analytics:** View original word count, summary word count, compression percentage, reading time, sentence count, and processing time.
- **Keyword Extraction:** Identify important keywords and phrases from the original document using TF-IDF.
- **Multiple Download Formats:** Download generated summaries in TXT, PDF, and DOCX formats.
- **Interactive Interface:** Use a Streamlit-based interface to upload documents, configure summarization settings, and view results.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Web application interface |
| NLTK | Sentence tokenization |
| Scikit-learn | TF-IDF vectorization and cosine similarity |
| NumPy | Numerical computations |
| PyPDF | PDF text extraction |
| python-docx | DOCX file processing and generation |
| ReportLab | PDF generation |
| Heapq | Selecting top-ranked sentences |

## Summarization Algorithms

### 1. TF-IDF (Term Frequency–Inverse Document Frequency)

TF-IDF is used to calculate the importance of words across sentences in a document.

**Working:**
1. The input document is cleaned and divided into sentences.
2. TF-IDF vectorization converts sentences into numerical representations.
3. Sentence scores are calculated using their average TF-IDF weights.
4. The highest-ranked sentences are selected according to the chosen summary length.
5. Selected sentences are arranged in their original order to generate the summary.

### 2. TextRank

TextRank is a graph-based ranking algorithm inspired by Google's PageRank algorithm.

**Working:**
1. The document is divided into individual sentences.
2. TF-IDF vectors are generated for the sentences.
3. Cosine similarity is calculated to measure relationships between sentences.
4. A similarity graph is constructed and normalized.
5. PageRank-style iterative scoring identifies the most important sentences.
6. The highest-ranked sentences are selected and arranged in their original order.

Both algorithms perform **extractive summarization**, meaning they select important sentences from the original document rather than generating entirely new sentences.

## Application Workflow

1. Upload a TXT, PDF, or DOCX document, or paste text directly.
2. Select the summarization algorithm (TF-IDF or TextRank).
3. Adjust the desired summary length using the slider.
4. Click **Generate Summary**.
5. View the generated summary and document analytics.
6. Explore important keywords extracted from the document.
7. Download the summary in TXT, PDF, or DOCX format.

## Project Structure

```text
DocumentSummarization/
│
├── app.py                 # Main Streamlit application
├── app_backup.py          # Backup of the original application
├── requirements.txt       # Python dependencies
├── .gitignore             # Git ignored files
└── README.md              # Project documentation
```

## Installation and Setup

### Prerequisites

- Python 3.10 or above
- pip (Python package manager)
- Git
- Visual Studio Code (recommended)

### Step 1: Clone the Repository

```bash
git clone https://github.com/awadhanisiddhi/Document-Summarization-System.git
```

### Step 2: Navigate to the Project Directory

```bash
cd Document-Summarization-System
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

### Step 4: Activate the Virtual Environment

**Windows PowerShell:**

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Run the Application

```bash
streamlit run app.py
```

The application will open in your default browser. Alternatively, visit:

```text
http://localhost:8501
```

## Screenshots

Add screenshots of the application interface and its output in this section.

Suggested screenshots:

1. SmartDoc home page and document upload interface.
2. Document analytics dashboard.
3. Generated summary using TF-IDF.
4. Generated summary using TextRank.
5. Keyword extraction and download options.

You can store screenshots in a folder named `screenshots/` and reference them like this:

```markdown
![SmartDoc Dashboard](screenshots/dashboard.png)
```

## Limitations

- The application performs extractive rather than abstractive summarization.
- PDF extraction works with text-based PDFs; scanned documents requiring OCR are not currently supported.
- Summary quality depends on the document content and the selected algorithm.
- The application currently focuses on English-language text processing.

## Future Scope

Potential enhancements include:

- Side-by-side comparison of TF-IDF and TextRank summaries.
- Highlighting selected sentences in the original document.
- Advanced document analytics and keyword-frequency visualization.
- Summary history and session management.
- Improved user interface and application deployment.
- Support for additional languages and OCR-based document extraction.

## Learning Outcomes

This project demonstrates practical implementation of:

- Natural Language Processing concepts.
- Text preprocessing and sentence tokenization.
- TF-IDF feature extraction.
- Graph-based TextRank summarization.
- Cosine similarity and sentence ranking.
- Document processing and file handling.
- Interactive application development using Streamlit.

## Author

**Siddhi Awadhani**

Computer Engineering Student

GitHub: [@awadhanisiddhi](https://github.com/awadhanisiddhi)

---

**SmartDoc – Making lengthy documents easier to understand.**