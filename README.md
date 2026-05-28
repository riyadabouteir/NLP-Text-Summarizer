# 📝 NLP Text Summarization Project

A complete NLP text summarization system with four algorithms and an interactive Streamlit UI.

## Project Structure

```
nlp_summarizer/
├── summarizer.py           # Core NLP module (all algorithms)
├── app.py                  # Streamlit web interface
├── NLP_Summarization.ipynb # Jupyter notebook walkthrough
├── requirements.txt        # Python dependencies
└── README.md
```

## Algorithms Implemented

| Method | Type | Description |
|--------|------|-------------|
| **TF-IDF** | Extractive | Scores sentences by Term Frequency × Inverse Document Frequency |
| **TextRank** | Extractive | Graph-based PageRank algorithm on sentence similarity |
| **Frequency** | Extractive | Normalized word-frequency sentence scoring |
| **BART (distilbart-cnn)** | Abstractive | Pre-trained transformer, generates new text |

## Evaluation

- **ROUGE-1** — Unigram overlap
- **ROUGE-2** — Bigram overlap  
- **ROUGE-L** — Longest Common Subsequence

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit app

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

### 3. Open the Jupyter notebook

```bash
jupyter notebook NLP_Summarization.ipynb
```

### 4. Use the Python API directly

```python
from summarizer import summarize

text = "Your long text here..."

# Extractive (no downloads needed)
result = summarize(text, method="textrank", num_sentences=3)
print(result["summary"])

# With ROUGE evaluation
result = summarize(text, method="tfidf", num_sentences=3, reference="Your reference summary")
print(result["rouge"])

# Abstractive (downloads ~500MB model on first use)
result = summarize(text, method="abstractive", max_length=150)
print(result["summary"])
```

## Notes

- Extractive methods (TF-IDF, TextRank, Frequency) work **completely offline** with no downloads
- Abstractive (BART) downloads ~500MB on first use, then caches locally
- All methods are implemented from scratch — no NLTK data files required
