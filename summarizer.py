"""
NLP Text Summarization Module
Supports: Extractive (TF-IDF, TextRank, Frequency, LSA, LexRank, Position-Biased)
          Abstractive (T5 — t5-small / t5-base / t5-large)

Advanced T5 features:
  • Multi-passage chunking for long documents
  • Beam search + length penalty controls
  • Automatic language detection (heuristic)
  • Repetition penalty & no-repeat n-gram
  • Bullet-point mode (structured output)
  • Headline mode (ultra-short)
"""

import re
import math
import heapq
from collections import Counter, defaultdict
from typing import Literal


# ─────────────────────────────────────────────
# Utility: Simple sentence & word tokenizers
# ─────────────────────────────────────────────

STOPWORDS = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "by","from","is","it","its","this","that","these","those","was","were",
    "be","been","being","have","has","had","do","does","did","will","would",
    "could","should","may","might","shall","can","not","no","nor","so","yet",
    "both","either","neither","as","if","then","than","too","very","just",
    "also","more","most","other","some","such","only","own","same","so","each",
    "he","she","they","we","you","i","me","him","her","us","them","who","what",
    "which","when","where","why","how","all","any","few","much","many","our",
    "your","their","my","his","into","over","after","before","about","through",
    "during","while","since","until","there","here","now","up","down","out",
}


def sent_tokenize(text: str) -> list[str]:
    text = text.strip()
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def word_tokenize(text: str) -> list[str]:
    return re.findall(r'\b[a-z]+\b', text.lower())


def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', ' ', text)
    return text.strip()


def detect_language_hint(text: str) -> str:
    """Heuristic language detection (French vs English) for T5 prefix selection."""
    french_markers = {"le","la","les","de","du","des","et","un","une","en","est","que","qui","dans","pour","sur","avec","il","elle","nous","vous","ils","elles","je","tu","pas","plus","au","aux","ce","se","son","sa","ses","leur","leurs","par","mais","ou","donc","or","ni","car"}
    words = set(word_tokenize(text[:500]))
    fr_hits = len(words & french_markers)
    return "fr" if fr_hits >= 5 else "en"


# ─────────────────────────────────────────────
# Method 1: TF-IDF Extractive
# ─────────────────────────────────────────────

def tfidf_summarize(text: str, num_sentences: int = 3) -> dict:
    text = clean_text(text)
    sentences = sent_tokenize(text)
    if len(sentences) <= num_sentences:
        return {"summary": text, "sentences_used": len(sentences), "method": "TF-IDF", "scores": {}}

    tf_matrix: dict[int, dict[str, float]] = {}
    for i, sent in enumerate(sentences):
        words = [w for w in word_tokenize(sent) if w not in STOPWORDS]
        total = len(words) or 1
        freq = Counter(words)
        tf_matrix[i] = {w: c / total for w, c in freq.items()}

    num_sents = len(sentences)
    df: dict[str, int] = defaultdict(int)
    for tf in tf_matrix.values():
        for w in tf:
            df[w] += 1
    idf = {w: math.log(num_sents / (1 + count)) for w, count in df.items()}

    scores: dict[int, float] = {}
    for i, tf in tf_matrix.items():
        scores[i] = sum(tf_val * idf.get(w, 0) for w, tf_val in tf.items())

    top_indices = sorted(heapq.nlargest(num_sentences, scores, key=scores.get))
    summary = " ".join(sentences[i] for i in top_indices)
    return {
        "summary": summary,
        "sentences_used": len(top_indices),
        "total_sentences": len(sentences),
        "method": "TF-IDF Extractive",
        "scores": {sentences[i]: round(scores[i], 4) for i in range(len(sentences))},
    }


# ─────────────────────────────────────────────
# Method 2: TextRank Extractive
# ─────────────────────────────────────────────

def _sentence_similarity(s1: str, s2: str) -> float:
    words1 = set(word_tokenize(s1)) - STOPWORDS
    words2 = set(word_tokenize(s2)) - STOPWORDS
    if not words1 or not words2:
        return 0.0
    return len(words1 & words2) / (math.log(len(words1) + 1) + math.log(len(words2) + 1))


def textrank_summarize(text: str, num_sentences: int = 3, damping: float = 0.85, iterations: int = 30) -> dict:
    text = clean_text(text)
    sentences = sent_tokenize(text)
    n = len(sentences)
    if n <= num_sentences:
        return {"summary": text, "sentences_used": n, "method": "TextRank", "scores": {}}

    sim_matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                sim_matrix[i][j] = _sentence_similarity(sentences[i], sentences[j])
    for i in range(n):
        row_sum = sum(sim_matrix[i]) or 1
        sim_matrix[i] = [v / row_sum for v in sim_matrix[i]]

    scores = [1.0 / n] * n
    for _ in range(iterations):
        scores = [
            (1 - damping) / n + damping * sum(sim_matrix[j][i] * scores[j] for j in range(n))
            for i in range(n)
        ]

    top_indices = sorted(heapq.nlargest(num_sentences, range(n), key=lambda i: scores[i]))
    summary = " ".join(sentences[i] for i in top_indices)
    return {
        "summary": summary,
        "sentences_used": len(top_indices),
        "total_sentences": n,
        "method": "TextRank Extractive",
        "scores": {sentences[i]: round(scores[i], 4) for i in range(n)},
    }


# ─────────────────────────────────────────────
# Method 3: Frequency-Based Extractive
# ─────────────────────────────────────────────

def frequency_summarize(text: str, num_sentences: int = 3) -> dict:
    text = clean_text(text)
    sentences = sent_tokenize(text)
    if len(sentences) <= num_sentences:
        return {"summary": text, "sentences_used": len(sentences), "method": "Frequency", "scores": {}}

    words = [w for w in word_tokenize(text) if w not in STOPWORDS]
    freq = Counter(words)
    max_freq = max(freq.values()) if freq else 1
    freq_norm = {w: c / max_freq for w, c in freq.items()}

    scores = {i: sum(freq_norm.get(w, 0) for w in word_tokenize(sent) if w not in STOPWORDS)
              for i, sent in enumerate(sentences)}
    top_indices = sorted(heapq.nlargest(num_sentences, scores, key=scores.get))
    summary = " ".join(sentences[i] for i in top_indices)
    return {
        "summary": summary,
        "sentences_used": len(top_indices),
        "total_sentences": len(sentences),
        "method": "Frequency-Based Extractive",
        "scores": {sentences[i]: round(scores[i], 4) for i in range(len(sentences))},
    }


# ─────────────────────────────────────────────
# Method 4: LSA (Latent Semantic Analysis)
# ─────────────────────────────────────────────

def _build_term_sentence_matrix(sentences: list[str]):
    vocab: dict[str, int] = {}
    tokenized = []
    for sent in sentences:
        words = [w for w in word_tokenize(sent) if w not in STOPWORDS]
        tokenized.append(words)
        for w in words:
            if w not in vocab:
                vocab[w] = len(vocab)
    matrix = [[0.0] * len(sentences) for _ in range(len(vocab))]
    for j, words in enumerate(tokenized):
        total = len(words) or 1
        freq = Counter(words)
        for w, c in freq.items():
            matrix[vocab[w]][j] = c / total
    return matrix, vocab


def _svd_power(matrix, k: int = 3, iterations: int = 20):
    rows = len(matrix)
    cols = len(matrix[0]) if rows else 0
    if rows == 0 or cols == 0:
        return [0.0] * cols
    ATA = [[sum(matrix[r][i] * matrix[r][j] for r in range(rows)) for j in range(cols)] for i in range(cols)]
    scores = [0.0] * cols
    for _ in range(min(k, cols)):
        vec = [ATA[i][i] + 1e-9 for i in range(cols)]
        for _ in range(iterations):
            new_vec = [sum(ATA[i][j] * vec[j] for j in range(cols)) for i in range(cols)]
            norm = math.sqrt(sum(v * v for v in new_vec)) or 1.0
            vec = [v / norm for v in new_vec]
        sigma = math.sqrt(abs(sum(ATA[i][j] * vec[i] * vec[j] for i in range(cols) for j in range(cols))))
        for i in range(cols):
            scores[i] += sigma * abs(vec[i])
        for i in range(cols):
            for j in range(cols):
                ATA[i][j] -= sigma * vec[i] * vec[j]
    return scores


def lsa_summarize(text: str, num_sentences: int = 3, num_concepts: int = 3) -> dict:
    text = clean_text(text)
    sentences = sent_tokenize(text)
    n = len(sentences)
    if n <= num_sentences:
        return {"summary": text, "sentences_used": n, "method": "LSA", "scores": {}}
    matrix, _ = _build_term_sentence_matrix(sentences)
    if not matrix:
        return frequency_summarize(text, num_sentences)
    scores_raw = _svd_power(matrix, k=min(num_concepts, n - 1))
    scores = {i: scores_raw[i] if i < len(scores_raw) else 0.0 for i in range(n)}
    top_indices = sorted(heapq.nlargest(num_sentences, scores, key=scores.get))
    summary = " ".join(sentences[i] for i in top_indices)
    return {
        "summary": summary,
        "sentences_used": len(top_indices),
        "total_sentences": n,
        "method": "LSA Extractive",
        "scores": {sentences[i]: round(scores[i], 4) for i in range(n)},
    }


# ─────────────────────────────────────────────
# Method 5: LexRank Extractive
# ─────────────────────────────────────────────

def _cosine_similarity(s1: str, s2: str) -> float:
    words1 = [w for w in word_tokenize(s1) if w not in STOPWORDS]
    words2 = [w for w in word_tokenize(s2) if w not in STOPWORDS]
    if not words1 or not words2:
        return 0.0
    freq1 = Counter(words1)
    freq2 = Counter(words2)
    vocab = set(freq1) | set(freq2)
    dot = sum(freq1.get(w, 0) * freq2.get(w, 0) for w in vocab)
    norm1 = math.sqrt(sum(v * v for v in freq1.values()))
    norm2 = math.sqrt(sum(v * v for v in freq2.values()))
    return dot / (norm1 * norm2) if norm1 and norm2 else 0.0


def lexrank_summarize(text: str, num_sentences: int = 3, threshold: float = 0.1,
                      damping: float = 0.85, iterations: int = 30) -> dict:
    text = clean_text(text)
    sentences = sent_tokenize(text)
    n = len(sentences)
    if n <= num_sentences:
        return {"summary": text, "sentences_used": n, "method": "LexRank", "scores": {}}

    sim_matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                sim = _cosine_similarity(sentences[i], sentences[j])
                sim_matrix[i][j] = sim if sim >= threshold else 0.0
    for i in range(n):
        row_sum = sum(sim_matrix[i]) or 1.0
        sim_matrix[i] = [v / row_sum for v in sim_matrix[i]]

    scores = [1.0 / n] * n
    for _ in range(iterations):
        scores = [
            (1 - damping) / n + damping * sum(sim_matrix[j][i] * scores[j] for j in range(n))
            for i in range(n)
        ]

    top_indices = sorted(heapq.nlargest(num_sentences, range(n), key=lambda i: scores[i]))
    summary = " ".join(sentences[i] for i in top_indices)
    return {
        "summary": summary,
        "sentences_used": len(top_indices),
        "total_sentences": n,
        "method": "LexRank Extractive",
        "scores": {sentences[i]: round(scores[i], 4) for i in range(n)},
    }


# ─────────────────────────────────────────────
# Method 6: Position-Biased
# ─────────────────────────────────────────────

def position_summarize(text: str, num_sentences: int = 3, lead_bias: float = 1.5) -> dict:
    text = clean_text(text)
    sentences = sent_tokenize(text)
    n = len(sentences)
    if n <= num_sentences:
        return {"summary": text, "sentences_used": n, "method": "Position-Biased", "scores": {}}

    words = [w for w in word_tokenize(text) if w not in STOPWORDS]
    freq = Counter(words)
    max_freq = max(freq.values()) if freq else 1
    freq_norm = {w: c / max_freq for w, c in freq.items()}

    lead_zone = max(1, int(n * 0.20))
    tail_zone = max(1, int(n * 0.15))
    scores = {}
    for i, sent in enumerate(sentences):
        base = sum(freq_norm.get(w, 0) for w in word_tokenize(sent) if w not in STOPWORDS)
        if i < lead_zone or i >= n - tail_zone:
            multiplier = lead_bias
        elif i == lead_zone or i == n - tail_zone - 1:
            multiplier = 1.0 + (lead_bias - 1.0) * 0.5
        else:
            multiplier = 1.0
        scores[i] = base * multiplier

    top_indices = sorted(heapq.nlargest(num_sentences, scores, key=scores.get))
    summary = " ".join(sentences[i] for i in top_indices)
    return {
        "summary": summary,
        "sentences_used": len(top_indices),
        "total_sentences": n,
        "method": "Position-Biased Extractive",
        "scores": {sentences[i]: round(scores[i], 4) for i in range(n)},
    }


# ─────────────────────────────────────────────
# Method 7: T5 Abstractive — Advanced
# ─────────────────────────────────────────────

T5_MODELS = {
    "t5-small":  {"label": "T5-Small (~242MB) — Rapide",       "max_input": 512},
    "t5-base":   {"label": "T5-Base (~892MB) — Équilibré",      "max_input": 512},
    "t5-large":  {"label": "T5-Large (~2.9GB) — Haute qualité", "max_input": 512},
}

_model_cache: dict[str, tuple] = {}   # model_name -> (model, tokenizer)


def load_t5_model(model_name: str = "t5-small"):
    """
    Lazy-load T5ForConditionalGeneration + AutoTokenizer.
    Avoids the 'text2text-generation' pipeline task that was removed
    from newer versions of transformers (≥ 4.52).
    Returns (model, tokenizer).
    """
    if model_name not in _model_cache:
        from transformers import T5ForConditionalGeneration, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model     = T5ForConditionalGeneration.from_pretrained(model_name)
        model.eval()
        _model_cache[model_name] = (model, tokenizer)
    return _model_cache[model_name]


def _t5_generate(model, tokenizer, prompt: str, max_length: int, min_length: int,
                 num_beams: int, length_penalty: float, repetition_penalty: float,
                 no_repeat_ngram_size: int) -> str:
    """Run T5 generation; returns decoded string."""
    import torch
    inputs = tokenizer(prompt, return_tensors="pt",
                       max_length=512, truncation=True, padding=False)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            early_stopping=True,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def _chunk_text(text: str, max_words: int = 450, overlap: int = 50) -> list[str]:
    """
    Split long text into overlapping word-level chunks.
    Overlap ensures context continuity between chunks.
    """
    words = text.split()
    if len(words) <= max_words:
        return [text]
    chunks = []
    step = max_words - overlap
    for start in range(0, len(words), step):
        chunk = " ".join(words[start: start + max_words])
        chunks.append(chunk)
        if start + max_words >= len(words):
            break
    return chunks


def _merge_summaries(summaries: list[str]) -> str:
    """Merge multiple chunk summaries, removing obvious duplicates."""
    seen_sents = set()
    merged = []
    for s in summaries:
        for sent in sent_tokenize(s):
            key = frozenset(word_tokenize(sent))
            if key not in seen_sents and len(key) > 3:
                seen_sents.add(key)
                merged.append(sent)
    return " ".join(merged)


def abstractive_summarize(
    text: str,
    max_length: int = 150,
    min_length: int = 40,
    model_name: str = "t5-small",
    num_beams: int = 4,
    length_penalty: float = 1.0,
    repetition_penalty: float = 1.2,
    no_repeat_ngram_size: int = 3,
    output_mode: str = "paragraph",   # "paragraph" | "bullets" | "headline"
    chunk_strategy: str = "auto",     # "auto" | "single" | "multi"
) -> dict:
    """
    Advanced abstractive summarization using T5.

    Parameters
    ----------
    output_mode : "paragraph" | "bullets" | "headline"
        paragraph  — standard flowing summary
        bullets    — structured bullet-point summary (T5 generates; we split by ';')
        headline   — ultra-short, title-like summary
    chunk_strategy : "auto" | "single" | "multi"
        auto   — uses multi-chunk for texts > 450 words
        single — always truncate to first 450 words
        multi  — always chunk (useful for long documents)
    """
    text = clean_text(text)
    model, tokenizer = load_t5_model(model_name)

    # Adjust prompt & length params per mode
    if output_mode == "headline":
        prefix = "summarize in one sentence: "
        max_length = min(max_length, 40)
        min_length = min(min_length, 10)
        num_beams = max(num_beams, 4)
    elif output_mode == "bullets":
        prefix = "summarize as key points separated by semicolons: "
    else:
        prefix = "summarize: "

    word_count = len(text.split())
    use_multi = (chunk_strategy == "multi") or (chunk_strategy == "auto" and word_count > 450)

    gkw = dict(
        max_length=max_length, min_length=min_length,
        num_beams=num_beams, length_penalty=length_penalty,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
    )

    if use_multi:
        chunks = _chunk_text(text, max_words=450, overlap=50)
        chunk_summaries = [
            _t5_generate(model, tokenizer, prefix + chunk, **gkw)
            for chunk in chunks
        ]
        if len(chunks) > 1:
            merged_input = _merge_summaries(chunk_summaries)
            if len(merged_input.split()) > 80:
                summary = _t5_generate(
                    model, tokenizer, "summarize: " + merged_input,
                    max_length=max_length, min_length=min(min_length, 30),
                    num_beams=num_beams, length_penalty=length_penalty,
                    repetition_penalty=repetition_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                )
            else:
                summary = merged_input
        else:
            summary = chunk_summaries[0]
        chunks_used = len(chunks)
    else:
        truncated = " ".join(text.split()[:450])
        summary = _t5_generate(model, tokenizer, prefix + truncated, **gkw)
        chunks_used = 1

    # Post-process bullet mode: split on ';' and format
    if output_mode == "bullets":
        points = [p.strip().rstrip(".") for p in summary.split(";") if p.strip()]
        if len(points) <= 1:
            # Fallback: split on '. '
            points = [p.strip() for p in re.split(r'\.\s+', summary) if p.strip()]
        summary = "\n• " + "\n• ".join(points) if points else summary

    return {
        "summary": summary,
        "method": f"T5 Abstractive ({model_name})",
        "model": model_name,
        "output_mode": output_mode,
        "chunks_used": chunks_used,
        "max_length": max_length,
        "min_length": min_length,
        "num_beams": num_beams,
        "length_penalty": length_penalty,
        "repetition_penalty": repetition_penalty,
        "no_repeat_ngram_size": no_repeat_ngram_size,
    }


# ─────────────────────────────────────────────
# ROUGE Score Evaluation
# ─────────────────────────────────────────────

def compute_rouge(hypothesis: str, reference: str) -> dict:
    def ngrams(tokens, n):
        return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

    def lcs_length(a, b):
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                dp[i][j] = dp[i-1][j-1] + 1 if a[i-1] == b[j-1] else max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]

    hyp = word_tokenize(hypothesis)
    ref = word_tokenize(reference)

    def rouge_n(n):
        hyp_ng = ngrams(hyp, n)
        ref_ng = ngrams(ref, n)
        overlap = sum((hyp_ng & ref_ng).values())
        precision = overlap / max(sum(hyp_ng.values()), 1)
        recall = overlap / max(sum(ref_ng.values()), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}

    lcs = lcs_length(hyp, ref)
    rouge_l_p = lcs / max(len(hyp), 1)
    rouge_l_r = lcs / max(len(ref), 1)
    rouge_l_f = 2 * rouge_l_p * rouge_l_r / max(rouge_l_p + rouge_l_r, 1e-9)

    return {
        "rouge1": rouge_n(1),
        "rouge2": rouge_n(2),
        "rougeL": {"precision": round(rouge_l_p, 4), "recall": round(rouge_l_r, 4), "f1": round(rouge_l_f, 4)},
    }


# ─────────────────────────────────────────────
# Unified API
# ─────────────────────────────────────────────

Method = Literal["tfidf", "textrank", "frequency", "lsa", "lexrank", "position", "abstractive"]


def summarize(
    text: str,
    method: Method = "textrank",
    num_sentences: int = 3,
    max_length: int = 150,
    min_length: int = 40,
    reference: str = "",
    lead_bias: float = 1.5,
    num_concepts: int = 3,
    lexrank_threshold: float = 0.1,
    # T5 advanced params
    t5_model: str = "t5-small",
    t5_num_beams: int = 4,
    t5_length_penalty: float = 1.0,
    t5_repetition_penalty: float = 1.2,
    t5_no_repeat_ngram: int = 3,
    t5_output_mode: str = "paragraph",
    t5_chunk_strategy: str = "auto",
) -> dict:
    """Unified summarization interface."""
    text = text.strip()
    if not text:
        return {"error": "Empty input text.", "summary": ""}

    word_count_in = len(text.split())
    char_count_in = len(text)

    if method == "tfidf":
        result = tfidf_summarize(text, num_sentences)
    elif method == "textrank":
        result = textrank_summarize(text, num_sentences)
    elif method == "frequency":
        result = frequency_summarize(text, num_sentences)
    elif method == "lsa":
        result = lsa_summarize(text, num_sentences, num_concepts)
    elif method == "lexrank":
        result = lexrank_summarize(text, num_sentences, threshold=lexrank_threshold)
    elif method == "position":
        result = position_summarize(text, num_sentences, lead_bias)
    elif method == "abstractive":
        result = abstractive_summarize(
            text,
            max_length=max_length,
            min_length=min_length,
            model_name=t5_model,
            num_beams=t5_num_beams,
            length_penalty=t5_length_penalty,
            repetition_penalty=t5_repetition_penalty,
            no_repeat_ngram_size=t5_no_repeat_ngram,
            output_mode=t5_output_mode,
            chunk_strategy=t5_chunk_strategy,
        )
    else:
        return {"error": f"Unknown method '{method}'.", "summary": ""}

    summary = result["summary"]
    result["stats"] = {
        "input_words": word_count_in,
        "input_chars": char_count_in,
        "output_words": len(summary.split()),
        "output_chars": len(summary),
        "compression_ratio": round(len(summary) / max(char_count_in, 1), 3),
    }

    if reference.strip():
        result["rouge"] = compute_rouge(summary, reference)

    return result