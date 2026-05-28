"""
NLP Text Summarization — Streamlit Interface (v2)
Run with: streamlit run app.py
"""

import streamlit as st
import time
import sys
import os
import json

# ── Robust path fix: works locally AND on Streamlit Cloud ──────────────────
_here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
if _here not in sys.path:
    sys.path.insert(0, _here)

# ── Graceful import with user-facing error instead of blank screen ──────────
try:
    from summarizer import summarize, sent_tokenize, T5_MODELS
    _IMPORT_OK = True
    _IMPORT_ERR = ""
except Exception as _e:
    _IMPORT_OK = False
    _IMPORT_ERR = str(_e)
    # Stubs so the rest of the file doesn't crash at definition time
    T5_MODELS = {"t5-small": {"label": "T5-Small"}}
    def summarize(*a, **kw): return {"error": "Import failed", "summary": ""}
    def sent_tokenize(t): return t.split(".")

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NLP Summarizer",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Manrope:wght@300;400;600;700;800&display=swap');

  html, body, [class*="css"] { font-family: 'Manrope', sans-serif; }

  .stApp { background: #0d0f14; color: #e8e6e0; }

  .main-header {
    font-family: 'Space Mono', monospace;
    font-size: 2.6rem; font-weight: 700;
    color: #f5f0e8; letter-spacing: -0.02em; margin-bottom: 0.2rem;
  }
  .sub-header {
    font-family: 'Manrope', sans-serif;
    font-size: 0.95rem; color: #7a7870;
    letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 2rem;
  }

  .card {
    background: #161820; border: 1px solid #252830;
    border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem;
  }

  .method-badge {
    display: inline-block; background: #1e2530;
    border: 1px solid #3a9e7e; color: #3a9e7e;
    font-family: 'Space Mono', monospace; font-size: 0.72rem;
    padding: 3px 10px; border-radius: 20px;
    letter-spacing: 0.06em; margin-bottom: 0.8rem;
  }
  .t5-badge {
    display: inline-block; background: #1e1530;
    border: 1px solid #9e7e3a; color: #c9a84c;
    font-family: 'Space Mono', monospace; font-size: 0.72rem;
    padding: 3px 10px; border-radius: 20px;
    letter-spacing: 0.06em; margin-bottom: 0.8rem;
  }

  .summary-box {
    background: #12141a; border-left: 3px solid #3a9e7e;
    padding: 1.2rem 1.4rem; border-radius: 0 8px 8px 0;
    font-size: 1.02rem; line-height: 1.75; color: #ddd9d0;
  }
  .summary-box-t5 {
    background: #12141a; border-left: 3px solid #c9a84c;
    padding: 1.2rem 1.4rem; border-radius: 0 8px 8px 0;
    font-size: 1.02rem; line-height: 1.9; color: #ddd9d0;
    white-space: pre-line;
  }
  .summary-headline {
    background: #12141a; border-left: 3px solid #9e3a7e;
    padding: 1rem 1.4rem; border-radius: 0 8px 8px 0;
    font-size: 1.3rem; font-weight: 700; line-height: 1.5; color: #f0ebff;
    font-family: 'Space Mono', monospace;
  }

  .stat-grid {
    display: grid; grid-template-columns: repeat(4, 1fr);
    gap: 0.8rem; margin-top: 1rem;
  }
  .stat-box {
    background: #1a1c24; border: 1px solid #252830;
    border-radius: 8px; padding: 0.8rem; text-align: center;
  }
  .stat-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.5rem; font-weight: 700; color: #3a9e7e; display: block;
  }
  .stat-value-gold { color: #c9a84c !important; }
  .stat-label {
    font-size: 0.72rem; color: #5c5a54;
    text-transform: uppercase; letter-spacing: 0.06em;
  }

  .rouge-table {
    width: 100%; border-collapse: collapse;
    font-family: 'Space Mono', monospace; font-size: 0.82rem; margin-top: 0.8rem;
  }
  .rouge-table th {
    background: #1a1c24; color: #7a7870; padding: 8px 12px;
    text-align: left; font-weight: 400; text-transform: uppercase;
    letter-spacing: 0.05em; font-size: 0.72rem;
  }
  .rouge-table td { padding: 8px 12px; border-top: 1px solid #252830; color: #c8c4bc; }
  .rouge-table tr:hover td { background: #1e2028; }
  .highlight-score { color: #3a9e7e; font-weight: 700; }
  .highlight-gold   { color: #c9a84c; font-weight: 700; }

  [data-testid="stSidebar"] {
    background: #101218; border-right: 1px solid #1e2028;
  }

  .section-label {
    font-family: 'Space Mono', monospace; font-size: 0.72rem; color: #5c5a54;
    text-transform: uppercase; letter-spacing: 0.1em;
    margin-bottom: 0.5rem; display: block;
  }

  .info-box {
    background: #141820; border: 1px solid #252830;
    border-radius: 8px; padding: 1rem;
    font-size: 0.88rem; color: #8a8880; line-height: 1.6;
  }
  .info-box strong { color: #b8b4ac; }

  .t5-param-grid {
    display: grid; grid-template-columns: repeat(3, 1fr);
    gap: 0.5rem; margin: 0.8rem 0;
  }
  .t5-param-box {
    background: #1a1c24; border: 1px solid #2a2830;
    border-radius: 6px; padding: 0.5rem 0.8rem; text-align: center;
  }
  .t5-param-val {
    font-family: 'Space Mono', monospace;
    font-size: 1.1rem; font-weight: 700; color: #c9a84c;
  }
  .t5-param-lbl {
    font-size: 0.65rem; color: #5c5a54;
    text-transform: uppercase; letter-spacing: 0.06em;
  }

  .history-item {
    background: #161820; border: 1px solid #252830;
    border-radius: 8px; padding: 0.8rem 1rem; margin-bottom: 0.5rem;
    cursor: pointer; transition: border-color 0.15s;
  }
  .history-item:hover { border-color: #3a9e7e; }
  .history-method {
    font-family: 'Space Mono', monospace; font-size: 0.68rem;
    color: #3a9e7e; text-transform: uppercase; letter-spacing: 0.08em;
  }
  .history-preview {
    font-size: 0.85rem; color: #8a8880; margin-top: 0.3rem;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }

  .chunk-badge {
    display: inline-flex; align-items: center; gap: 0.3rem;
    background: #1e2030; border: 1px solid #3a7e9e;
    color: #5abdc8; font-family: 'Space Mono', monospace;
    font-size: 0.68rem; padding: 2px 8px;
    border-radius: 12px; margin-left: 0.5rem;
  }

  .bar-track { background: #1a1c24; border-radius: 4px; height: 8px; margin-top: 6px; }
  .bar-fill  { height: 8px; border-radius: 4px; background: linear-gradient(90deg,#3a9e7e,#5abda0); }
  .bar-fill-gold { background: linear-gradient(90deg,#9e7e3a,#c9a84c) !important; }

  .stTabs [data-baseweb="tab-list"] {
    background: #101218; border-bottom: 1px solid #252830; gap: 2px;
  }
  .stTabs [data-baseweb="tab"] {
    font-family: 'Space Mono', monospace; font-size: 0.78rem;
    color: #5c5a54 !important; background: transparent;
    border: none; padding: 10px 18px;
  }
  .stTabs [aria-selected="true"] {
    color: #3a9e7e !important; border-bottom: 2px solid #3a9e7e !important;
    background: transparent !important;
  }

  .stTextArea textarea {
    background: #161820 !important; border: 1px solid #252830 !important;
    color: #e8e6e0 !important; font-family: 'Manrope', sans-serif !important;
    border-radius: 8px !important; font-size: 0.95rem !important;
  }
  .stButton button {
    background: #3a9e7e !important; color: #0d0f14 !important;
    font-family: 'Space Mono', monospace !important; font-weight: 700 !important;
    font-size: 0.85rem !important; border: none !important;
    border-radius: 8px !important; padding: 0.6rem 1.6rem !important;
    letter-spacing: 0.04em !important; transition: all 0.15s !important;
  }
  .stButton button:hover { background: #5abda0 !important; transform: translateY(-1px) !important; }
  .stSelectbox > div > div {
    background: #161820 !important; border: 1px solid #252830 !important;
    color: #e8e6e0 !important; border-radius: 8px !important;
  }
  .stSlider > div > div > div { background: #3a9e7e !important; }
  .stMetric { background: #161820; border-radius: 8px; padding: 0.8rem; }
  div[data-testid="stExpander"] {
    background: #161820; border: 1px solid #252830; border-radius: 8px;
  }
</style>
""", unsafe_allow_html=True)


# ─── Import-error banner ────────────────────────────────────────────────────
if not _IMPORT_OK:
    st.error(
        f"**Erreur d'import — le module summarizer.py n'a pas pu être chargé.**\n\n"
        f"`{_IMPORT_ERR}`\n\n"
        "Vérifiez que `summarizer.py` est dans le même dossier que `app.py` "
        "et que toutes les dépendances de `requirements.txt` sont installées."
    )
    st.stop()

# ─── Session state ───────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []   # list of result dicts


def add_to_history(result: dict, input_text: str):
    entry = {
        "method": result.get("method", "?"),
        "summary": result.get("summary", ""),
        "input_preview": input_text[:120],
        "stats": result.get("stats", {}),
        "rouge": result.get("rouge"),
        "timestamp": time.strftime("%H:%M:%S"),
    }
    st.session_state.history.insert(0, entry)
    if len(st.session_state.history) > 10:
        st.session_state.history.pop()


# ─── Sample texts ────────────────────────────────────────────────────────────
SAMPLES = {
    "IA & Machine Learning": """Artificial intelligence (AI) is transforming industries worldwide, from healthcare and finance to transportation and education. Machine learning, a subset of AI, enables computers to learn from data without being explicitly programmed. Deep learning, built on neural networks with many layers, has achieved remarkable breakthroughs in image recognition, natural language processing, and game playing. Natural language processing (NLP) allows computers to understand, interpret, and generate human language in a way that is both meaningful and useful. Applications like virtual assistants, translation services, and sentiment analysis are now commonplace thanks to NLP advances. Researchers continue to push boundaries, developing models like GPT and BERT that demonstrate near-human text comprehension and generation capabilities. However, challenges remain around bias, explainability, energy consumption, and ethical deployment of these powerful systems. Governments and organizations are working to establish frameworks and regulations to ensure AI is developed and used responsibly.""",

    "Changement Climatique": """Climate change refers to long-term shifts in temperatures and weather patterns, primarily caused by human activities since the 1800s, especially the burning of fossil fuels. Burning fossil fuels like coal, oil, and gas produces greenhouse gas emissions that act like a blanket wrapped around the Earth, trapping the sun's heat and raising temperatures. The consequences of climate change include intense droughts, water scarcity, severe fires, rising sea levels, flooding, melting polar ice, catastrophic storms, and declining biodiversity. People are experiencing climate change in diverse ways. Climate change affects our health, ability to grow food, housing, safety, and work. Some of us are already more vulnerable to climate impacts such as people living in small island nations and other developing countries. Conditions like sea-level rise and saltwater intrusion have advanced to the point where whole communities have had to relocate. Many more people face risks of falling into poverty. The Paris Agreement on climate change is a global pact between nations to limit global warming by reducing greenhouse gas emissions. Net zero means cutting greenhouse gas emissions to as close to zero as possible, with any remaining emissions re-absorbed from the atmosphere by oceans and forests.""",

    "Informatique Quantique": """Quantum computing represents a fundamentally different approach to computation that harnesses the principles of quantum mechanics to process information. Unlike classical computers that use bits — binary digits that can be either 0 or 1 — quantum computers use quantum bits or qubits. Qubits can exist in a superposition of states, meaning they can be 0 and 1 simultaneously until measured. This property, along with quantum entanglement and quantum interference, gives quantum computers the ability to explore many possible solutions to a problem simultaneously. Quantum entanglement allows qubits to be correlated in such a way that the state of one qubit instantly influences the state of another, regardless of distance. Quantum computers excel at specific tasks such as factoring large numbers, simulating molecular interactions for drug discovery, optimizing complex systems, and breaking certain types of encryption. Companies like IBM, Google, and startups such as IonQ are racing to build practical quantum computers. Google claimed quantum supremacy in 2019 when their Sycamore processor solved a specific problem in 200 seconds that would take classical supercomputers thousands of years. Despite these advances, quantum computers currently suffer from high error rates and require extreme cooling to near absolute zero temperatures.""",

    "Long Document (Test multi-chunk)": """The history of the internet is a fascinating journey from a small academic network to the global infrastructure that underpins modern society. In the late 1960s, the United States Department of Defense funded ARPANET, a project designed to create a resilient communications network that could survive a nuclear strike. The original network connected just four universities: UCLA, Stanford Research Institute, UC Santa Barbara, and the University of Utah. In 1969, the first message was sent over ARPANET — famously, the system crashed after just two letters, "LO," before the intended word "LOGIN" was completed. Throughout the 1970s, the network grew steadily, and protocols like FTP and email emerged. The 1980s saw the introduction of TCP/IP as the standard communication protocol, which allowed different networks to interconnect — creating the foundation for what we now call the internet. The domain name system (DNS) was introduced in 1983, replacing hard-to-remember numerical addresses with human-readable names. Tim Berners-Lee invented the World Wide Web in 1989 while working at CERN, proposing a system of hyperlinked documents accessible over the internet. The first web browser, Mosaic, launched in 1993 and made the web accessible to the general public. The mid-1990s saw an explosion of commercial activity online, with companies like Amazon, Yahoo, and eBay launching services. The dot-com bubble of the late 1990s saw massive investment in internet companies, followed by a dramatic crash in 2000 when many overvalued companies collapsed. However, survivors like Google — founded in 1998 — went on to reshape the digital landscape. Broadband internet replaced dial-up connections in the early 2000s, dramatically increasing speeds and enabling new services like YouTube, launched in 2005. The rise of smartphones in the late 2000s, led by Apple's iPhone in 2007, transformed internet access from a desktop activity into an always-on mobile experience. Social media platforms like Facebook, Twitter, and Instagram changed how people communicate and share information. Today, the internet connects over five billion people worldwide and is the backbone of the global economy, enabling remote work, e-commerce, streaming entertainment, and instant communication across borders. Emerging technologies like 5G, edge computing, and the Internet of Things promise to further expand connectivity in the coming decades.""",
}

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="main-header" style="font-size:1.4rem">⚙ Config</p>', unsafe_allow_html=True)

    st.markdown('<span class="section-label">Algorithme</span>', unsafe_allow_html=True)
    method = st.selectbox(
        "Method",
        options=["textrank", "tfidf", "frequency", "lsa", "lexrank", "position", "abstractive"],
        format_func=lambda x: {
            "textrank":    "🔗 TextRank (Graph)",
            "tfidf":       "📊 TF-IDF",
            "frequency":   "📈 Frequency",
            "lsa":         "🧮 LSA (Latent Semantic)",
            "lexrank":     "🌐 LexRank (Cosine Graph)",
            "position":    "📍 Position-Biased",
            "abstractive": "🤖 T5 Abstractive ✦",
        }[x],
        label_visibility="collapsed",
    )

    # ── Extractive params ──
    num_sentences = 3
    lead_bias = 1.5
    num_concepts = 3
    lexrank_threshold = 0.1

    if method != "abstractive":
        st.markdown('<span class="section-label">Phrases de sortie</span>', unsafe_allow_html=True)
        num_sentences = st.slider("Sentences", 1, 10, 3, label_visibility="collapsed")

    if method == "position":
        st.markdown('<span class="section-label">Biais Lead/Tail</span>', unsafe_allow_html=True)
        lead_bias = st.slider("Positional bias", 1.0, 3.0, 1.5, step=0.1, label_visibility="collapsed")
    if method == "lsa":
        st.markdown('<span class="section-label">Concepts latents</span>', unsafe_allow_html=True)
        num_concepts = st.slider("SVD concepts (k)", 1, 8, 3, label_visibility="collapsed")
    if method == "lexrank":
        st.markdown('<span class="section-label">Seuil similarité</span>', unsafe_allow_html=True)
        lexrank_threshold = st.slider("Cosine threshold", 0.0, 0.5, 0.1, step=0.05, label_visibility="collapsed")

    # ── T5 params ──
    t5_model = "t5-small"
    max_length = 150
    min_length = 40
    t5_num_beams = 4
    t5_length_penalty = 1.0
    t5_repetition_penalty = 1.2
    t5_no_repeat_ngram = 3
    t5_output_mode = "paragraph"
    t5_chunk_strategy = "auto"

    if method == "abstractive":
        st.markdown('<span class="section-label">Modèle T5</span>', unsafe_allow_html=True)
        t5_model = st.selectbox(
            "T5 Model",
            options=list(T5_MODELS.keys()),
            format_func=lambda x: T5_MODELS[x]["label"],
            label_visibility="collapsed",
        )

        st.markdown('<span class="section-label">Mode de sortie</span>', unsafe_allow_html=True)
        t5_output_mode = st.selectbox(
            "Output mode",
            options=["paragraph", "bullets", "headline"],
            format_func=lambda x: {
                "paragraph": "📄 Paragraphe fluide",
                "bullets":   "• Points clés",
                "headline":  "📰 Titre / Headline",
            }[x],
            label_visibility="collapsed",
        )

        st.markdown('<span class="section-label">Longueur de sortie</span>', unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            max_length = st.number_input("Max tokens", 30, 400, 150, step=10)
        with col_b:
            min_length = st.number_input("Min tokens", 5, 150, 40, step=5)

        with st.expander("🔬 Paramètres avancés T5"):
            t5_num_beams = st.slider("Beam search (num_beams)", 1, 8, 4)
            t5_length_penalty = st.slider("Length penalty", 0.5, 2.0, 1.0, step=0.1,
                help=">1 favorise les résumés plus longs, <1 les raccourcit")
            t5_repetition_penalty = st.slider("Repetition penalty", 1.0, 2.0, 1.2, step=0.05,
                help="Pénalise les répétitions — augmentez si le résumé bégaie")
            t5_no_repeat_ngram = st.slider("No-repeat n-gram", 0, 5, 3,
                help="Interdit la répétition de n-grammes consécutifs")
            t5_chunk_strategy = st.radio(
                "Stratégie long texte",
                options=["auto", "single", "multi"],
                format_func=lambda x: {"auto":"🤖 Auto","single":"✂ Tronquer","multi":"🔀 Multi-chunk"}[x],
                horizontal=True,
            )

    st.divider()
    st.markdown('<span class="section-label">Info méthode</span>', unsafe_allow_html=True)
    info_text = {
        "textrank":    "<strong>TextRank</strong> construit un graphe de phrases et applique un algorithme de type PageRank pour sélectionner les phrases les plus centrales.",
        "tfidf":       "<strong>TF-IDF</strong> note chaque phrase selon l'importance des mots relativement au document.",
        "frequency":   "<strong>Frequency</strong> classe les phrases par fréquence normalisée des mots significatifs.",
        "lsa":         "<strong>LSA</strong> applique la décomposition SVD pour découvrir les sujets latents et sélectionner les phrases les plus riches sémantiquement.",
        "lexrank":     "<strong>LexRank</strong> construit un graphe de similarité cosinus seuillé et applique un PageRank continu.",
        "position":    "<strong>Position-Biased</strong> combine la fréquence des mots avec un biais positionnel : introduction et conclusion sont boostées.",
        "abstractive": "<strong>T5</strong> génère un résumé entièrement nouveau (abstractif) via un transformer seq2seq pré-entraîné.<br><br>✦ Nouveautés : choix du modèle (small/base/large), mode bullet/headline, chunking multi-passage pour longs textes, beam search avancé.",
    }
    st.markdown(f'<div class="info-box">{info_text[method]}</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown('<span class="section-label">Évaluation ROUGE</span>', unsafe_allow_html=True)
    enable_rouge = st.checkbox("Activer ROUGE", value=False)
    reference_text = ""
    if enable_rouge:
        reference_text = st.text_area("Résumé de référence", height=90,
                                      placeholder="Collez un résumé de référence (ground truth)…")

# ─── Main header ─────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">📝 NLP Text Summarizer</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">TF-IDF · TextRank · Frequency · LSA · LexRank · Position · T5 Abstractive</p>', unsafe_allow_html=True)

tab_main, tab_t5, tab_compare, tab_history, tab_about = st.tabs([
    "  Résumer  ", "  T5 Studio  ", "  Comparer  ", "  Historique  ", "  À propos  "
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Main Summarize
# ══════════════════════════════════════════════════════════════════════════════
with tab_main:
    col_input, col_output = st.columns([1, 1], gap="large")

    with col_input:
        st.markdown('<span class="section-label">Texte d\'entrée</span>', unsafe_allow_html=True)
        sample_col, _ = st.columns([2, 1])
        with sample_col:
            chosen_sample = st.selectbox("Charger un exemple",
                                         ["— personnalisé —"] + list(SAMPLES.keys()),
                                         label_visibility="collapsed")
        default_text = SAMPLES.get(chosen_sample, "") if chosen_sample != "— personnalisé —" else ""
        input_text = st.text_area("Input text", value=default_text, height=340,
                                  placeholder="Collez ou tapez le texte à résumer…",
                                  label_visibility="collapsed")
        words_in  = len(input_text.split()) if input_text.strip() else 0
        sents_in  = len(sent_tokenize(input_text)) if input_text.strip() else 0
        st.caption(f"📊 {words_in} mots · {sents_in} phrases · {len(input_text)} caractères")
        run = st.button("▶  Résumer", use_container_width=True)

    with col_output:
        st.markdown('<span class="section-label">Résumé</span>', unsafe_allow_html=True)

        if run:
            if not input_text.strip():
                st.warning("Veuillez saisir du texte.")
            else:
                with st.spinner("Résumé en cours…"):
                    t0 = time.time()
                    kwargs = dict(
                        text=input_text, method=method, reference=reference_text,
                        num_sentences=num_sentences, lead_bias=lead_bias,
                        num_concepts=num_concepts, lexrank_threshold=lexrank_threshold,
                        max_length=max_length, min_length=min_length,
                        t5_model=t5_model, t5_num_beams=t5_num_beams,
                        t5_length_penalty=t5_length_penalty,
                        t5_repetition_penalty=t5_repetition_penalty,
                        t5_no_repeat_ngram=t5_no_repeat_ngram,
                        t5_output_mode=t5_output_mode,
                        t5_chunk_strategy=t5_chunk_strategy,
                    )
                    result = summarize(**kwargs)
                    elapsed = time.time() - t0

                if "error" in result and result["error"]:
                    st.error(result["error"])
                else:
                    add_to_history(result, input_text)
                    summary = result["summary"]
                    stats   = result.get("stats", {})
                    is_t5   = method == "abstractive"

                    # Badge
                    badge_cls = "t5-badge" if is_t5 else "method-badge"
                    st.markdown(f'<div class="{badge_cls}">MÉTHODE: {result["method"]}</div>', unsafe_allow_html=True)

                    # T5 metadata row
                    if is_t5:
                        chunks = result.get("chunks_used", 1)
                        mode_label = {"paragraph":"Paragraphe","bullets":"Points clés","headline":"Titre"}.get(t5_output_mode,"")
                        st.markdown(
                            f'<span style="font-size:0.78rem;color:#5c5a54">Mode: <span style="color:#c9a84c">{mode_label}</span>'
                            f'  ·  Beams: <span style="color:#c9a84c">{t5_num_beams}</span>'
                            f'  ·  Rep. penalty: <span style="color:#c9a84c">{t5_repetition_penalty}</span>'
                            f'{"  <span class=\'chunk-badge\'>🔀 "+str(chunks)+" chunks</span>" if chunks>1 else ""}</span>',
                            unsafe_allow_html=True
                        )

                    # Summary box
                    box_cls = "summary-headline" if t5_output_mode=="headline" and is_t5 else ("summary-box-t5" if is_t5 else "summary-box")
                    st.markdown(f'<div class="{box_cls}">{summary}</div>', unsafe_allow_html=True)

                    # Stats
                    comp = stats.get("compression_ratio", 0)
                    pct  = int(comp * 100)
                    val_cls = "stat-value-gold" if is_t5 else ""
                    fill_cls = "bar-fill-gold" if is_t5 else ""
                    st.markdown(f"""
                    <div class="stat-grid">
                      <div class="stat-box"><span class="stat-value {val_cls}">{stats.get('output_words','—')}</span><span class="stat-label">Mots sortie</span></div>
                      <div class="stat-box"><span class="stat-value {val_cls}">{stats.get('input_words','—')}</span><span class="stat-label">Mots entrée</span></div>
                      <div class="stat-box"><span class="stat-value {val_cls}">{pct}%</span><span class="stat-label">Taille vs original</span></div>
                      <div class="stat-box"><span class="stat-value {val_cls}">{elapsed:.2f}s</span><span class="stat-label">Temps</span></div>
                    </div>
                    <div style="margin-top:0.6rem">
                      <div class="bar-track"><div class="bar-fill {fill_cls}" style="width:{min(pct,100)}%"></div></div>
                      <p style="font-size:0.72rem;color:#5c5a54;margin-top:4px">Compression: {100-pct}% de réduction</p>
                    </div>
                    """, unsafe_allow_html=True)

                    st.code(summary, language=None)

                    # ROUGE
                    if "rouge" in result:
                        st.markdown("---")
                        st.markdown('<span class="section-label">Scores ROUGE</span>', unsafe_allow_html=True)
                        rouge = result["rouge"]
                        rows = ""
                        for metric, vals in rouge.items():
                            h_cls = "highlight-gold" if is_t5 else "highlight-score"
                            rows += f"<tr><td>{metric}</td><td class='{h_cls}'>{vals['f1']}</td><td>{vals['precision']}</td><td>{vals['recall']}</td></tr>"
                        st.markdown(f"""<table class="rouge-table">
                          <tr><th>Métrique</th><th>F1 ↑</th><th>Précision</th><th>Rappel</th></tr>
                          {rows}</table>""", unsafe_allow_html=True)

                    # Sentence scores (extractive)
                    if "scores" in result and result["scores"]:
                        with st.expander("📊 Scores par phrase"):
                            sorted_scores = sorted(result["scores"].items(), key=lambda x: x[1], reverse=True)
                            max_s = max(s for _, s in sorted_scores) if sorted_scores else 1
                            for sent, score in sorted_scores[:8]:
                                bar_w = int(score / max_s * 100)
                                st.markdown(f"""
                                <div style="margin-bottom:0.6rem">
                                  <div style="font-size:0.82rem;color:#b8b4ac;margin-bottom:3px">{sent[:110]}{'…' if len(sent)>110 else ''}</div>
                                  <div style="display:flex;align-items:center;gap:0.6rem">
                                    <div style="flex:1;height:4px;background:#1a1c24;border-radius:2px">
                                      <div style="width:{bar_w}%;height:4px;background:#3a9e7e;border-radius:2px"></div>
                                    </div>
                                    <span style="font-family:'Space Mono',monospace;font-size:0.72rem;color:#3a9e7e">{score:.4f}</span>
                                  </div>
                                </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="height:340px;display:flex;flex-direction:column;align-items:center;
                        justify-content:center;color:#353530;text-align:center">
              <div style="font-size:3rem;margin-bottom:1rem">📄</div>
              <div style="font-family:'Space Mono',monospace;font-size:0.85rem">
                Entrez du texte à gauche,<br>puis cliquez sur Résumer
              </div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — T5 Studio (advanced abstractive playground)
# ══════════════════════════════════════════════════════════════════════════════
with tab_t5:
    st.markdown('<span class="section-label">T5 Abstractive Studio — Playground avancé</span>', unsafe_allow_html=True)

    st.markdown("""
    <div class="card" style="border-color:#9e7e3a">
      <p style="color:#8a8880;font-size:0.9rem;line-height:1.7;margin:0">
        Comparez les trois modes de sortie T5 côte à côte, ou explorez l'effet des hyperparamètres
        sur la qualité du résumé. Idéal pour les longs documents grâce au <strong style="color:#c9a84c">chunking multi-passage</strong>.
      </p>
    </div>""", unsafe_allow_html=True)

    studio_text = st.text_area(
        "Texte à analyser",
        height=180,
        placeholder="Collez un long document ici pour tester les modes T5…",
        label_visibility="collapsed",
        key="studio_input",
    )
    words_studio = len(studio_text.split()) if studio_text.strip() else 0
    st.caption(f"📊 {words_studio} mots — {'⚠️ Long document : chunking automatique activé' if words_studio > 450 else '✓ Document court : passage unique'}")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        studio_model = st.selectbox("Modèle", list(T5_MODELS.keys()),
                                    format_func=lambda x: x, key="studio_model")
    with c2:
        studio_beams = st.number_input("Beams", 1, 8, 4, key="studio_beams")
    with c3:
        studio_maxlen = st.number_input("Max tokens", 30, 300, 130, step=10, key="studio_maxlen")
    with c4:
        studio_rep = st.slider("Rep. penalty", 1.0, 2.0, 1.2, step=0.05, key="studio_rep")

    run_studio = st.button("🤖  Lancer T5 Studio (3 modes)", use_container_width=False, key="studio_btn")

    if run_studio and studio_text.strip():
        modes = [
            ("paragraph", "📄 Paragraphe", "#c9a84c"),
            ("bullets",   "• Points clés",  "#5abda0"),
            ("headline",  "📰 Headline",     "#c87e9e"),
        ]
        cols = st.columns(3)
        for col, (mode, label, color) in zip(cols, modes):
            with col:
                with st.spinner(f"Génération {label}…"):
                    t0 = time.time()
                    r = summarize(
                        studio_text, method="abstractive",
                        max_length=studio_maxlen, min_length=20,
                        t5_model=studio_model, t5_num_beams=studio_beams,
                        t5_repetition_penalty=studio_rep, t5_no_repeat_ngram=3,
                        t5_output_mode=mode, t5_chunk_strategy="auto",
                    )
                    dt = time.time() - t0
                chunks = r.get("chunks_used", 1)
                st.markdown(
                    f'<div style="font-family:\'Space Mono\',monospace;font-size:0.75rem;'
                    f'color:{color};text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.5rem">'
                    f'{label}{"  🔀 "+str(chunks)+" chunks" if chunks>1 else ""}</div>',
                    unsafe_allow_html=True
                )
                st.markdown(
                    f'<div style="background:#12141a;border-left:3px solid {color};'
                    f'padding:1rem 1.2rem;border-radius:0 8px 8px 0;font-size:0.88rem;'
                    f'line-height:1.8;color:#ddd9d0;min-height:160px;white-space:pre-line">'
                    f'{r["summary"]}</div>',
                    unsafe_allow_html=True
                )
                s = r.get("stats", {})
                st.caption(f"{s.get('output_words','?')} mots · {dt:.1f}s")

    elif run_studio:
        st.warning("Veuillez entrer du texte.")

    st.markdown("---")
    st.markdown('<span class="section-label">Guide des hyperparamètres T5</span>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
    <table style="width:100%;border-collapse:collapse;font-size:0.85rem;color:#8a8880">
      <tr style="border-bottom:1px solid #252830">
        <th style="text-align:left;padding:8px;color:#b8b4ac;font-weight:600">Paramètre</th>
        <th style="text-align:left;padding:8px;color:#b8b4ac;font-weight:600">Valeur recommandée</th>
        <th style="text-align:left;padding:8px;color:#b8b4ac;font-weight:600">Effet</th>
      </tr>
      <tr><td style="padding:8px"><code style="color:#c9a84c">num_beams</code></td><td style="padding:8px">4–6</td><td style="padding:8px">Plus élevé = meilleure qualité, plus lent</td></tr>
      <tr style="background:#161820"><td style="padding:8px"><code style="color:#c9a84c">length_penalty</code></td><td style="padding:8px">0.8–1.2</td><td style="padding:8px">&gt;1 favorise les résumés longs, &lt;1 les raccourcit</td></tr>
      <tr><td style="padding:8px"><code style="color:#c9a84c">repetition_penalty</code></td><td style="padding:8px">1.1–1.5</td><td style="padding:8px">Évite les répétitions — crucial pour t5-small</td></tr>
      <tr style="background:#161820"><td style="padding:8px"><code style="color:#c9a84c">no_repeat_ngram</code></td><td style="padding:8px">2–4</td><td style="padding:8px">Interdit la répétition de n-grammes consécutifs</td></tr>
      <tr><td style="padding:8px"><code style="color:#c9a84c">chunk_strategy</code></td><td style="padding:8px">auto</td><td style="padding:8px">Multi-chunk pour &gt;450 mots, sinon single-pass</td></tr>
    </table>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Compare Methods
# ══════════════════════════════════════════════════════════════════════════════
with tab_compare:
    st.markdown('<span class="section-label">Comparer toutes les méthodes extractives</span>', unsafe_allow_html=True)

    cmp_text = st.text_area("Texte à comparer", height=160,
                             placeholder="Collez du texte pour comparer toutes les méthodes extractives…",
                             label_visibility="collapsed", key="cmp_input")
    cmp_n = st.slider("Phrases par résumé", 1, 6, 2, key="cmp_n")
    cmp_run = st.button("⚡  Lancer la comparaison", use_container_width=False, key="cmp_btn")

    if cmp_run and cmp_text.strip():
        methods = ["tfidf", "textrank", "frequency", "lsa", "lexrank", "position"]
        labels  = {"tfidf":"TF-IDF","textrank":"TextRank","frequency":"Frequency",
                   "lsa":"LSA","lexrank":"LexRank","position":"Position-Biased"}
        row1, row2 = st.columns(3), st.columns(3)
        for col, m in zip(list(row1) + list(row2), methods):
            with col:
                r = summarize(cmp_text, method=m, num_sentences=cmp_n)
                s = r.get("stats", {})
                st.markdown(f'<div class="method-badge">{labels[m]}</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="summary-box" style="min-height:140px;font-size:0.88rem">{r["summary"]}</div>',
                    unsafe_allow_html=True
                )
                st.caption(f"{s.get('output_words','?')} mots · {int(s.get('compression_ratio',0)*100)}% de l'original")
    elif cmp_run:
        st.warning("Veuillez entrer du texte.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — History
# ══════════════════════════════════════════════════════════════════════════════
with tab_history:
    st.markdown('<span class="section-label">Historique de la session (10 derniers résumés)</span>', unsafe_allow_html=True)

    if not st.session_state.history:
        st.markdown("""
        <div style="height:200px;display:flex;flex-direction:column;align-items:center;
                    justify-content:center;color:#353530;text-align:center">
          <div style="font-size:2.5rem;margin-bottom:0.8rem">📋</div>
          <div style="font-family:'Space Mono',monospace;font-size:0.82rem">Aucun résumé pour l'instant</div>
        </div>""", unsafe_allow_html=True)
    else:
        if st.button("🗑 Effacer l'historique", key="clear_hist"):
            st.session_state.history = []
            st.rerun()

        for i, entry in enumerate(st.session_state.history):
            is_t5 = "T5" in entry["method"]
            border = "#9e7e3a" if is_t5 else "#252830"
            method_color = "#c9a84c" if is_t5 else "#3a9e7e"
            s = entry["stats"]
            with st.expander(f"[{entry['timestamp']}] {entry['method']} — {s.get('output_words','?')} mots"):
                st.markdown(
                    f'<div class="{"summary-box-t5" if is_t5 else "summary-box"}">{entry["summary"]}</div>',
                    unsafe_allow_html=True
                )
                st.caption(
                    f"Entrée : {s.get('input_words','?')} mots · "
                    f"Sortie : {s.get('output_words','?')} mots · "
                    f"Compression : {int(s.get('compression_ratio',0)*100)}%"
                )
                if entry.get("rouge"):
                    rouge = entry["rouge"]
                    st.markdown(
                        f"ROUGE-1 F1: **{rouge['rouge1']['f1']}**  ·  "
                        f"ROUGE-2 F1: **{rouge['rouge2']['f1']}**  ·  "
                        f"ROUGE-L F1: **{rouge['rougeL']['f1']}**"
                    )
                # Export button
                export_data = json.dumps({
                    "method": entry["method"], "summary": entry["summary"],
                    "stats": entry["stats"], "rouge": entry.get("rouge"),
                }, ensure_ascii=False, indent=2)
                st.download_button(
                    "⬇ Exporter JSON", data=export_data,
                    file_name=f"summary_{entry['timestamp'].replace(':','')}.json",
                    mime="application/json", key=f"dl_{i}",
                )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — About
# ══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("""
<div class="card">
  <p class="main-header" style="font-size:1.2rem">📚 À propos du projet</p>
  <p style="color:#8a8880;line-height:1.8">
    Système de résumé automatique combinant 6 méthodes extractives (sans dépendances lourdes)
    et une méthode abstractive T5 avancée avec chunking multi-passage.
  </p>
</div>

<div class="card">
  <span class="section-label">Méthodes extractives</span>
  <p style="color:#8a8880;line-height:1.8;margin-top:0.5rem">
    <strong style="color:#b8b4ac">TF-IDF</strong> — Note chaque phrase par TF×IDF des mots.<br><br>
    <strong style="color:#b8b4ac">TextRank</strong> — Algorithme de type PageRank sur un graphe de phrases.<br><br>
    <strong style="color:#b8b4ac">Frequency</strong> — Classement par fréquence normalisée des mots significatifs.<br><br>
    <strong style="color:#b8b4ac">LSA</strong> — SVD tronquée sur la matrice termes-phrases pour capturer les sujets latents.<br><br>
    <strong style="color:#b8b4ac">LexRank</strong> — PageRank continu sur graphe de similarité cosinus.<br><br>
    <strong style="color:#b8b4ac">Position-Biased</strong> — Fréquence + biais positionnel (introduction et conclusion boostées).
  </p>
</div>

<div class="card" style="border-color:#9e7e3a">
  <span class="section-label">T5 Abstractive — Nouveautés v2</span>
  <p style="color:#8a8880;line-height:1.8;margin-top:0.5rem">
    <strong style="color:#c9a84c">Choix du modèle</strong> — t5-small (~242MB), t5-base (~892MB), t5-large (~2.9GB).<br><br>
    <strong style="color:#c9a84c">3 modes de sortie</strong> — Paragraphe fluide · Points clés (bullet points) · Headline ultra-court.<br><br>
    <strong style="color:#c9a84c">Chunking multi-passage</strong> — Découpage automatique des textes longs en chunks chevauchants (450 mots, overlap 50), résumé par chunk puis fusion et second passage T5 sur la fusion.<br><br>
    <strong style="color:#c9a84c">Beam search avancé</strong> — Contrôle du nombre de beams, length_penalty, repetition_penalty, no_repeat_ngram_size.<br><br>
    <strong style="color:#c9a84c">T5 Studio</strong> — Onglet dédié pour comparer les 3 modes côte à côte et tester les hyperparamètres.
  </p>
</div>

<div class="card">
  <span class="section-label">Évaluation ROUGE</span>
  <p style="color:#8a8880;line-height:1.8;margin-top:0.5rem">
    <strong style="color:#b8b4ac">ROUGE-1</strong> — Recouvrement unigramme (F1, précision, rappel).<br>
    <strong style="color:#b8b4ac">ROUGE-2</strong> — Recouvrement bigramme.<br>
    <strong style="color:#b8b4ac">ROUGE-L</strong> — Plus longue sous-séquence commune.
  </p>
</div>

<div class="card">
  <span class="section-label">Stack technique</span>
  <p style="font-family:'Space Mono',monospace;font-size:0.82rem;color:#5c5a54;line-height:2">
    Python 3.12 · Streamlit · HuggingFace Transformers · PyTorch · T5 (Google)
  </p>
</div>
    """, unsafe_allow_html=True)
