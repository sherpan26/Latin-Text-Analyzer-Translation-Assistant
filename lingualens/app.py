from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import streamlit as st

from analyzer import LatinAnalyzer


st.set_page_config(page_title="LinguaLens", page_icon="🏛️", layout="wide")

st.title("🏛️ LinguaLens – Latin NLP Translation Assistant")
st.caption(
    "Grammar-aware Latin text analysis using tokenization, lemmatization, morphology heuristics, and gloss ranking."
)

# Cache analyzer so app feels fast after first load
@st.cache_resource
def get_analyzer() -> LatinAnalyzer:
    return LatinAnalyzer(dictionary_path="latin_dictionary.csv")


analyzer = get_analyzer()

# Sidebar
st.sidebar.header("Settings")
show_raw = st.sidebar.checkbox("Show raw token table columns", value=False)
use_samples = st.sidebar.checkbox("Show sample inputs", value=True)

if use_samples:
    sample_path = Path("sample_inputs.txt")
    if sample_path.exists():
        samples = [line.strip() for line in sample_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        selected_sample = st.sidebar.selectbox("Sample Latin input", ["(None)"] + samples)
    else:
        selected_sample = "(None)"
else:
    selected_sample = "(None)"

default_text = "puella rosam amat"
if selected_sample != "(None)":
    default_text = selected_sample

text = st.text_area("Enter Latin text", value=default_text, height=120, placeholder="e.g., puella rosam amat")

colA, colB = st.columns([1, 1])
with colA:
    run_btn = st.button("Analyze Text", type="primary", use_container_width=True)
with colB:
    clear_btn = st.button("Clear", use_container_width=True)

if clear_btn:
    st.rerun()

if run_btn and text.strip():
    start = time.perf_counter()
    df, summary = analyzer.analyze_text(text)
    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

    # Metrics row
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Tokens", summary["token_count"])
    m2.metric("Gloss Coverage", f"{summary['coverage_pct']}%")
    m3.metric("Resolved Glosses", summary["resolved_glosses"])
    m4.metric("Avg Token Score", summary["avg_token_confidence"])
    m5.metric("Latency", f"{elapsed_ms} ms")

    st.subheader("Assisted Translation")
    st.success(summary["translation_hint"] or "No translation hint generated.")

    if summary["sentence_pattern_detected"]:
        st.info("Detected a likely grammar pattern (e.g., nominative noun + accusative noun + verb).")
    else:
        st.warning("No clear sentence pattern detected; showing word-by-word gloss fallback.")

    st.subheader("Token-Level Analysis")
    display_df = df.copy()

    # Nice column names for UI
    display_df = display_df.rename(
        columns={
            "token": "Token",
            "normalized": "Normalized",
            "lemma": "Lemma",
            "pos": "POS",
            "gloss": "Gloss",
            "morphology": "Morphology",
            "notes": "Notes",
            "token_confidence": "Token Score (0-5)",
        }
    )

    if not show_raw:
        keep_cols = ["Token", "Lemma", "POS", "Gloss", "Morphology", "Token Score (0-5)"]
        display_df = display_df[keep_cols]

    st.dataframe(display_df, use_container_width=True)

    # Download CSV
    csv_bytes = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Analysis CSV",
        data=csv_bytes,
        file_name="lingualens_analysis.csv",
        mime="text/csv",
    )

    # Optional NLP availability status
    st.subheader("Pipeline Status")
    c1, c2 = st.columns(2)
    c1.write(f"**CLTK available:** {'✅' if analyzer.cltk_available else '❌'}")
    c2.write(f"**Stanza available:** {'✅' if analyzer.stanza_available else '❌'}")

    if not analyzer.stanza_available:
        st.caption(
            "Tip: To enable Stanza Latin models, run: "
            "`python -c \"import stanza; stanza.download('la')\"`"
        )

elif run_btn and not text.strip():
    st.error("Please enter some Latin text.")
