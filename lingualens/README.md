# LinguaLens – Latin NLP Translation Assistant

A lightweight **Latin NLP tool** that performs:
- **Tokenization**
- **Lemmatization** (optional CLTK/Stanza support)
- **POS tagging** (optional)
- **Rule-based morphological parsing**
- **Dictionary gloss lookup**
- **Assisted translation hints** with confidence scoring

## Why this project?
Latin is highly inflected, so a basic dictionary lookup is not enough. LinguaLens combines
**NLP + rule-based parsing** to produce grammar-aware translation hints.

## Tech Stack
- Python
- Streamlit
- Pandas
- CLTK (optional)
- Stanza (optional)

## Features
- Real-time Latin text analysis in a web UI
- Token-level output: token, lemma, POS, gloss, morphology, confidence
- Rule-based suffix parsing for common noun/verb endings
- Confidence scoring using dictionary + annotation + morphology signals
- CSV export of analysis results

## Quick Start
```bash
pip install -r requirements.txt
streamlit run app.py
```

##Optional: Download Stanza Latin model
```bash
python -c "import stanza; stanza.download('la')"
```bash
