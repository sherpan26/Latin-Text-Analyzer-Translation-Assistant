from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

from morphology import infer_morphology


@dataclass
class TokenAnalysis:
    token: str
    normalized: str
    lemma: str
    pos: str
    gloss: str
    morphology: str
    notes: str
    token_confidence: int  # 0-5


class LatinAnalyzer:
    def __init__(self, dictionary_path: str = "latin_dictionary.csv") -> None:
        self.dictionary_df = pd.read_csv(dictionary_path).fillna("")
        self.dictionary_map = self._build_dictionary_map(self.dictionary_df)

        self.cltk_available = False
        self.stanza_available = False
        self._cltk_nlp = None
        self._stanza_nlp = None

        self._init_optional_nlp()

    @staticmethod
    def _build_dictionary_map(df: pd.DataFrame) -> Dict[str, List[Dict[str, str]]]:
        mapping: Dict[str, List[Dict[str, str]]] = {}
        for _, row in df.iterrows():
            key = str(row["word"]).strip().lower()
            if not key:
                continue
            mapping.setdefault(key, []).append(
                {
                    "lemma": str(row.get("lemma", "")).strip(),
                    "pos": str(row.get("pos", "")).strip(),
                    "gloss": str(row.get("gloss", "")).strip(),
                }
            )
        return mapping

    def _init_optional_nlp(self) -> None:
        # CLTK (optional)
        try:
            from cltk import NLP  # type: ignore
            self._cltk_nlp = NLP(language="lat")
            self.cltk_available = True
        except Exception:
            self.cltk_available = False
            self._cltk_nlp = None

        # Stanza (optional)
        try:
            import stanza  # type: ignore
            # We do NOT force model download here (keeps first run simple).
            # User can run: python -c "import stanza; stanza.download('la')"
            try:
                self._stanza_nlp = stanza.Pipeline(
                    "la",
                    processors="tokenize,pos,lemma",
                    use_gpu=False,
                    verbose=False,
                )
                self.stanza_available = True
            except Exception:
                self.stanza_available = False
                self._stanza_nlp = None
        except Exception:
            self.stanza_available = False
            self._stanza_nlp = None

    @staticmethod
    def tokenize(text: str) -> List[str]:
        # Keep alphabetic + apostrophe-like patterns simple
        return re.findall(r"[A-Za-z]+", text)

    def _lookup_dictionary(self, token: str) -> Optional[Dict[str, str]]:
        matches = self.dictionary_map.get(token.lower())
        if not matches:
            return None
        # choose first for demo simplicity
        return matches[0]

    def _cltk_token_data(self, text: str) -> Dict[str, Dict[str, str]]:
        """
        Best-effort CLTK extraction. Returns token->attrs map.
        CLTK APIs can vary by version, so this stays defensive.
        """
        out: Dict[str, Dict[str, str]] = {}
        if not self.cltk_available or self._cltk_nlp is None:
            return out

        try:
            doc = self._cltk_nlp.analyze(text)
            # CLTK objects differ by versions; try common attributes.
            words = getattr(doc, "words", None)
            if words:
                for w in words:
                    token = str(getattr(w, "string", getattr(w, "text", ""))).strip()
                    if not token:
                        continue
                    lemma = str(getattr(w, "lemma", "") or "").strip()
                    pos = str(getattr(w, "upos", getattr(w, "pos", "")) or "").strip()
                    if token:
                        out[token.lower()] = {"lemma": lemma, "pos": pos}
        except Exception:
            return {}
        return out

    def _stanza_token_data(self, text: str) -> Dict[str, Dict[str, str]]:
        out: Dict[str, Dict[str, str]] = {}
        if not self.stanza_available or self._stanza_nlp is None:
            return out

        try:
            doc = self._stanza_nlp(text)
            for sent in doc.sentences:
                for word in sent.words:
                    token = (word.text or "").strip()
                    if not token:
                        continue
                    out[token.lower()] = {
                        "lemma": (word.lemma or "").strip(),
                        "pos": (word.upos or "").strip(),
                    }
        except Exception:
            return {}
        return out

    @staticmethod
    def _format_morph(features: Dict[str, str]) -> str:
        keys = ["pos_guess", "case", "number", "person", "tense"]
        parts = [f"{k}:{features[k]}" for k in keys if k in features and features[k]]
        return "; ".join(parts) if parts else "unknown"

    def analyze_text(self, text: str) -> Tuple[pd.DataFrame, Dict[str, object]]:
        tokens = self.tokenize(text)
        cltk_map = self._cltk_token_data(text)
        stanza_map = self._stanza_token_data(text)

        rows: List[TokenAnalysis] = []

        for tok in tokens:
            normalized = tok.lower()
            dict_hit = self._lookup_dictionary(normalized)
            morph = infer_morphology(normalized)

            cltk_hit = cltk_map.get(normalized, {})
            stanza_hit = stanza_map.get(normalized, {})

            lemma = ""
            pos = ""
            gloss = ""
            notes: List[str] = []
            score = 0

            # dictionary
            if dict_hit:
                lemma = dict_hit.get("lemma", "") or lemma
                pos = dict_hit.get("pos", "") or pos
                gloss = dict_hit.get("gloss", "") or gloss
                score += 1
                notes.append("dictionary match")

            # CLTK
            if cltk_hit:
                if cltk_hit.get("lemma"):
                    lemma = cltk_hit["lemma"]
                    score += 1
                if cltk_hit.get("pos"):
                    pos = cltk_hit["pos"]
                    score += 1
                notes.append("CLTK annotation")

            # Stanza
            if stanza_hit:
                if stanza_hit.get("lemma") and not lemma:
                    lemma = stanza_hit["lemma"]
                elif stanza_hit.get("lemma"):
                    # count only if new signal
                    score += 1
                if stanza_hit.get("pos") and not pos:
                    pos = stanza_hit["pos"]
                elif stanza_hit.get("pos"):
                    score += 1
                notes.append("Stanza annotation")

            # morphology rule
            if morph.get("rule_match") != "none":
                score += 1
                notes.append(f"suffix rule: {morph.get('rule_match')}")

            if not lemma:
                lemma = normalized
            if not pos:
                pos = morph.get("pos_guess", "unknown")
            if not gloss:
                gloss = "(no dictionary gloss found)"

            morphology_str = self._format_morph(morph)

            rows.append(
                TokenAnalysis(
                    token=tok,
                    normalized=normalized,
                    lemma=lemma,
                    pos=pos,
                    gloss=gloss,
                    morphology=morphology_str,
                    notes=", ".join(dict.fromkeys(notes)) if notes else "heuristic only",
                    token_confidence=min(score, 5),
                )
            )

        df = pd.DataFrame([r.__dict__ for r in rows])

        summary = self._build_summary(df)
        return df, summary

    def _build_summary(self, df: pd.DataFrame) -> Dict[str, object]:
        if df.empty:
            return {
                "token_count": 0,
                "resolved_glosses": 0,
                "coverage_pct": 0.0,
                "avg_token_confidence": 0.0,
                "translation_hint": "",
                "sentence_pattern_detected": False,
            }

        resolved_glosses = (df["gloss"] != "(no dictionary gloss found)").sum()
        token_count = len(df)
        coverage_pct = round((resolved_glosses / token_count) * 100, 1) if token_count else 0.0
        avg_conf = round(float(df["token_confidence"].mean()), 2)

        translation_hint, pattern_detected = self._assisted_translation(df)

        return {
            "token_count": token_count,
            "resolved_glosses": int(resolved_glosses),
            "coverage_pct": coverage_pct,
            "avg_token_confidence": avg_conf,
            "translation_hint": translation_hint,
            "sentence_pattern_detected": pattern_detected,
        }

    def _assisted_translation(self, df: pd.DataFrame) -> Tuple[str, bool]:
        """
        Very simple grammar-aware hint:
        Try NOM-like noun + ACC-like noun + verb pattern.
        Falls back to word-by-word gloss.
        """
        if df.empty:
            return "", False

        # Build simple row helpers
        rows = df.to_dict(orient="records")
        verb_idx = None
        subj_idx = None
        obj_idx = None

        for i, r in enumerate(rows):
            pos = str(r.get("pos", "")).lower()
            morph = str(r.get("morphology", "")).lower()

            if verb_idx is None and "verb" in pos:
                verb_idx = i

            if subj_idx is None and ("noun" in pos or "adj" in pos) and "case:nominative" in morph:
                subj_idx = i

            if obj_idx is None and ("noun" in pos or "adj" in pos) and "case:accusative" in morph:
                obj_idx = i

        # Compose smarter translation if pattern found
        if verb_idx is not None and subj_idx is not None:
            subj = rows[subj_idx]["gloss"].split("(")[0].strip()
            verb = rows[verb_idx]["gloss"].split("(")[0].strip()
            if obj_idx is not None:
                obj = rows[obj_idx]["gloss"].split("(")[0].strip()
                return f"Possible translation: The {subj} {verb} the {obj}.", True
            return f"Possible translation: The {subj} {verb}.", True

        # Fallback: word-by-word gloss
        glosses = []
        for r in rows:
            g = str(r.get("gloss", "")).strip()
            glosses.append(g.split("(")[0].strip() if g else r.get("token", ""))
        return "Word-by-word gloss: " + " ".join(glosses), False
