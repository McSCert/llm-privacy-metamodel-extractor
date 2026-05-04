"""
chunker.py — Hierarchical legal text chunker for PrivacyPolicyMetamodel v4.

Design rationale
----------------
Legal statutes have a strict hierarchy:
    Part → Chapter → Section → Article → Paragraph/Clause

Flat chunking (e.g. fixed 512-token windows) destroys this structure and
causes the RAG retriever to mix clauses from different articles, producing
hallucinated cross-article extractions.

This module preserves the hierarchy by:
1. Detecting structural boundaries with law-specific regex patterns.
2. Assigning every chunk a (law, article_ref, level, parent_ref) tuple.
3. Attaching concept_tags — lightweight keyword heuristics that tell the
   retriever which metamodel concepts a chunk is likely to contain.

The result is that retrieve("LegalBasis", "GDPR") can pre-filter to chunks
tagged "LegalBasis" before running cosine similarity, keeping retrieval
both fast and precise.

Supported input formats
-----------------------
- PDF via pdfplumber   (preferred — preserves layout)
- Plain text .txt      (fallback for pre-extracted text)

Supported laws (regex patterns pre-configured)
----------------------------------------------
- GDPR          Article N, Recital N
- LGPD          Artigo N / Art. N
- CCPA / CPRA   Section NNNN / § NNNN
- PIPEDA        Schedule N / Principle N
- Generic       Article N / Section N / § N  (catches most other laws)
"""

from __future__ import annotations

import re
import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

log = logging.getLogger(__name__)


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    """
    One structural unit of a legal text.

    chunk_id      : SHA-1 of (law + article_ref + text[:64]) — stable across runs.
    law           : canonical short name, e.g. "GDPR", "CCPA", "LGPD".
    article_ref   : structural reference, e.g. "Art.6", "Art.6(1)(a)", "§1798.100".
    parent_ref    : reference of the enclosing structural unit, e.g. "Art.6" for
                    a clause chunk, or "" for top-level articles.
    level         : hierarchy level — "part", "chapter", "article", "clause".
    text          : raw text of the chunk (no headers stripped — they provide context).
    char_offset   : character offset of the chunk's first character in the source doc.
    concept_tags  : list of metamodel class names this chunk likely contains,
                    e.g. ["LegalBasis", "ConsentWithdrawal"].
                    Populated by concept_tagger() after chunking.
    """
    chunk_id:     str
    law:          str
    article_ref:  str
    parent_ref:   str
    level:        str
    text:         str
    char_offset:  int
    concept_tags: list[str] = field(default_factory=list)

    @classmethod
    def make(
        cls,
        law: str,
        article_ref: str,
        parent_ref: str,
        level: str,
        text: str,
        char_offset: int,
    ) -> "Chunk":
        raw = f"{law}|{article_ref}|{text[:64]}"
        chunk_id = hashlib.sha1(raw.encode()).hexdigest()[:16]
        return cls(
            chunk_id=chunk_id,
            law=law,
            article_ref=article_ref,
            parent_ref=parent_ref,
            level=level,
            text=text.strip(),
            char_offset=char_offset,
        )


# ── Structural regex patterns per law ─────────────────────────────────────────

# Each entry: list of (level, compiled_regex) in DESCENDING hierarchy order.
# The chunker scans for the highest-level matches first, then sub-chunks within
# each match for the next level down.

_GDPR_PATTERNS = [
    ("chapter", re.compile(
        r"^CHAPTER\s+[IVXLCDM]+\b",
        re.MULTILINE | re.IGNORECASE,
    )),
    ("article", re.compile(
        r"^Article\s+\d+\b",
        re.MULTILINE | re.IGNORECASE,
    )),
    ("clause", re.compile(
        r"^\s*\d+\.\s+",           # "1. Processing shall be..."
        re.MULTILINE,
    )),
]

_LGPD_PATTERNS = [
    ("chapter", re.compile(
        r"^CAP[IÍ]TULO\s+[IVXLCDM]+\b",
        re.MULTILINE | re.IGNORECASE,
    )),
    ("article", re.compile(
        r"^Art(?:igo)?\.?\s*\d+[oº]?\b",
        re.MULTILINE | re.IGNORECASE,
    )),
    ("clause", re.compile(
        r"^\s*[§§]\s*\d+[oº]?\b|^\s*\d+\.\s+",
        re.MULTILINE,
    )),
]

_CCPA_PATTERNS = [
    ("section", re.compile(
        r"^(?:Section\s+|§\s*)1798\.\d+(?:\.\d+)*\b",
        re.MULTILINE | re.IGNORECASE,
    )),
    ("clause", re.compile(
        r"^\s*\([a-z]\)\s+",       # "(a) For purposes of this title..."
        re.MULTILINE,
    )),
]

_PIPEDA_PATTERNS = [
    ("schedule", re.compile(
        r"^Schedule\s+\d+\b",
        re.MULTILINE | re.IGNORECASE,
    )),
    # Principle level: "4.1 Principle 1 — Accountability", "4.2 Principle 2 — ..."
    # PIPEDA numbers all principles as X.Y (e.g. 4.1, 4.2 ... 4.10).
    # Previous regex (^Principle\s+\d+ | ^\d+\.\s+\w+) failed because:
    #   - Lines start with "4.1" not "Principle"
    #   - ^\d+\.\s+\w+ needs whitespace after dot but finds digit ("1" in "4.1")
    # Fix: match the actual format — one or more digits, dot, one or more digits, space.
    ("principle", re.compile(
        r"^\d+\.\d+\s+",
        re.MULTILINE,
    )),
    # Clause level: sub-clauses within a principle
    #   "4.3.1 An organization shall..."  — numbered sub-clause
    #   "(a) implementing procedures..."  — lettered sub-clause (common in PIPEDA)
    ("clause", re.compile(
        r"^\s*\d+\.\d+\.\d+\s+|^\s*\([a-z]\)\s+",
        re.MULTILINE,
    )),
]

_GENERIC_PATTERNS = [
    ("chapter", re.compile(
        r"^(?:CHAPTER|PART|TITLE)\s+[IVXLCDM\d]+\b",
        re.MULTILINE | re.IGNORECASE,
    )),
    ("article", re.compile(
        r"^(?:Article|Section|Art\.?|§)\s*\d+\b",
        re.MULTILINE | re.IGNORECASE,
    )),
    ("clause", re.compile(
        r"^\s*\d+\.\s+|\s*\([a-z]\)\s+",
        re.MULTILINE,
    )),
]

_LAW_PATTERNS: dict[str, list] = {
    "GDPR":   _GDPR_PATTERNS,
    "LGPD":   _LGPD_PATTERNS,
    "CCPA":   _CCPA_PATTERNS,
    "CPRA":   _CCPA_PATTERNS,
    "PIPEDA": _PIPEDA_PATTERNS,
}


# ── Concept tagger ────────────────────────────────────────────────────────────

# Keyword sets per metamodel class — deliberately broad to maximise recall.
# Precision is handled by the cosine similarity re-ranking step in retriever.py.

_CONCEPT_KEYWORDS: dict[str, list[str]] = {
    # ── Actor ─────────────────────────────────────────────────────────────────
    # Previously MISSING — caused Actor to never be extracted from any chunk.
    # PIPEDA: "organization", "accountable", "responsible", "designate"
    # GDPR:   "controller", "processor", "data subject"
    # CCPA:   "business", "service provider", "third party"
    "Actor": [
        # GDPR / LGPD
        "controller", "processor", "data subject", "third party",
        "controlador", "operador", "titular",
        # PIPEDA
        "organization", "accountable", "accountability", "responsible",
        "designate", "designated individual", "compliance officer",
        # CCPA
        "business", "service provider", "contractor",
        # Generic
        "data fiduciary", "data principal", "individual",
    ],

    "LegalBasis": [
        # GDPR
        "lawful", "legal basis", "consent", "contract", "legitimate interest",
        "legal obligation", "vital interest", "public task",
        # LGPD
        "consentimento", "base legal", "hipótese", "legítimo interesse",
        # PIPEDA — uses "knowledge and consent" not "legal basis"
        "knowledge and consent", "without knowledge", "without consent",
        "implied consent", "express consent", "opt-in", "opt-out",
        # CCPA
        "business purpose", "commercial purpose", "authorized",
    ],

    "ProcessingActivity": [
        # Generic
        "collect", "store", "use", "share", "transfer", "delete", "process",
        "processing", "disclose", "retain", "handle", "record", "transmit",
        "gather", "compile", "aggregate", "combine", "analyse", "analyze",
        # PIPEDA
        "collecting", "using", "disclosing", "personal information",
        "make use of",
        # LGPD
        "colet", "tratar", "tratamento", "armazenar", "compartilhar",
    ],

    # ── Constraint ────────────────────────────────────────────────────────────
    # Previously MISSING — caused Constraint to never be extracted from any chunk.
    # PIPEDA: "limiting", "safeguard", "only for the purpose", "shall not"
    # GDPR:   "purpose limitation", "data minimisation", "necessity"
    "Constraint": [
        # GDPR
        "purpose limitation", "data minimisation", "minimization",
        "necessity", "adequate", "relevant", "not excessive",
        "shall not", "must not", "prohibited", "restricted",
        "only for", "solely for", "limited to",
        # PIPEDA
        "limiting", "safeguard", "only for the purpose", "not use",
        "not disclose", "not collect", "beyond the purposes",
        "shall not use", "shall not disclose",
        # CCPA
        "may not", "cannot sell", "do not sell",
        # LGPD
        "não poderá", "vedado", "limitado",
        # Generic
        "restriction", "prohibited from", "compliance", "obligation",
        "must", "shall", "policy", "procedure", "implement",
    ],

    "RetentionPolicy": [
        # GDPR
        "retain", "retention", "storage limitation", "no longer than necessary",
        "delete", "deletion", "erase", "erasure", "keep", "period", "duration",
        # PIPEDA
        "as long as necessary", "destroy", "destroyed", "no longer required",
        "kept only", "retained only", "retention schedule",
        # LGPD
        "prazo", "conservação", "armazenamento", "eliminação",
        # Generic
        "archive", "purge", "dispose",
    ],

    "ConsentWithdrawal": [
        # GDPR
        "withdraw", "withdrawal", "revoke", "revocation",
        "as easy as", "at any time",
        # PIPEDA
        "opt out", "opt-out", "unsubscribe", "withdraw consent",
        "right to withdraw", "challenge compliance",
        # LGPD
        "retirar", "revogar", "reti",
        # CCPA
        "opt-out of sale", "do not sell my",
    ],

    "DataTransfer": [
        # GDPR
        "third country", "international transfer", "adequacy",
        "standard contractual", "binding corporate rules", "bcr", "sccs",
        "cross-border", "recipient country",
        # PIPEDA
        "transfer to a third party", "third-party",
        "transferred outside", "transferred to",
        # LGPD
        "transferência internacional", "país terceiro",
        # Generic
        "transfer", "international",
    ],

    "Right": [
        # GDPR
        "right to", "rights of", "access", "rectification", "erasure",
        "restriction", "portability", "objection", "automated",
        # PIPEDA
        "challenge", "individual access", "access to personal",
        "make a complaint", "right of access",
        # CCPA
        "opt-out", "right to know", "right to delete", "right to correct",
        # LGPD
        "direito", "acesso", "retificação", "exclusão",
    ],

    "Purpose": [
        # Generic
        "purpose", "purposes", "objective", "goal",
        "specific purpose", "processing purpose",
        # PIPEDA — "identifying purposes" is Principle 2
        "identifying purposes", "identified purpose", "stated purpose",
        "for which", "intended purpose", "business purpose",
        # LGPD
        "finalidade", "específica",
    ],

    "PersonalData": [
        "personal data", "personal information", "sensitive", "special category",
        "biometric", "health", "financial", "location",
        "dados pessoais", "dados sensíveis", "informação pessoal",
    ],
}


def concept_tagger(text: str) -> list[str]:
    """
    Return a list of metamodel class names likely present in the chunk text.
    Uses case-insensitive keyword matching — fast, deterministic, no model needed.
    """
    text_lower = text.lower()
    tags = []
    for concept, keywords in _CONCEPT_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            tags.append(concept)
    return tags


# ── Text extraction from PDF ───────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract raw text from a PDF preserving paragraph structure.
    Uses pdfplumber with layout-aware extraction.
    Falls back to a simple join on extraction failure.
    """
    try:
        import pdfplumber
    except ImportError:
        raise ImportError("pdfplumber is required for PDF extraction. pip install pdfplumber")

    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text(x_tolerance=2, y_tolerance=3)
            if text:
                pages.append(text)

    return "\n\n".join(pages)


def extract_text_from_file(path: Path) -> str:
    """Route to correct extractor based on file extension."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(path)
    elif suffix in (".txt", ".md"):
        return path.read_text(encoding="utf-8", errors="replace")
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Use .pdf or .txt")


# ── Core chunking logic ───────────────────────────────────────────────────────

def _split_by_pattern(
    text: str,
    pattern: re.Pattern,
    level: str,
    law: str,
    parent_ref: str,
    base_offset: int,
) -> list[Chunk]:
    """
    Split `text` at every match of `pattern`.
    Each segment from one match boundary to the next becomes one Chunk.
    The matched header text is included in the chunk (provides context for embedding).
    """
    chunks = []
    matches = list(pattern.finditer(text))

    if not matches:
        return chunks

    for i, m in enumerate(matches):
        start = m.start()
        end   = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        segment = text[start:end]

        if len(segment.strip()) < 30:   # skip near-empty segments
            continue

        # Article ref = the matched header line, cleaned
        article_ref = segment.split("\n")[0].strip()
        article_ref = re.sub(r"\s+", " ", article_ref)[:80]

        chunks.append(Chunk.make(
            law=law,
            article_ref=article_ref,
            parent_ref=parent_ref,
            level=level,
            text=segment,
            char_offset=base_offset + start,
        ))

    return chunks


def chunk_text(
    text: str,
    law: str,
    min_chunk_chars: int = 100,
    max_chunk_chars: int = 4000,
) -> list[Chunk]:
    """
    Hierarchically chunk a legal text string.

    Parameters
    ----------
    text            : full document text (from extract_text_from_file)
    law             : canonical law name — drives which regex patterns are used
    min_chunk_chars : segments shorter than this are merged into their parent
    max_chunk_chars : segments longer than this are split at paragraph boundaries

    Returns
    -------
    List of Chunk objects in document order, tagged with concept_tags.
    """
    patterns = _LAW_PATTERNS.get(law.upper(), _GENERIC_PATTERNS)
    all_chunks: list[Chunk] = []

    log.info(f"Chunking {law} document ({len(text):,} chars)")

    # ── Level 0: top-level split (chapter / schedule / section) ──────────────
    top_level_pattern = patterns[0]
    top_level_name    = top_level_pattern[0]
    top_level_regex   = top_level_pattern[1]

    top_chunks = _split_by_pattern(
        text, top_level_regex, top_level_name, law, parent_ref="", base_offset=0
    )

    # If the document has no detectable top-level structure, treat entire text
    # as one top-level chunk and fall through to article splitting.
    if not top_chunks:
        log.debug(f"No {top_level_name} boundaries found — treating as flat document")
        top_chunks = [Chunk.make(
            law=law, article_ref=law, parent_ref="",
            level="document", text=text, char_offset=0,
        )]

    # ── Level 1: article / section split within each top-level chunk ─────────
    if len(patterns) >= 2:
        article_pattern = patterns[1][1]
        article_level   = patterns[1][0]

        for top in top_chunks:
            art_chunks = _split_by_pattern(
                top.text, article_pattern, article_level,
                law, parent_ref=top.article_ref, base_offset=top.char_offset,
            )

            if not art_chunks:
                # No articles found — keep the top-level chunk as-is
                all_chunks.append(top)
                continue

            # ── Level 2: clause split within each article ─────────────────
            if len(patterns) >= 3:
                clause_pattern = patterns[2][1]
                clause_level   = patterns[2][0]

                for art in art_chunks:
                    clause_chunks = _split_by_pattern(
                        art.text, clause_pattern, clause_level,
                        law, parent_ref=art.article_ref,
                        base_offset=art.char_offset,
                    )

                    # Always keep the article itself (gives full-article context)
                    all_chunks.append(art)

                    for cl in clause_chunks:
                        if len(cl.text) >= min_chunk_chars:
                            all_chunks.append(cl)
            else:
                all_chunks.extend(art_chunks)
    else:
        all_chunks.extend(top_chunks)

    # ── Post-processing: tag concepts and split oversized chunks ─────────────
    final: list[Chunk] = []
    for ch in all_chunks:
        ch.concept_tags = concept_tagger(ch.text)

        if len(ch.text) > max_chunk_chars:
            # Split at paragraph boundary (\n\n) without losing metadata
            for sub in _split_large_chunk(ch, max_chunk_chars):
                final.append(sub)
        else:
            final.append(ch)

    log.info(
        f"Produced {len(final)} chunks for {law} "
        f"(articles: {sum(1 for c in final if c.level in ('article','section','principle'))},"
        f" clauses: {sum(1 for c in final if c.level in ('clause',))})"
    )
    return final


def _split_large_chunk(chunk: Chunk, max_chars: int) -> list[Chunk]:
    """
    Split an oversized chunk at paragraph boundaries (blank lines).
    Preserves all metadata from the parent chunk.
    """
    paragraphs = re.split(r"\n{2,}", chunk.text)
    sub_chunks = []
    current_text = ""
    current_offset = chunk.char_offset

    for para in paragraphs:
        if len(current_text) + len(para) > max_chars and current_text:
            sub_chunks.append(Chunk.make(
                law=chunk.law,
                article_ref=chunk.article_ref,
                parent_ref=chunk.parent_ref,
                level=chunk.level,
                text=current_text,
                char_offset=current_offset,
            ))
            current_offset += len(current_text)
            current_text = para
        else:
            current_text = current_text + "\n\n" + para if current_text else para

    if current_text.strip():
        sub_chunks.append(Chunk.make(
            law=chunk.law,
            article_ref=chunk.article_ref,
            parent_ref=chunk.parent_ref,
            level=chunk.level,
            text=current_text,
            char_offset=current_offset,
        ))

    # Re-tag the sub-chunks
    for sc in sub_chunks:
        sc.concept_tags = concept_tagger(sc.text)

    return sub_chunks if sub_chunks else [chunk]


# ── Public entry point ────────────────────────────────────────────────────────

def chunk_file(
    path: Path | str,
    law: str,
    min_chunk_chars: int = 100,
    max_chunk_chars: int = 4000,
) -> list[Chunk]:
    """
    Full pipeline: file → text → chunks.

    Parameters
    ----------
    path            : path to .pdf or .txt file
    law             : canonical law name, e.g. "GDPR", "CCPA", "LGPD"
    min_chunk_chars : minimum chunk size (shorter chunks merged to parent)
    max_chunk_chars : maximum chunk size (longer chunks split at paragraphs)

    Returns
    -------
    List of tagged Chunk objects ready for embedding and storage.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    log.info(f"Loading {path.name} as {law}")
    text = extract_text_from_file(path)
    return chunk_text(text, law, min_chunk_chars, max_chunk_chars)