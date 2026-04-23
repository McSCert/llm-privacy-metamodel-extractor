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

Concept tagger design
---------------------
concept_tagger(text, law) is LAW-AWARE.

GDPR/LGPD/CCPA use GDPR-derived vocabulary ("processing", "legal basis",
"erasure", etc.). PIPEDA uses a completely different vocabulary
("collection, use or disclosure", "knowledge and consent", "individual
access", "render anonymous"). Using a single keyword list across laws
causes systematic false-positive tagging that burns LLM retries on
articles where the concept is simply not present.

Each law group has its own keyword dict. Unknown laws fall back to the
GDPR/general set. Extend _PIPEDA_CONCEPT_KEYWORDS when adding PIPEDA-like
laws (e.g., Canada's Bill C-27 / CPPA uses similar vocabulary).
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

""" # BEFORE — overly broad, fires on any numbered line
_PIPEDA_PATTERNS = [
    ("schedule", re.compile(
        r"^Schedule\s+\d+\b",
        re.MULTILINE | re.IGNORECASE,
    )),
    ("principle", re.compile(
        r"^Principle\s+\d+\b|^\d+\.\s+Accountability\b|^\d+\.\s+\w+",
        re.MULTILINE | re.IGNORECASE,
    )),
    ("clause", re.compile(
        r"^\s*\d+\.\d+\s+",
        re.MULTILINE,
    )),
] """

# AFTER — targets only the actual 10 PIPEDA principle headers
_PIPEDA_PATTERNS = [
    ("schedule", re.compile(
        r"^Schedule\s+1\b",          # Only Schedule 1 — the privacy principles
        re.MULTILINE | re.IGNORECASE,
    )),
    ("principle", re.compile(
        # Matches "4.1 Accountability", "4.2 Identifying Purposes", etc.
        # The \d+\.\d+ pattern is specific to the X.Y numbering PIPEDA uses.
        # Also matches explicit "Principle N" headers if present.
        r"^(?:Principle\s+\d+\b|"
        r"4\.(?:1|2|3|4|5|6|7|8|9|10)\s)",
        re.MULTILINE | re.IGNORECASE,
    )),
    ("clause", re.compile(
        # Matches sub-clauses like "4.1.1", "4.3.2", etc.
        r"^\s*4\.\d+\.\d+\s+",
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


# ── Concept keyword sets ──────────────────────────────────────────────────────
#
# Two separate keyword dicts: one for GDPR-family laws, one for PIPEDA.
#
# Design principle: prefer MULTI-WORD phrases over single words wherever
# possible. Single words ("access", "transfer", "retain") are the primary
# source of false-positive tagging. A multi-word phrase that is specific
# to a concept beats three single-word keywords that fire on everything.
#
# Recall vs precision trade-off:
#   - The tagger is the FIRST filter. False negatives here mean the concept
#     is never extracted (silent miss). False positives cause wasted LLM
#     calls, caught by the absence check in run_pipeline.py.
#   - Err slightly toward recall (add a keyword when unsure), but not with
#     words so generic they fire on every article.

# ── GDPR / LGPD / CCPA / generic ─────────────────────────────────────────────

_GDPR_CONCEPT_KEYWORDS: dict[str, list[str]] = {

    "LegalBasis": [
        "lawful", "legal basis", "legal ground",
        "consent", "contract", "legitimate interest",
        "legal obligation", "vital interest", "public task",
        # LGPD
        "consentimento", "base legal", "hipótese",
        # CCPA
        "business purpose", "commercial purpose",
    ],

    "ProcessingActivity": [
        "collect", "store", "use", "share", "transfer", "delete", "process",
        "processing", "disclose", "handle", "transmit",
        # LGPD
        "colet", "tratar", "tratamento",
    ],

    "RetentionPolicy": [
        "retention", "storage limitation", "no longer than necessary",
        "delete", "deletion", "erase", "erasure",
        "retention period", "as long as necessary",
        # LGPD
        "prazo de conservação", "conservação", "armazenamento",
    ],

    "ConsentWithdrawal": [
        "withdraw", "withdrawal", "revoke", "revocation",
        "opt out", "opt-out", "unsubscribe",
        "as easy as", "as easy to withdraw",
        # LGPD
        "retirar", "revogar",
    ],

    "DataTransfer": [
        "third country", "international transfer", "cross-border transfer",
        "adequacy decision", "adequacy", "standard contractual clauses",
        "binding corporate rules", "bcr", "sccs",
        "recipient country", "transfer outside", "transferred outside",
        "transfer to a third country",
        # LGPD
        "transferência internacional", "país terceiro",
    ],

    "Right": [
        "right to access", "right of access",
        "right to erasure", "right to be forgotten",
        "right to rectification", "right to correction",
        "right to portability", "data portability",
        "right to object", "right to restriction",
        "automated decision", "automated processing",
        # CCPA
        "opt-out of sale", "right to know", "right to delete",
        # LGPD
        "direito de acesso", "direito de retificação",
        "direito à exclusão", "direito de portabilidade",
    ],

    "Purpose": [
        "purpose", "purposes", "objective",
        "specific purpose", "processing purpose", "business purpose",
        "identified purpose", "stated purpose",
        # LGPD
        "finalidade",
    ],

    "PersonalData": [
        "personal data", "personal information",
        "sensitive", "special category",
        "biometric", "health data", "financial data",
        "location data", "behavioral data",
        # LGPD
        "dados pessoais", "dados sensíveis", "informação pessoal",
    ],

    "Constraint": [
        "purpose limitation", "data minimisation", "data minimization",
        "storage limitation", "accuracy", "integrity and confidentiality",
        "security measure", "technical measure", "organisational measure",
        "safeguard", "encryption", "pseudonymisation",
        # CCPA
        "reasonable security",
    ],
}


# ── PIPEDA ────────────────────────────────────────────────────────────────────
#
# PIPEDA Schedule 1 has 10 Principles (4.1–4.10). The vocabulary is almost
# entirely different from GDPR. Key differences:
#
#   GDPR "processing"           → PIPEDA "collection, use or disclosure"
#   GDPR "data subject"         → PIPEDA "individual"
#   GDPR "personal data"        → PIPEDA "personal information"
#   GDPR "legal basis"          → PIPEDA "knowledge and consent"
#   GDPR "right to erasure"     → PIPEDA does not have this right
#   GDPR "right to portability" → PIPEDA does not have this right
#   GDPR "data transfer (SCC)"  → PIPEDA "transfer to third party
#                                          + comparable protection"
#
# Principle mapping to metamodel concepts:
#   4.1 Accountability        → Actor, Constraint (org responsibility)
#   4.2 Identifying Purposes  → Purpose
#   4.3 Consent               → LegalBasis, ConsentWithdrawal
#   4.4 Limiting Collection   → Constraint, ProcessingActivity
#   4.5 Use, Disclosure, Ret. → ProcessingActivity, RetentionPolicy
#   4.6 Accuracy              → Constraint
#   4.7 Safeguards            → Constraint
#   4.8 Openness              → (metadata — no direct metamodel concept)
#   4.9 Individual Access     → Right
#  4.10 Challenging Compliance → (metadata — no direct metamodel concept)

_PIPEDA_CONCEPT_KEYWORDS: dict[str, list[str]] = {

    "LegalBasis": [
        # Principle 4.3 — Consent is the primary legal basis in PIPEDA
        "knowledge and consent",
        "consent of the individual",
        "implied consent",
        "express consent",
        "meaningful consent",
        "without the knowledge and consent",
        "consent is required",
        "require consent",
        # Exceptions to consent requirement (alternative bases)
        "without consent",
        "reasonable person would consider appropriate",
        "legitimate business purpose",
        "collected without the knowledge",
    ],

    "ProcessingActivity": [
        # PIPEDA's canonical phrase — appears with AND without Oxford comma
        # "collection, use, or disclosure" (Oxford comma — Schedule 1 text)
        # "collection, use or disclosure" (no Oxford comma — some sections)
        "collection, use",           # matches both variants from the start
        "use, or disclosure",        # Oxford comma variant
        "use or disclosure",         # non-Oxford variant
        "use and disclosure",        # alternate conjunction
        "collect personal information",
        "use personal information",
        "disclose personal information",
        "disclosed to third parties",
        "collection of personal information",
        "handling of personal information",
        # Principle 4.4 — Limiting Collection
        "limiting collection",
        "limit the collection",
        "collected by fair and lawful means",
        "collected for purposes",
    ],

    "RetentionPolicy": [
        # Principle 4.5 — Limiting Use, Disclosure, and Retention
        "retain only as long as necessary",
        "no longer required",
        "retention of personal information",
        "retention period",
        "retention schedule",
        "destroy", "erase", "render anonymous",
        "as long as necessary for the fulfilment",
        "no longer needed",
        "minimum and maximum retention",
    ],

    "ConsentWithdrawal": [
        # Principle 4.3 — Consent withdrawal
        "withdraw consent",
        "withdrawal of consent",
        "individual may withdraw",
        "refuse or withdraw",
        "notification of the implications of withdrawing",
        "reasonable notice",
        "opt out",
        "consequences of withdrawing",
    ],

    "DataTransfer": [
        # Principle 4.1 — Accountability for third-party transfers
        # PIPEDA does not use "adequacy decision" or "SCCs" — it uses
        # "comparable level of protection" and "contractual means"
        "transfer to a third party",
        "transfer personal information",
        "comparable level of protection",
        "contractual or other means",
        "transferred outside",
        "transfer outside canada",
        "third-party organization",
        "organization to which the information is transferred",
        "accountability for transfers",
    ],

    "Right": [
        # Principle 4.9 — Individual Access
        # PIPEDA rights are limited compared to GDPR: mainly access + correction
        "individual access",
        "access to personal information",
        "access request",
        "right of access",
        "informed of the existence",
        "existence, use and disclosure",
        "challenge the accuracy",
        "correction or amendment",
        "annotation",
        "right to challenge",
        # Principle 4.10 — Challenging Compliance (right to complain)
        "challenge compliance",
        "address a complaint",
        "file a complaint",
        "privacy commissioner",
    ],

    "Purpose": [
        # Principle 4.2 — Identifying Purposes
        "identifying purposes",
        "purposes for which information is collected",
        "specified at or before the time of collection",
        "stated purpose",
        "identified purpose",
        "new purpose",
        "primary purpose",
        "secondary purpose",
        "purposes identified",
        "purpose of the collection",
        "purpose for collecting",
    ],

    "PersonalData": [
        # PIPEDA uses "personal information" exclusively — never "personal data"
        "personal information",
        "sensitive information",
        "identifiable individual",
        "about an identifiable individual",
        "information about an individual",
        # Sensitive personal information (PIPEDA does not use "special category")
        "medical information", "health information",
        "financial information",
        "ethnic origin", "racial origin",
        "religious beliefs", "political opinions",
        "biometric information",
    ],

    "Constraint": [
        # Principle 4.6 — Accuracy
        "accuracy", "as accurate, complete",
        "update personal information",
        # Principle 4.7 — Safeguards
        "security safeguards",
        "appropriate safeguards",
        "protect personal information",
        "unauthorized access",
        "loss or theft",
        "unauthorized disclosure",
        "physical measures", "organizational measures", "technological measures",
        # Principle 4.4 — Limiting Collection
        "limited to that which is necessary",
        "collected by fair and lawful means",
        "not collected indiscriminately",
        # Principle 4.8 — Openness
        "readily available",
        "make available",
    ],
}


# ── Dispatch table ────────────────────────────────────────────────────────────

# Maps law name (upper-case) to its keyword dict.
# Unknown laws fall back to the GDPR/general set.
_LAW_CONCEPT_KEYWORDS: dict[str, dict[str, list[str]]] = {
    "GDPR":   _GDPR_CONCEPT_KEYWORDS,
    "LGPD":   _GDPR_CONCEPT_KEYWORDS,   # LGPD keywords included in GDPR set
    "CCPA":   _GDPR_CONCEPT_KEYWORDS,
    "CPRA":   _GDPR_CONCEPT_KEYWORDS,
    "PIPEDA": _PIPEDA_CONCEPT_KEYWORDS,
}


def concept_tagger(text: str, law: str = "") -> list[str]:
    """
    Return a list of metamodel class names likely present in the chunk text.

    Uses case-insensitive keyword matching — fast, deterministic, no model.
    Law-aware: uses PIPEDA-specific keywords for PIPEDA chunks, GDPR-derived
    keywords for all other laws.

    Parameters
    ----------
    text : raw chunk text
    law  : canonical law name, e.g. "GDPR", "PIPEDA". Empty string falls
           back to the GDPR/general keyword set.
    """
    keyword_dict = _LAW_CONCEPT_KEYWORDS.get(law.upper(), _GDPR_CONCEPT_KEYWORDS)
    text_lower   = text.lower()
    tags = []
    for concept, keywords in keyword_dict.items():
        if any(kw in text_lower for kw in keywords):
            tags.append(concept)
    return tags


# ── Text extraction from PDF ───────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: Path, law: str = "") -> str:
    import pdfplumber
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            if law.upper() == "PIPEDA":
                # PIPEDA PDF is bilingual with English on the left half.
                # Crop to the left 55% of the page to extract English only.
                width  = page.width
                height = page.height
                english_column = page.crop((0, 0, width * 0.5, height))
                text = english_column.extract_text()
            else:
                text = page.extract_text(x_tolerance=2, y_tolerance=3)
            if text:
                pages.append(text)
    return "\n\n".join(pages)


def extract_text_from_file(path: Path, law: Law) -> str:
    """Route to correct extractor based on file extension."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(path, law)
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
    law             : canonical law name — drives regex patterns AND keyword set
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
    # Pass `law` to concept_tagger so it uses the correct keyword set.
    final: list[Chunk] = []
    for ch in all_chunks:
        ch.concept_tags = concept_tagger(ch.text, law=law)

        if len(ch.text) > max_chunk_chars:
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

    # Pass chunk.law so sub-chunks get the same law-appropriate tags
    for sc in sub_chunks:
        sc.concept_tags = concept_tagger(sc.text, law=chunk.law)

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
    law             : canonical law name, e.g. "GDPR", "CCPA", "LGPD", "PIPEDA"
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
    text = extract_text_from_file(path, law)
    return chunk_text(text, law, min_chunk_chars, max_chunk_chars)