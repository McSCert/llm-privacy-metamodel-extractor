#!/usr/bin/env python3
"""
run_pipeline.py — Privacy Policy Extraction Pipeline (PrivacyPolicyMetamodel v4)

PLACEMENT
---------
This file lives at the PROJECT ROOT, alongside the three sub-packages:

    full_artifacts_n0_rejectionloop/
    ├── gap_analysis/
    │   ├── __init__.py
    │   ├── gap_analysis.py       <- GapAnalyser
    │   └── repository.py         <- ModelRepository
    ├── privacy_schema/
    │   ├── __init__.py
    │   ├── enums.py
    │   ├── models.py             <- Pydantic models
    │   └── prompts.py            <- build_concept_prompt, build_assembler_prompt
    ├── rag_pipeline/
    │   ├── __init__.py
    │   ├── chunker.py
    │   ├── embedder.py
    │   ├── retriever.py          <- Retriever
    │   └── store.py              <- ChunkStore, ingest_file
    └── run_pipeline.py           <- THIS FILE

Run from the project root (activate your venv first):
    source .venv/bin/activate
    python run_pipeline.py --input PIPEDA=../../laws/pipeda.pdf --articles "Principle 4.1" --backend local

Quick-start examples
--------------------
Prototype on one PIPEDA principle, local Ollama:
    python run_pipeline.py \\
        --input PIPEDA=../../laws/pipeda.pdf \\
        --articles "Principle 4.1" \\
        --backend local --local-model llama3.1:8b

Full run, Anthropic API:
    python run_pipeline.py \\
        --input GDPR=../../laws/gdpr.pdf CCPA=../../laws/ccpa.txt \\
        --backend anthropic

Ingest only (no LLM calls, cheap first step):
    python run_pipeline.py --input PIPEDA=../../laws/pipeda.pdf --stage ingest

Dry-run (tests all code paths, zero API cost):
    python run_pipeline.py --input PIPEDA=../../laws/pipeda.pdf --dry-run

Gap report from an existing repository:
    python run_pipeline.py --stage analyse

Pipeline Stages
---------------
  1  INGEST     file -> chunk -> embed -> ChunkStore (SQLite)
  2  EXTRACT    retrieve per (article x concept) -> LLM Pass 1 -> validate
  3  ASSEMBLE   collect Pass-1 outputs -> LLM Pass 2 -> PolicyStatement
  4  STORE      write validated statements -> ModelRepository (SQLite)
  5  ANALYSE    GapAnalyser -> gap report

================================================================================
KNOWN ISSUES
================================================================================

[ISSUE-1] TF-IDF embedder fitted per-law on its own corpus only.
  NOT URGENT. Swap to SentenceTransformerEmbedder (multilingual-e5-base)
  when you want cross-lingual retrieval quality. Stub in rag_pipeline/embedder.py.

[ISSUE-2] PIPEDA chunker produces "principle"-level chunks.
  Already handled — the article enumeration query includes level='principle'.
  Use --articles "Principle 4.1" to target a single principle.

[ISSUE-3] LLM call volume = |laws| x |articles| x |concepts| + |articles|.
  MITIGATIONS:
    (a) --articles  filter  — target one or a few principles
    (b) concept-tag pre-filter (ON by default) — ~30-40% fewer calls
    (c) --backend local      — route calls to a free local model

[ISSUE-4] No concept extractor for "Regulation".
  Synthesised from JURISDICTION_MAP below. Add new laws to that dict.

[ISSUE-5] Pass-1 extracts ONE concept instance per article per call.
  Under-extracts multi-value concepts (Right, Purpose, Constraint).
  Increase --top-k for more context. Future fix: multi-instance prompt.

[ISSUE-6] Corrective retry prompt is minimal.
  Set --max-retries 3-4 with local models. Monitor failure rate in summary.

[ISSUE-LOCAL] Local model size guidance:
   7B  — high JSON failure rate, smoke-testing only
  13B  — acceptable for prototypes
  70B Q4 quant — close to Claude Haiku quality, suitable for research
  Recommended: llama3.1:70b-instruct-q4 or mistral-large via Ollama.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
import urllib.request
import urllib.error
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


# =============================================================================
# PATH SETUP
# Add the project root to sys.path so the three sub-packages are importable.
# run_pipeline.py is already AT the project root, so __file__'s parent IS root.
# =============================================================================

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# =============================================================================
# IMPORTS  —  clean sub-package imports, no stubs needed
# =============================================================================

# rag_pipeline sub-package
from rag_pipeline.store    import ChunkStore, ingest_file
from rag_pipeline.retriever import Retriever

# privacy_schema sub-package
from privacy_schema.prompts import (
    build_concept_prompt,
    build_assembler_prompt,
    SYSTEM_PROMPT,
)
from privacy_schema.models import (
    LegalBasisModel,
    ProcessingActivityModel,
    RetentionPolicyModel,
    ConsentWithdrawalModel,
    RightModel,
    PurposeModel,
    DataTransferModel,
    ConstraintModel,
    ActorModel,
    PolicyStatementModel,
)

# gap_analysis sub-package
from gap_analyses.repository  import ModelRepository
from gap_analyses.gap_analysis import GapAnalyser


# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")


# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
DEFAULT_LOCAL_URL       = "http://localhost:11434/v1"   # Ollama default
DEFAULT_LOCAL_MODEL     = "llama3.1:8b"
DEFAULT_TOP_K           = 3
DEFAULT_RETRIES         = 2

# Concepts extracted in Pass 1.
PASS1_CONCEPTS = [
    "LegalBasis",
    "ProcessingActivity",
    "Actor",
    "Purpose",
    "Right",
    "Constraint",
    "RetentionPolicy",
    "DataTransfer",
    "ConsentWithdrawal",
]

# These concepts produce a list in the assembler even though Pass-1
# returns one object per call. See ISSUE-5.
LIST_CONCEPTS = {
    "Purpose", "Right", "Constraint",
    "RetentionPolicy", "DataTransfer", "ConsentWithdrawal",
}

# Jurisdiction metadata for synthesising regulation objects. See ISSUE-4.
# Add your law here when you add a new input law.
JURISDICTION_MAP: dict[str, dict] = {
    "GDPR": {
        "jurisdictionId": "EU",
        "name":           "European Union",
        "description":    "EU General Data Protection Regulation",
        "source_clause":  "GDPR",
    },
    "LGPD": {
        "jurisdictionId": "BR",
        "name":           "Brazil",
        "description":    "Lei Geral de Proteção de Dados Pessoais",
        "source_clause":  "LGPD",
    },
    "CCPA": {
        "jurisdictionId": "CA-US",
        "name":           "California, United States",
        "description":    "California Consumer Privacy Act",
        "source_clause":  "CCPA",
    },
    "CPRA": {
        "jurisdictionId": "CA-US",
        "name":           "California, United States",
        "description":    "California Privacy Rights Act",
        "source_clause":  "CPRA",
    },
    "PIPEDA": {
        "jurisdictionId": "CA",
        "name":           "Canada",
        "description":    "Personal Information Protection and Electronic Documents Act",
        "source_clause":  "PIPEDA",
    },
}

# Structurally valid empty fallbacks used when all retries are exhausted.
EMPTY_FALLBACKS: dict[str, str] = {
    "LegalBasis": json.dumps({
        "basisId": "", "type": "LegalObligation",
        "evidence": "", "jurisdiction": [], "source_clause": "",
    }),
    "ProcessingActivity": json.dumps({
        "activityId": "", "description": "", "action": "Use",
        "riskAssessmentReference": None, "dataProcessed": [], "source_clause": "",
    }),
    "Actor": json.dumps({
        "actorId": "", "name": "", "role": "DataController", "source_clause": "",
    }),
    "Purpose":            "[]",
    "Right":              "[]",
    "Constraint":         "[]",
    "RetentionPolicy":    "[]",
    "DataTransfer":       "[]",
    "ConsentWithdrawal":  "[]",
}


def _strip_underscores(obj: Any) -> Any:
    """
    Recursively strip leading underscores from dict keys.
 
    Some local LLMs prefix field names with _ (e.g. "_source_clause",
    "_dataProcessed"). This breaks Pydantic alias matching because the
    models use camelCase aliases ("source_clause", "dataProcessed").
 
    Only strips a SINGLE leading underscore. Double-underscore keys
    ("__something") are left unchanged.
 
    Called in _assemble_one_statement() after json.loads() and before
    PolicyStatementModel.model_validate().
 
    Parameters
    ----------
    obj : the parsed JSON value — dict, list, or scalar.
          json.loads() always returns one of these three types.
 
    Returns
    -------
    The same structure with underscore-prefixed keys renamed.
    """
    if isinstance(obj, dict):
        return {
            (k[1:] if k.startswith("_") and not k.startswith("__") else k):
            _strip_underscores(v)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_strip_underscores(item) for item in obj]
    return obj   # scalar (str, int, float, bool, None) — unchanged


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ExtractionResult:
    law:      str
    article:  str
    concept:  str
    success:  bool
    json_str: str
    errors:   list[str] = field(default_factory=list)
    attempts: int = 0


@dataclass
class PipelineStats:
    laws_ingested:      list[str] = field(default_factory=list)
    articles_processed: int = 0
    articles_skipped:   int = 0
    pass1_attempts:     int = 0
    pass1_success:      int = 0
    pass1_absent:       int = 0   # concept genuinely not in text (tagger false positive)
    pass1_failed:       int = 0   # real model failure — wrong enum, missing required field
    pass2_attempts:     int = 0
    pass2_success:      int = 0
    pass2_failed:       int = 0
    statements_stored:  int = 0
    api_calls:          int = 0
    tokens_in:          int = 0
    tokens_out:         int = 0

    def log_summary(self) -> None:
        b = "=" * 60
        log.info(b)
        log.info("PIPELINE SUMMARY")
        log.info(b)
        log.info(f"  Laws ingested        : {self.laws_ingested}")
        log.info(
            f"  Articles processed   : {self.articles_processed}  "
            f"(skipped by filter: {self.articles_skipped})"
        )
        log.info(
            f"  Pass-1 calls         : {self.pass1_attempts}  "
            f"(ok={self.pass1_success}  "
            f"absent={self.pass1_absent}  "
            f"failed={self.pass1_failed})"
        )
        # Failure rate excludes absent — those are not real failures
        real_attempts = self.pass1_attempts - self.pass1_absent
        if real_attempts > 0:
            fail_pct = 100 * self.pass1_failed / real_attempts
            log.info(
                f"  Pass-1 failure rate  : {fail_pct:.1f}%  "
                f"(excludes {self.pass1_absent} absent-concept skips)"
                + ("  ← model may be too small (ISSUE-LOCAL)" if fail_pct > 20 else "")
            )
        log.info(
            f"  Pass-2 calls         : {self.pass2_attempts}  "
            f"(ok={self.pass2_success}  failed={self.pass2_failed})"
        )
        log.info(f"  Statements stored    : {self.statements_stored}")
        log.info(f"  Total LLM calls      : {self.api_calls}")
        if self.tokens_in or self.tokens_out:
            log.info(f"  Tokens  in / out     : {self.tokens_in} / {self.tokens_out}")
        log.info(b)


# =============================================================================
# LLM BACKENDS
# =============================================================================

class LLMBackend(ABC):
    """
    Abstract interface shared by all LLM backends.
    The pipeline only calls .call(system, user, stats) — it does not care
    which backend is active.
    """

    @abstractmethod
    def call(
        self,
        system:     str,
        user:       str,
        stats:      PipelineStats,
        max_tokens: int = 2048,
    ) -> str: ...

    @staticmethod
    def _strip_fences(raw: str) -> str:
        """Strip accidental markdown code fences from model output."""
        raw = raw.strip()
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1].lstrip("json").strip()
        return raw.strip()


# ── Anthropic ─────────────────────────────────────────────────────────────────

class AnthropicBackend(LLMBackend):
    """
    Calls the Anthropic Messages API.
    Reads ANTHROPIC_API_KEY from the environment.
    Applies exponential back-off on 429 rate-limit errors.
    """

    def __init__(self, model: str):
        try:
            import anthropic as _a
            self._anthropic = _a
            self._client    = _a.Anthropic()
        except ImportError:
            log.error("anthropic SDK not installed.  pip install anthropic")
            sys.exit(1)
        self.model = model
        log.info(f"Backend: Anthropic API  (model={model})")

    def call(
        self,
        system:     str,
        user:       str,
        stats:      PipelineStats,
        max_tokens: int = 2048,
    ) -> str:
        for attempt in range(4):
            try:
                resp = self._client.messages.create(
                    model      = self.model,
                    max_tokens = max_tokens,
                    system     = system,
                    messages   = [{"role": "user", "content": user}],
                )
                stats.api_calls  += 1
                stats.tokens_in  += resp.usage.input_tokens
                stats.tokens_out += resp.usage.output_tokens
                raw = " ".join(b.text for b in resp.content if b.type == "text")
                return self._strip_fences(raw)

            except self._anthropic.RateLimitError:
                wait = 2 ** attempt
                log.warning(f"Rate limit — waiting {wait}s (retry {attempt+1}/3)")
                time.sleep(wait)

            except self._anthropic.APIError as exc:
                log.error(f"Anthropic API error: {exc}")
                raise

        raise RuntimeError("Exhausted Anthropic rate-limit retries.")


# ── Local OpenAI-compatible ───────────────────────────────────────────────────

class LocalBackend(LLMBackend):
    """
    Calls any OpenAI-compatible /v1/chat/completions endpoint via urllib
    (no extra dependency).

    Compatible runners and their default base URLs:
      Ollama      http://localhost:11434/v1    (ollama serve)
      llama.cpp   http://localhost:8080/v1     (./server -m model.gguf)
      LM Studio   http://localhost:1234/v1     (local server tab)
      vLLM        http://localhost:8000/v1     (python -m vllm.entrypoints.openai.api_server)

    See ISSUE-LOCAL for model size guidance.
    """

    def __init__(self, base_url: str, model: str):
        self.model     = model
        self._endpoint = base_url.rstrip("/") + "/chat/completions"
        self._models_url = base_url.rstrip("/") + "/models"
        self._check_connection()
        log.info(f"Backend: Local LLM  (endpoint={self._endpoint}  model={model})")
        log.warning(
            "Local backend active — monitor Pass-1 failure rate. "
            "See ISSUE-LOCAL for model size guidance."
        )

    def _check_connection(self) -> None:
        """Verify the server is reachable and the model is available at startup."""
        try:
            req = urllib.request.Request(
                self._models_url,
                headers={"Accept": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                data      = json.loads(resp.read())
                available = [m.get("id", "") for m in data.get("data", [])]
                if available and self.model not in available:
                    log.warning(
                        f"Model '{self.model}' not in server list: {available}. "
                        f"Check --local-model. Continuing anyway."
                    )
                elif available:
                    log.info(f"  Model '{self.model}' confirmed available.")
        except urllib.error.URLError as exc:
            log.error(
                f"Cannot reach local LLM server at {self._models_url}.\n"
                f"  Is the server running?\n"
                f"  Ollama:    ollama serve  &&  ollama pull {self.model}\n"
                f"  llama.cpp: ./server -m your_model.gguf\n"
                f"  Error: {exc}"
            )
            sys.exit(1)
        except Exception:
            log.debug("Models endpoint not available — skipping model check.")

    def call(
        self,
        system:     str,
        user:       str,
        stats:      PipelineStats,
        max_tokens: int = 2048,
    ) -> str:
        payload = json.dumps({
            "model":       self.model,
            "max_tokens":  max_tokens,
            "temperature": 0.0,        # deterministic output — critical for JSON
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        }).encode("utf-8")

        req = urllib.request.Request(
            self._endpoint,
            data    = payload,
            headers = {"Content-Type": "application/json", "Accept": "application/json"},
            method  = "POST",
        )

        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=300) as resp:
                    data = json.loads(resp.read())

                stats.api_calls  += 1
                usage             = data.get("usage", {})
                stats.tokens_in  += usage.get("prompt_tokens",    0)
                stats.tokens_out += usage.get("completion_tokens", 0)

                raw = data["choices"][0]["message"]["content"]
                return self._strip_fences(raw)

            except urllib.error.HTTPError as exc:
                if exc.code == 503 and attempt < 3:
                    wait = 5 * (attempt + 1)
                    log.warning(
                        f"Server 503 (model loading?) — waiting {wait}s "
                        f"(attempt {attempt+1}/3)"
                    )
                    time.sleep(wait)
                else:
                    body = exc.read().decode("utf-8", errors="replace")
                    log.error(f"Local LLM HTTP {exc.code}: {body[:300]}")
                    raise

            except urllib.error.URLError as exc:
                log.error(f"Local LLM connection error: {exc}")
                raise

            except KeyError:
                log.error(
                    f"Unexpected response from local LLM. "
                    f"Raw: {str(data)[:300]}"
                )
                raise

        raise RuntimeError("Exhausted local LLM retries.")


# ── Dry-run ───────────────────────────────────────────────────────────────────

class DryRunBackend(LLMBackend):
    """Returns '{}' without any network call. Used with --dry-run."""

    def call(self, system: str, user: str, stats: PipelineStats,
             max_tokens: int = 2048) -> str:
        stats.api_calls += 1
        log.debug("[DRY RUN] skipping LLM call — returning '{}'")
        return "{}"


# =============================================================================
# ARTICLE FILTER
# =============================================================================

def _build_article_filter(articles_arg: Optional[str]) -> Optional[re.Pattern]:
    """
    Build a compiled regex from the --articles argument.

    --articles accepts a comma-separated list of substrings matched
    case-insensitively against article_ref. Each term is treated as a
    literal string (dots, parentheses, etc. are not regex metacharacters).

    Examples:
      --articles "Principle 4.1"         matches PIPEDA Principle 4.1 only
      --articles "4.1,4.2,4.3"           matches three PIPEDA principles
      --articles "Art.6,Art.7"           matches GDPR Art.6 and Art.7
      --articles "Art.6(1)(a)"           matches that specific GDPR sub-article

    Returns None when --articles is not set (all articles are processed).
    """
    if not articles_arg:
        return None
    terms = [t.strip() for t in articles_arg.split(",") if t.strip()]
    if not terms:
        return None
    pattern  = "|".join(re.escape(t) for t in terms)
    compiled = re.compile(pattern, re.IGNORECASE)
    log.info(f"Article filter active: '{articles_arg}'")
    return compiled


def _article_passes_filter(article_ref: str, filt: Optional[re.Pattern]) -> bool:
    if filt is None:
        return True
    return bool(filt.search(article_ref))


# =============================================================================
# STAGE 1 — INGEST
# =============================================================================

def stage_ingest(
    law_files: dict[str, Path],
    db_path:   Path,
    data_dir:  Path,
    stats:     PipelineStats,
) -> dict[str, Path]:
    """
    Chunk, embed, and store each law file into the ChunkStore.

    Delegates to rag_pipeline.store.ingest_file() which:
      1. Extracts text — PDF via pdfplumber, or plain .txt.
      2. Hierarchically chunks: chapter/schedule -> article/principle -> clause.
         PIPEDA -> schedule -> principle -> clause  (level="principle")
         GDPR   -> chapter  -> article   -> clause  (level="article")
      3. Fits a TF-IDF embedder on the chunk corpus (one model per law).
      4. Embeds and persists chunks to the SQLite ChunkStore.

    Returns {LAW: embedder_pickle_path} for use by the Retriever.
    """
    log.info("=" * 60)
    log.info("STAGE 1 — INGEST")
    log.info("=" * 60)

    data_dir.mkdir(parents=True, exist_ok=True)
    embedder_paths: dict[str, Path] = {}

    for law, file_path in law_files.items():
        if not file_path.exists():
            log.error(f"Input file not found: {file_path} — skipping {law}")
            continue

        emb_path = data_dir / f"chunks_{law.upper()}_embedder.pkl"

        try:
            result = ingest_file(
                path          = file_path,
                law           = law,
                db_path       = db_path,
                embedder_path = emb_path,
            )
        except Exception as exc:
            log.error(f"Ingest failed for {law}: {exc}")
            continue

        embedder_paths[law.upper()] = emb_path
        stats.laws_ingested.append(law.upper())
        log.info(
            f"  {law}: {result['chunks_produced']} chunks produced, "
            f"{result['chunks_written']} written  [{db_path.name}]"
        )

    return embedder_paths


# =============================================================================
# STAGE 2 — EXTRACT  (Pass 1: one concept per article per LLM call)
# =============================================================================

def _corrective_prompt(
    concept:     str,
    law:         str,
    article_ref: str,
    rag_text:    str,
    bad_json:    str,
    errors:      list[str],
) -> tuple[str, str]:
    """
    Rebuild the concept prompt with the failed output and Pydantic validation
    errors prepended. Asks the LLM to fix only the failing fields.
    Critical for local models which have higher JSON failure rates (ISSUE-6).
    """
    error_block = "\n".join(f"  - {e}" for e in errors)
    prefix = (
        "## CORRECTION NEEDED\n\n"
        "Your previous output failed schema validation:\n"
        f"{error_block}\n\n"
        "Fix ONLY the fields listed above. Keep all other fields unchanged.\n"
        "Return the complete corrected JSON — no markdown, no explanation.\n\n"
        "PREVIOUS (INVALID) OUTPUT:\n"
        f"{bad_json}\n\n"
        "──────────────────────────────────────────────────────────────────\n\n"
    )
    system, base_user = build_concept_prompt(concept, law, article_ref, rag_text)
    return system, prefix + base_user


# Signal fields per concept: if ALL of these are empty/null/missing in the
# LLM's response, the concept is genuinely absent from the text rather than
# being a model failure. This handles tagger false-positives gracefully.
#
# Rationale per concept:
#   LegalBasis         — "evidence" must be quoted/paraphrased text; empty = not found
#   ProcessingActivity — "description" summarises the activity; empty = nothing described
#   Actor              — "name" is the entity name; empty = no actor mentioned
#   Purpose            — "description" is the purpose text; empty = no purpose found
#   Right              — "type" is an enum; if LLM cannot pick one, concept is absent
#   Constraint         — "expression" is the constraint text; empty = none found
#   RetentionPolicy    — "duration" is the time value; 0 or missing = not specified
#   DataTransfer       — "mechanism" is a required enum; missing = not a transfer article
#   ConsentWithdrawal  — "channel" is a required enum; missing = not a withdrawal article
_ABSENCE_SIGNAL_FIELDS: dict[str, list[str]] = {
    "LegalBasis":         ["evidence"],
    "ProcessingActivity": ["description"],
    "Actor":              ["name"],
    "Purpose":            ["description"],
    "Right":              ["type"],
    "Constraint":         ["expression"],
    "RetentionPolicy":    ["duration"],
    "DataTransfer":       ["mechanism"],
    "ConsentWithdrawal":  ["channel"],
}


def _is_concept_absent(concept: str, parsed: dict) -> bool:
    """
    Return True when the LLM response indicates the concept is genuinely
    not present in the text — as opposed to a real extraction failure.

    A concept is considered absent when ALL its signal fields are either
    missing from the parsed dict, empty string, zero, or null.

    This prevents tagger false-positives from burning retries and inflating
    the failure rate. An absent concept is immediately returned as an empty
    fallback WITHOUT retrying — retrying would waste tokens on text that
    simply does not contain the concept.

    Examples that return True (absent):
      DataTransfer:       {} or {"mechanism": "", "destinationJurisdiction": ""}
      RetentionPolicy:    {"duration": 0} or {"duration": None}
      Right:              {} or {"type": ""}

    Examples that return False (real failure — should retry):
      DataTransfer:       {"mechanism": "INVALID_VALUE"}   <- wrong enum, retry
      LegalBasis:         {"type": "BadType"}              <- wrong enum, retry
    """
    signal_fields = _ABSENCE_SIGNAL_FIELDS.get(concept, [])
    if not signal_fields:
        return False   # unknown concept — do not suppress

    for field_name in signal_fields:
        value = parsed.get(field_name)
        # If any signal field has a non-empty value, the concept IS present
        # (even if the value turns out to be wrong — that's a real failure)
        if value is not None and value != "" and value != 0 and value != [] and value != {}:
            return False

    # All signal fields are empty/missing — concept is absent
    return True


def _extract_one_concept(
    concept:     str,
    law:         str,
    article_ref: str,
    rag_text:    str,
    backend:     LLMBackend,
    stats:       PipelineStats,
    max_retries: int,
    validators:  dict[str, Any],
) -> ExtractionResult:
    """
    Extract one metamodel concept from one article via the LLM.

    Flow per attempt:
      1. Build (or rebuild with correction) the concept prompt.
      2. Call the LLM backend.
      3. JSON-parse the response.
      4. Validate with the Pydantic model.
      5. On failure: increment attempt counter and rebuild corrective prompt.
      6. After max_retries exhausted: return structurally valid empty fallback.
    """
    validator    = validators[concept]
    system, user = build_concept_prompt(concept, law, article_ref, rag_text)
    last_raw     = ""
    last_errors: list[str] = []

    for attempt in range(1, max_retries + 2):   # initial + max_retries retries
        stats.pass1_attempts += 1

        if attempt > 1:
            system, user = _corrective_prompt(
                concept, law, article_ref, rag_text, last_raw, last_errors
            )

        last_raw = backend.call(system, user, stats)

        # ── JSON parse ────────────────────────────────────────────────────────
        try:
            parsed = json.loads(last_raw)
        except json.JSONDecodeError as exc:
            last_errors = [f"JSON parse error: {exc}"]
            log.debug(f"    JSON error {concept}@{article_ref}: {exc}")
            continue

        # ── Absence check ─────────────────────────────────────────────────────
        # If the LLM returned an empty/minimal object it means the concept is
        # genuinely not in this text (tagger false-positive). Do not retry —
        # retrying would produce the same result and waste tokens.
        if _is_concept_absent(concept, parsed):
            stats.pass1_absent += 1
            log.debug(
                f"    {concept}@{article_ref}: concept absent in text "
                f"(tagger false-positive) — skipping"
            )
            return ExtractionResult(
                law=law, article=article_ref, concept=concept,
                success=False,
                json_str=EMPTY_FALLBACKS.get(concept, "{}"),
                errors=["concept absent in text"],
                attempts=attempt,
            )

        # ── Pydantic validation ────────────────────────────────────────────────
        try:
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                validator.model_validate(parsed)

            stats.pass1_success += 1
            return ExtractionResult(
                law=law, article=article_ref, concept=concept,
                success=True, json_str=last_raw, attempts=attempt,
            )

        except Exception as exc:
            raw_errors = getattr(exc, "errors", None)
            if callable(raw_errors):
                last_errors = [
                    "{}: {}".format(
                        ".".join(str(loc) for loc in e.get("loc", [])),
                        e.get("msg", ""),
                    )
                    for e in raw_errors()
                ]
            else:
                last_errors = [str(exc)]
            log.debug(
                f"    Validation error {concept}@{article_ref}: {last_errors[:2]}"
            )

    # ── All retries exhausted ─────────────────────────────────────────────────
    stats.pass1_failed += 1
    log.warning(
        f"  REJECTED {concept}@{law}/{article_ref} after {attempt} attempt(s). "
        f"Errors: {last_errors[:2]}"
    )
    return ExtractionResult(
        law=law, article=article_ref, concept=concept,
        success=False,
        json_str=EMPTY_FALLBACKS.get(concept, "{}"),
        errors=last_errors,
        attempts=attempt,
    )


def stage_extract(
    db_path:          Path,
    embedder_paths:   dict[str, Path],
    backend:          LLMBackend,
    stats:            PipelineStats,
    top_k:            int,
    max_retries:      int,
    article_filter:   Optional[re.Pattern],
    use_concept_tags: bool = True,
) -> dict[str, list[dict]]:
    """
    Pass 1: for every (law, article, concept) triple that passes the filters,
    retrieve RAG context and run one LLM extraction call.

    Article levels handled:
      "article"   — GDPR, LGPD, CCPA
      "section"   — CCPA variant
      "principle" — PIPEDA
      "document"  — fallback when no structure detected

    Returns:
        {
          "PIPEDA": [
            { "article_ref":  "Principle 4.1 — Accountability",
              "concepts": {
                "LegalBasis":         "{...}",
                "ProcessingActivity": "{...}",
                ...
              }
            }
          ]
        }
    """
    log.info("=" * 60)
    log.info("STAGE 2 — EXTRACT (Pass 1)")
    log.info("=" * 60)
    log.info(f"  Concept-tag filter : {'ON' if use_concept_tags else 'OFF'}")
    if article_filter:
        log.info(f"  Article filter     : {article_filter.pattern}")

    validators: dict[str, Any] = {
        "LegalBasis":         LegalBasisModel,
        "ProcessingActivity": ProcessingActivityModel,
        "Actor":              ActorModel,
        "Purpose":            PurposeModel,
        "Right":              RightModel,
        "Constraint":         ConstraintModel,
        "RetentionPolicy":    RetentionPolicyModel,
        "DataTransfer":       DataTransferModel,
        "ConsentWithdrawal":  ConsentWithdrawalModel,
    }

    results: dict[str, list[dict]] = {}

    with ChunkStore(db_path) as store:
        available_laws = store.laws()
        log.info(f"  Laws in ChunkStore : {available_laws}")

        for law in available_laws:
            if law not in embedder_paths:
                log.warning(
                    f"  No embedder found for {law} — skipping. "
                    f"Run --stage ingest first."
                )
                continue

            # ── Enumerate structural chunks ───────────────────────────────────
            # "principle" covers PIPEDA; "article"/"section" covers GDPR/CCPA.
            # We work at this level (not clause) to avoid duplicate extractions.
            cur = store._conn.execute(
                """
                SELECT DISTINCT article_ref, concept_tags
                FROM   chunks
                WHERE  law   = ?
                  AND  level IN ('article', 'section', 'principle', 'document')
                ORDER  BY article_ref
                """,
                (law,),
            )
            all_articles = [
                {
                    "article_ref":  row["article_ref"],
                    "concept_tags": set(json.loads(row["concept_tags"])),
                }
                for row in cur.fetchall()
            ]

            # Fallback if chunker produced no article/principle-level chunks
            if not all_articles:
                log.warning(
                    f"  No article/principle-level chunks for {law}. "
                    f"Falling back to all levels — extraction may be coarse."
                )
                cur = store._conn.execute(
                    "SELECT DISTINCT article_ref, concept_tags "
                    "FROM chunks WHERE law=? ORDER BY article_ref LIMIT 100",
                    (law,),
                )
                all_articles = [
                    {
                        "article_ref":  row["article_ref"],
                        "concept_tags": set(json.loads(row["concept_tags"])),
                    }
                    for row in cur.fetchall()
                ]

            # ── Apply article filter ──────────────────────────────────────────
            to_process = []
            for art in all_articles:
                if _article_passes_filter(art["article_ref"], article_filter):
                    to_process.append(art)
                else:
                    stats.articles_skipped += 1

            if not to_process:
                log.warning(
                    f"  Article filter matched 0 articles for {law}.\n"
                    f"  Available article_refs (first 20):\n"
                    + "\n".join(
                        f"    '{a['article_ref']}'"
                        for a in all_articles[:20]
                    )
                    + ("\n    ..." if len(all_articles) > 20 else "")
                )
                continue

            log.info(
                f"  {law}: processing {len(to_process)} / {len(all_articles)} article(s)"
            )

            retriever   = Retriever(db_path, {law: embedder_paths[law]})
            law_results = []

            for art in to_process:
                stats.articles_processed += 1
                article_ref  = art["article_ref"]
                article_tags = art["concept_tags"]
                art_record   = {"article_ref": article_ref, "concepts": {}}

                log.info(f"  [{law}] {article_ref}")

                for concept in PASS1_CONCEPTS:

                    # ── Concept-tag pre-filter ────────────────────────────────
                    if use_concept_tags and concept not in article_tags:
                        art_record["concepts"][concept] = (
                            EMPTY_FALLBACKS.get(concept, "{}")
                        )
                        continue

                    # ── RAG retrieval ─────────────────────────────────────────
                    rag_text = retriever.retrieve_for_prompt(
                        concept=concept, law=law, top_k=top_k,
                    )
                    if rag_text.startswith("[No relevant"):
                        log.debug(
                            f"    No RAG chunks for {concept}@{law}/{article_ref}"
                        )
                        art_record["concepts"][concept] = (
                            EMPTY_FALLBACKS.get(concept, "{}")
                        )
                        continue

                    # ── LLM extraction + Pydantic validation ──────────────────
                    result = _extract_one_concept(
                        concept, law, article_ref, rag_text,
                        backend, stats, max_retries, validators,
                    )
                    art_record["concepts"][concept] = result.json_str

                    log.info(
                        f"    {concept:22s} {'OK' if result.success else 'FAIL':4s} "
                        f"(attempts={result.attempts})"
                    )

                law_results.append(art_record)

            retriever.close()   # close once after ALL articles for this law
            results[law] = law_results

    return results


# =============================================================================
# STAGE 3 — ASSEMBLE (Pass 2)  +  STAGE 4 — STORE
# =============================================================================

def _synthesise_regulation_json(law: str, article_ref: str) -> str:
    """Build a minimal regulation object from static metadata. See ISSUE-4."""
    jur = JURISDICTION_MAP.get(law.upper())
    if jur is None:
        log.warning(
            f"  {law} not in JURISDICTION_MAP — using generic entry. "
            f"Add it to JURISDICTION_MAP in this script (ISSUE-4)."
        )
        jur = {
            "jurisdictionId": law.upper(), "name": law,
            "description": "", "source_clause": law,
        }
    return json.dumps([{
        "regulationId": "", "name": law.upper(), "version": "",
        "description":  jur.get("description", ""),
        "jurisdiction": [jur],
        "source_clause": article_ref,
    }])


def _wrap_for_assembler(concept: str, json_str: str) -> str:
    """
    Pass-1 returns one JSON object per call. The assembler expects a list
    for multi-instance concepts. Wrap as needed. See ISSUE-5.
    """
    if concept not in LIST_CONCEPTS:
        return json_str   # single-object: Actor, LegalBasis, ProcessingActivity

    if json_str.strip().startswith("["):
        return json_str   # already a list (or "[]" fallback)

    try:
        obj = json.loads(json_str)
        return json.dumps([obj]) if obj else "[]"
    except json.JSONDecodeError:
        return "[]"


def _assemble_one_statement(
    law:         str,
    article_ref: str,
    concepts:    dict[str, str],
    backend:     LLMBackend,
    stats:       PipelineStats,
    max_retries: int,
) -> Optional[dict]:
    """
    Pass 2: compose all Pass-1 outputs into a single PolicyStatement.

    The LLM is NOT re-reading the legal text. It receives the already-
    validated concept JSONs and assembles them, applying cross-concept
    OCL consistency checks (e.g. Consent without ConsentWithdrawal).

    OCL warnings are logged but do not block storage.
    Returns None if all retries fail — the article is then not stored.
    """
    stats.pass2_attempts += 1

    def _get(concept: str) -> str:
        raw = concepts.get(concept, EMPTY_FALLBACKS.get(concept, "{}"))
        return _wrap_for_assembler(concept, raw)

    system, user = build_assembler_prompt(
        actor_json               = _get("Actor"),
        purposes_json            = _get("Purpose"),
        processing_activity_json = _get("ProcessingActivity"),
        legal_basis_json         = _get("LegalBasis"),
        regulations_json         = _synthesise_regulation_json(law, article_ref),
        constraints_json         = _get("Constraint"),
        rights_json              = _get("Right"),
        source_clause            = article_ref,
        retention_json           = _get("RetentionPolicy"),
        transfers_json           = _get("DataTransfer"),
        withdrawal_json          = _get("ConsentWithdrawal"),
    )

    # Strengthen the system prompt for local models that add wrappers or
    # underscore prefixes. Appended here rather than in prompts.py so it
    # only affects Pass-2 and does not change Pass-1 behaviour.
    system = system + (
        "\n\nCRITICAL OUTPUT RULES FOR THIS CALL:\n"
        "- Return JSON fields DIRECTLY at the top level — no wrapper key.\n"
        "- WRONG:  {\"PolicyStatement\": {\"statementId\": \"...\"}}\n"
        "- CORRECT: {\"statementId\": \"...\"}\n"
        "- Do NOT prefix field names with underscore.\n"
        "- WRONG:  \"_source_clause\", \"_dataProcessed\"\n"
        "- CORRECT: \"source_clause\", \"dataProcessed\"\n"
        "- Copy field names EXACTLY as shown in the schema.\n"
    )

    last_raw     = ""
    last_errors: list[str] = []

    for attempt in range(1, max_retries + 2):
        if attempt > 1:
            prefix = (
                "## CORRECTION NEEDED — the PolicyStatement failed validation:\n\n"
                + "\n".join(f"  - {e}" for e in last_errors)
                + "\n\nPREVIOUS (INVALID) OUTPUT:\n"
                + last_raw
                + "\n\nReturn ONLY the corrected PolicyStatement JSON.\n\n"
                "──────────────────────────────────────────────────────────────\n\n"
            )
            user = prefix + user

        last_raw = backend.call(system, user, stats, max_tokens=4096)
        #print("ASSEMBLER RAW OUTPUT:", last_raw[:500])  
        try:
            parsed = json.loads(last_raw)
        except json.JSONDecodeError as exc:
            last_errors = [f"JSON parse error: {exc}"]
            continue
        
        # ── Fix 1: unwrap single-key wrapper ──────────────────────────────────
        # Local models sometimes return {"PolicyStatement": {...}} instead of
        # the object directly. Unwrap any single-key dict whose value is a dict.
        if isinstance(parsed, dict) and len(parsed) == 1:
            sole_key = next(iter(parsed))
            sole_val = parsed[sole_key]
            if isinstance(sole_val, dict):
                log.debug(f"    Unwrapping assembler response from key '{sole_key}'")
                parsed = sole_val
 
        # ── Fix 2: strip leading underscores from field names ─────────────────
        # Local models sometimes prefix field names with _ (e.g. "_source_clause").
        # _strip_underscores is defined at module level below EMPTY_FALLBACKS.
 
        parsed = _strip_underscores(parsed)

        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                validated = PolicyStatementModel.model_validate(parsed)

            for w in caught:
                log.warning(f"  OCL warning @ {law}/{article_ref}: {w.message}")

            stats.pass2_success += 1
            return validated.model_dump(by_alias=True)

        except Exception as exc:
            raw_errors = getattr(exc, "errors", None)
            if callable(raw_errors):
                last_errors = [
                    "{}: {}".format(
                        ".".join(str(loc) for loc in e.get("loc", [])),
                        e.get("msg", ""),
                    )
                    for e in raw_errors()
                ]
            else:
                last_errors = [str(exc)]
            log.warning(
                f"  Assembly error @ {law}/{article_ref}: {last_errors[:2]}"
            )

    stats.pass2_failed += 1
    log.error(
        f"  ASSEMBLY FAILED @ {law}/{article_ref} after {attempt} attempt(s). "
        f"Last errors: {last_errors}"
    )
    return None


def stage_assemble_and_store(
    extraction_results: dict[str, list[dict]],
    repo_path:          Path,
    backend:            LLMBackend,
    stats:              PipelineStats,
    max_retries:        int,
) -> None:
    log.info("=" * 60)
    log.info("STAGE 3 — ASSEMBLE (Pass 2)  +  STAGE 4 — STORE")
    log.info("=" * 60)

    with ModelRepository(repo_path) as repo:
        for law, articles in extraction_results.items():
            log.info(f"  {law}: assembling {len(articles)} article(s)")
            for art in articles:
                article_ref = art["article_ref"]
                statement   = _assemble_one_statement(
                    law, article_ref, art["concepts"],
                    backend, stats, max_retries,
                )
                if statement is not None:
                    stmt_id = repo.store(law, article_ref, statement)
                    stats.statements_stored += 1
                    log.info(f"  Stored [{law}] {article_ref} -> {stmt_id}")
                else:
                    log.warning(
                        f"  Skipped [{law}] {article_ref} — assembly failed"
                    )


# =============================================================================
# STAGE 5 — GAP ANALYSIS
# =============================================================================

def stage_analyse(
    repo_path:   Path,
    report_path: Optional[Path],
) -> str:
    log.info("=" * 60)
    log.info("STAGE 5 — GAP ANALYSIS")
    log.info("=" * 60)

    with ModelRepository(repo_path) as repo:
        repo_stats = repo.stats()
        if repo_stats["total"] == 0:
            msg = (
                "ModelRepository is empty — nothing to analyse.\n"
                "Run --stage all or --stage extract first."
            )
            log.warning(msg)
            return msg

        log.info(f"  Repository: {repo_stats['total']} total statements")
        for law, n in repo_stats["by_law"].items():
            log.info(f"    {law}: {n} statements")

        analyser = GapAnalyser(repo)
        report   = analyser.full_report()

    print("\n" + report)

    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding="utf-8")
        log.info(f"  Report written to {report_path}")

    return report


# =============================================================================
# CLI
# =============================================================================

def _parse_law_files(inputs: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for item in inputs:
        if "=" not in item:
            log.error(f"Bad --input item '{item}'. Expected LAW=path  e.g. PIPEDA=data/pipeda.pdf")
            continue
        law, path_str = item.split("=", 1)
        result[law.upper()] = Path(path_str)
    return result


def _discover_embedder_paths(db_path: Path, data_dir: Path) -> dict[str, Path]:
    """Reconstruct {LAW: embedder_path} from disk after a prior --stage ingest."""
    paths: dict[str, Path] = {}
    if not db_path.exists():
        return paths
    try:
        with ChunkStore(db_path) as store:
            for law in store.laws():
                p = data_dir / f"chunks_{law}_embedder.pkl"
                if p.exists():
                    paths[law] = p
                else:
                    log.warning(f"Embedder pickle missing for {law}: {p}")
    except Exception as exc:
        log.error(f"Could not open ChunkStore at {db_path}: {exc}")
    return paths


def _make_backend(args: argparse.Namespace) -> LLMBackend:
    if args.dry_run:
        log.info("Backend: Dry-run (no network calls)")
        return DryRunBackend()
    if args.backend == "local":
        return LocalBackend(base_url=args.local_url, model=args.local_model)
    return AnthropicBackend(model=args.model)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_pipeline.py",
        description="Privacy Policy Extraction Pipeline (PrivacyPolicyMetamodel v4)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Prototype on one PIPEDA principle (local Ollama):\n"
            "  python run_pipeline.py \\\n"
            "    --input PIPEDA=data/pipeda.pdf \\\n"
            "    --articles 'Principle 4.1' \\\n"
            "    --backend local --local-model llama3.1:8b\n\n"
            "Full run (Anthropic):\n"
            "  python run_pipeline.py \\\n"
            "    --input GDPR=data/gdpr.pdf CCPA=data/ccpa.txt\n\n"
            "Ingest only:\n"
            "  python run_pipeline.py --input PIPEDA=data/pipeda.pdf --stage ingest\n\n"
            "Dry-run (zero API cost, tests all code paths):\n"
            "  python run_pipeline.py --input PIPEDA=data/pipeda.pdf --dry-run\n\n"
            "Gap report from existing repo:\n"
            "  python run_pipeline.py --stage analyse\n"
        ),
    )

    io = p.add_argument_group("Input / Output")
    io.add_argument(
        "--input", nargs="+", metavar="LAW=PATH",
        help="Legal text files as LAW=PATH pairs, e.g. PIPEDA=data/pipeda.pdf",
    )
    io.add_argument("--db",     default="data/chunks.db",    metavar="PATH",
                    help="SQLite ChunkStore path (default: data/chunks.db)")
    io.add_argument("--repo",   default="data/model_repo.db", metavar="PATH",
                    help="SQLite ModelRepository path (default: data/model_repo.db)")
    io.add_argument("--report", default="data/gap_report.txt", metavar="PATH",
                    help="Gap analysis report output (default: data/gap_report.txt)")
    io.add_argument("--data-dir", default="data", metavar="DIR",
                    help="Directory for TF-IDF embedder pickles (default: data/)")

    be = p.add_argument_group("LLM Backend")
    be.add_argument(
        "--backend", choices=["anthropic", "local"], default="anthropic",
        help=(
            "LLM backend (default: anthropic). "
            "'local' uses any OpenAI-compatible server "
            "(Ollama / llama.cpp / LM Studio / vLLM)."
        ),
    )
    be.add_argument(
        "--model", default=DEFAULT_ANTHROPIC_MODEL, metavar="MODEL",
        help=f"Anthropic model string (default: {DEFAULT_ANTHROPIC_MODEL})",
    )
    be.add_argument(
        "--local-url", default=DEFAULT_LOCAL_URL, metavar="URL",
        help=(
            f"Base URL of the local server (default: {DEFAULT_LOCAL_URL} [Ollama]). "
            "llama.cpp: http://localhost:8080/v1  "
            "LM Studio: http://localhost:1234/v1"
        ),
    )
    be.add_argument(
        "--local-model", default=DEFAULT_LOCAL_MODEL, metavar="MODEL",
        help=(
            f"Model name on the local server (default: {DEFAULT_LOCAL_MODEL}). "
            "Recommended for research quality: llama3.1:70b-instruct-q4"
        ),
    )

    ex = p.add_argument_group("Extraction Control")
    ex.add_argument(
        "--articles", metavar="FILTER",
        help=(
            "Comma-separated substrings matched against article_ref (case-insensitive). "
            "Only matching articles are processed. "
            "Examples: "
            "'Principle 4.1'  (single PIPEDA principle); "
            "'4.1,4.2,4.3'   (three principles); "
            "'Art.6,Art.7'   (GDPR articles)."
        ),
    )
    ex.add_argument(
        "--top-k", type=int, default=DEFAULT_TOP_K,
        help=f"RAG chunks per concept per article (default: {DEFAULT_TOP_K})",
    )
    ex.add_argument(
        "--max-retries", type=int, default=DEFAULT_RETRIES,
        help=(
            f"Validation retries per LLM call (default: {DEFAULT_RETRIES}). "
            "Use 3-4 with local models."
        ),
    )
    ex.add_argument(
        "--no-concept-tags", action="store_true",
        help="Disable concept-tag pre-filtering (~3x more calls, ISSUE-3).",
    )

    run = p.add_argument_group("Run Mode")
    run.add_argument(
        "--stage", choices=["ingest", "extract", "analyse", "all"], default="all",
        help=(
            "Stage to run (default: all). "
            "ingest=chunk+embed only; "
            "extract=LLM extraction (needs prior ingest); "
            "analyse=gap report only; "
            "all=end-to-end."
        ),
    )
    run.add_argument(
        "--dry-run", action="store_true",
        help="Skip all LLM calls — tests all code paths at zero cost.",
    )
    run.add_argument("--verbose", "-v", action="store_true",
                     help="Enable DEBUG logging.")

    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    db_path     = Path(args.db)
    repo_path   = Path(args.repo)
    report_path = Path(args.report)
    data_dir    = Path(args.data_dir)
    law_files   = _parse_law_files(args.input or [])
    stats       = PipelineStats()

    article_filter = _build_article_filter(args.articles)

    log.info("─" * 60)
    log.info("Privacy Policy Extraction Pipeline  (PrivacyPolicyMetamodel v4)")
    log.info("─" * 60)
    log.info(f"  Backend        : {args.backend}  (dry-run={args.dry_run})")
    if args.backend == "anthropic" and not args.dry_run:
        log.info(f"  Model          : {args.model}")
    elif args.backend == "local" and not args.dry_run:
        log.info(f"  Local URL      : {args.local_url}")
        log.info(f"  Local model    : {args.local_model}")
    log.info(f"  Stage          : {args.stage}")
    log.info(f"  Article filter : {args.articles or 'none (all articles)'}")
    log.info(f"  Laws           : {list(law_files.keys()) or '(auto-discover from db)'}")

    # ── Stage 1: Ingest ───────────────────────────────────────────────────────
    embedder_paths: dict[str, Path] = {}

    if args.stage in ("ingest", "all"):
        if not law_files:
            log.error("--input is required for --stage ingest or all.")
            sys.exit(1)
        embedder_paths = stage_ingest(law_files, db_path, data_dir, stats)

    # ── Stages 2-4: Extract → Assemble → Store ────────────────────────────────
    if args.stage in ("extract", "all"):
        if not embedder_paths:
            embedder_paths = _discover_embedder_paths(db_path, data_dir)
        if not embedder_paths:
            log.error(
                "No embedder pickle files found. "
                "Run --stage ingest first or provide --input."
            )
            sys.exit(1)

        backend = _make_backend(args)

        extraction_results = stage_extract(
            db_path          = db_path,
            embedder_paths   = embedder_paths,
            backend          = backend,
            stats            = stats,
            top_k            = args.top_k,
            max_retries      = args.max_retries,
            article_filter   = article_filter,
            use_concept_tags = not args.no_concept_tags,
        )

        stage_assemble_and_store(
            extraction_results = extraction_results,
            repo_path          = repo_path,
            backend            = backend,
            stats              = stats,
            max_retries        = args.max_retries,
        )

    # ── Stage 5: Gap Analysis ─────────────────────────────────────────────────
    if args.stage in ("analyse", "all"):
        stage_analyse(repo_path, report_path)

    stats.log_summary()


if __name__ == "__main__":
    main()