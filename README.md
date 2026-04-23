# llm-privacy-metamodel-extractor

> **LLM-driven extraction of privacy policy semantics from regulatory legal texts,
> structured against a formal MBSE metamodel with RAG retrieval and schema-validated output.**

This pipeline ingests privacy law documents (GDPR, CCPA, LGPD, PIPEDA, and others),
extracts structured compliance concepts using a two-pass LLM strategy, validates every
output against a Pydantic schema derived from **PrivacyPolicyMetamodel**, stores the
results in a model repository, and produces cross-law gap analysis reports.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Pipeline Stages](#pipeline-stages)
- [Project Structure](#project-structure)
- [Metamodel Concepts](#metamodel-concepts)
- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
- [Prompt Architecture](#prompt-architecture)
- [Gap Analysis Queries](#gap-analysis-queries)
- [Known Issues & Limitations](#known-issues--limitations)
- [Dependencies](#dependencies)

---

## Architecture Overview

The central design principle is that **the schema IS the prompt** — every extraction
target, every enum constraint, and every OCL rule in the metamodel has a direct,
traceable counterpart in the prompt system.

```
Privacy Law Documents (PDF / TXT)
        │
        ▼
 ┌─────────────┐
 │  INGEST     │  chunk → embed → SQLite ChunkStore
 └──────┬──────┘
        │  RAG retrieval (per article × per concept)
        ▼
 ┌─────────────┐
 │  EXTRACT    │  Pass 1 — one LLM call per concept per article
 │  (Pass 1)   │  → raw JSON → Pydantic validation → rejection loop
 └──────┬──────┘
        │  validated concept objects
        ▼
 ┌─────────────┐
 │  ASSEMBLE   │  Pass 2 — assemble concept objects into PolicyStatement
 │  (Pass 2)   │  → PolicyStatementModel.model_validate()
 └──────┬──────┘
        │
        ▼
 ┌─────────────┐
 │  STORE      │  ModelRepository (SQLite)
 └──────┬──────┘
        │
        ▼
 ┌─────────────┐
 │  ANALYSE    │  GapAnalyser → cross-law coverage matrix + gap report
 └─────────────┘
```

**Three-Layer Metamodel Mapping**

```
PrivacyPolicyMetamodel  (privacy_metamodel.json)
        │
        ▼
Pydantic Schema            (models.py / enums.py)
        │
        ▼
Prompt Templates           (prompts.py)
        │
        ▼
LLM Output → Validation → ModelRepository
```

Nothing in the prompts is invented independently of the metamodel. If a field is
removed from the metamodel, the prompt loses it automatically because
`build_concept_prompt()` derives enum grammars from `_ENUM_GRAMMARS`, which mirrors
`ControlledVocabularies` in the metamodel JSON.

---

## Pipeline Stages

| # | Stage | What it does |
|---|-------|-------------|
| 1 | **Ingest** | Reads law files → chunks by article/clause → fits TF-IDF embedder → stores vectors in SQLite `chunks` table |
| 2 | **Extract** | For each `(article × concept)` pair: RAG-retrieves top-k chunks → calls LLM with `build_concept_prompt()` → validates JSON against Pydantic model → retries on failure |
| 3 | **Assemble** | Collects all Pass-1 concept objects for an article → calls LLM with `build_assembler_prompt()` → validates full `PolicyStatementModel` |
| 4 | **Store** | Writes validated `PolicyStatement` instances to `ModelRepository` |
| 5 | **Analyse** | Runs `GapAnalyser` queries over the repository → outputs Markdown / JSON gap report |

---

## Project Structure

```
privacy-policy-extractor/
├── run_pipeline.py               ← CLI entry point, orchestrates all stages
│
├── rag_pipeline/
│   ├── chunker.py                ← Splits law documents into article-level chunks
│   ├── embedder.py               ← TF-IDF embedder (stub for SentenceTransformer)
│   ├── retriever.py              ← Cosine-similarity retrieval from ChunkStore
│   └── store.py                  ← SQLite chunk store with concept-tag index
│
├── privacy_schema/
│   ├── enums.py                  ← Controlled vocabularies (mirrors metamodel enums)
│   ├── models.py                 ← Pydantic extraction models + OCL validators
│   └── prompts.py                ← build_concept_prompt(), build_assembler_prompt()
│
├── gap_analysis/
│   ├── gap_analysis.py           ← GapAnalyser with 9 cross-law query functions
│   └── repository.py             ← ModelRepository (SQLite read/write)
│
├── privacy_metamodel.json     ← Source-of-truth metamodel
├── prompt_architecture.md        ← Detailed prompt design documentation
└── extraction_demo.py            ← Standalone demo (zero API cost dry-run)
```

---

## Metamodel Concepts

Nine concepts are extracted in **Pass 1**. Each becomes a typed Pydantic model.
Pass 2 assembles them into a single `PolicyStatement`.

| Concept | Metamodel Class | Key Fields |
|---------|----------------|------------|
| `LegalBasis` | `PolicyRules.LegalBasis` | `type` (LegalBasisType), `jurisdiction` |
| `ProcessingActivity` | `Processing.ProcessingActivity` | `action`, `dataProcessed`, `riskAssessmentReference` |
| `Actor` | `Actors.Actor` | `name`, `role` (ActorRole) |
| `Purpose` | `Processing.Purpose` | `description`, `category` (PurposeCategory) |
| `Right` | `PolicyRules.Right` | `type` (RightType), `triggerCondition`, `fulfillmentProcess` |
| `Constraint` | `PolicyRules.Constraint` | `type` (ConstraintType), `expression`, `enforcementLevel` |
| `RetentionPolicy` | `PolicyRules.RetentionPolicy` | `duration`, `unit`, `trigger` |
| `DataTransfer` | `Processing.DataTransfer` | `destinationCountry`, `mechanism` (TransferMechanism) |
| `ConsentWithdrawal` | `PolicyRules.ConsentWithdrawal` | `channel` (1..*), `deadline`, `effectOnPriorProcessing` |

**OCL Constraints enforced at validation time:**

| Constraint | Severity | Rule |
|-----------|----------|------|
| `constraint_2` | Error | Every `Regulation` must have ≥1 `Jurisdiction` |
| `constraint_3` | Warning | Transfer/Share of High/SpecialCategory data requires `riskAssessmentReference` |
| `constraint_4` | Warning | `LegalBasis.type = Consent` implies `ConsentWithdrawal` must be present |
| `constraint_dt1` | Error | `DataTransfer` must have a valid `mechanism` |
| `constraint01` | Error | `PrivacyPolicy` must contain ≥1 `PolicyStatement` |

---

## Quick Start

### Prerequisites

```bash
python >= 3.10
pip install pydantic anthropic sentence-transformers sqlite3
```

### Run the demo (no API key required)

```bash
python extraction_demo.py
```

### Ingest only (no LLM calls)

```bash
python run_pipeline.py \
    --input PIPEDA=data/pipeda.pdf \
    --stage ingest
```

### Single-article prototype with local Ollama

```bash
python run_pipeline.py \
    --input PIPEDA=data/pipeda.pdf \
    --articles "Principle 4.1" \
    --backend local --local-model llama3.1:8b
```

### Full multi-law run with Anthropic API

```bash
export ANTHROPIC_API_KEY=sk-...

python run_pipeline.py \
    --input GDPR=data/gdpr.pdf CCPA=data/ccpa.txt LGPD=data/lgpd.pdf PIPEDA=data/pipeda.pdf \
    --backend anthropic
```

### Gap report from an existing repository

```bash
python run_pipeline.py --stage analyse
```

### Dry run (tests all code paths, zero API cost)

```bash
python run_pipeline.py --input PIPEDA=data/pipeda.pdf --dry-run
```

---

## CLI Reference

```
run_pipeline.py [OPTIONS]

Input / Output
  --input  LAW=path [LAW=path ...]   Law files to ingest. e.g. GDPR=data/gdpr.pdf
  --db     PATH                      SQLite chunk store path (default: chunks.db)
  --repo   PATH                      Model repository path  (default: repository.db)
  --report PATH                      Gap report output path (default: gap_report.md)

LLM Backend
  --backend   {anthropic|local}      LLM backend (default: anthropic)
  --model     MODEL                  Anthropic model string (default: claude-sonnet-4-20250514)
  --local-url URL                    Local LLM base URL    (default: http://localhost:11434/v1)
  --local-model MODEL                Local model name      (default: llama3.1:8b)

Extraction Control
  --articles  ARTICLE [ARTICLE ...]  Filter to specific articles e.g. "Art.6" "Art.17"
  --top-k     N                      RAG top-k chunks per concept (default: 3)
  --retries   N                      LLM retry attempts on validation failure (default: 2)
  --no-concept-tags                  Disable concept-tag pre-filtering (~3x more calls)

Run Mode
  --stage  {ingest|extract|analyse|all}  Pipeline stage to run (default: all)
  --dry-run                              Skip all LLM calls (zero cost testing)
  -v / --verbose                         Enable DEBUG logging
```

---

## Prompt Architecture

Every concept prompt has **six sections**:

1. **Task header** — one unambiguous class name (`## Task: Extract a LegalBasis instance`)
2. **Cross-law mapping** — which article in each law maps to this concept, e.g. `GDPR Art.6(1)(a-f) | LGPD Art.7 | CCPA business-purpose | PIPEDA Sch.1`
3. **Output schema** — the JSON skeleton to fill, using Pydantic alias names, with multiplicity encoded as list wrappers or `null` defaults
4. **Enum grammar** — the exact vocabulary the LLM may output, e.g. `LegalBasisType: Consent | Contract | LegalObligation | ...`
5. **OCL hints** — natural-language versions of the OCL constraints that apply to this concept
6. **Retrieved context** — the RAG chunks injected at call time

See [`prompt_architecture.md`](prompt_architecture.md) for the full design rationale.

---

## Gap Analysis Queries

The `GapAnalyser` runs nine structured queries over the model repository.
Results are law-agnostic — adding a new law automatically appears in all outputs.

| Query | Function | Description |
|-------|----------|-------------|
| Q1 | `legal_basis_coverage()` | Which `LegalBasisType` values each law allows |
| Q2 | `rights_coverage()` | Which `RightType` values each law grants |
| Q3 | `retention_coverage()` | Which laws mandate retention rules |
| Q4 | `dpia_coverage()` | Which laws mandate impact assessments |
| Q5 | `transfer_coverage()` | Which `TransferMechanism` values each law recognises |
| Q6 | `constraint_coverage()` | Which `ConstraintType` values each law imposes |
| Q7 | `purpose_coverage()` | Which `PurposeCategory` values each law governs |
| Q8 | `cross_law_delta()` | Obligations present in law A but absent from law B |
| Q9 | `coverage_matrix()` | Full concept-by-law presence matrix (primary research deliverable) |

---

## Known Issues & Limitations

| ID | Severity | Description |
|----|----------|-------------|
| ISSUE-1 | Medium | TF-IDF embedder is fitted per-law on its own corpus only. No cross-lingual retrieval. Swap to `SentenceTransformerEmbedder` (e.g. `multilingual-e5-base`) for production quality. Stub available in `embedder.py`. |
| ISSUE-2 | Low | PIPEDA chunker produces principle-level chunks. Already handled via `level='principle'` in the article filter. Use `--articles "Principle 4.1"` to target a single principle. |
| ISSUE-3 | High | LLM call volume scales as `|laws| × |articles| × |concepts|`. Use `--top-k 3-4` with local models and `--articles` to limit scope. Concept-tag pre-filtering reduces calls ~3x by default. |
| ISSUE-4 | Low | Jurisdiction metadata for `Regulation` objects is synthesised statically. Add new laws to the `JURISDICTION_MAP` constant in `run_pipeline.py`. |
| ISSUE-5 | Low | Some concepts (Purpose, Right, Constraint, RetentionPolicy, DataTransfer, ConsentWithdrawal) produce lists in the assembler even though Pass 1 returns one object per call. The assembler prompt handles the merge. |

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `pydantic >= 2.0` | Schema validation and OCL constraint enforcement |
| `anthropic` | Anthropic API backend |
| `scikit-learn` | TF-IDF embedder |
| `numpy` | Cosine similarity for RAG retrieval |
| `sqlite3` | Chunk store and model repository (stdlib) |

**Optional (recommended for production):**

| Package | Purpose |
|---------|---------|
| `sentence-transformers` | Semantic embedder (`multilingual-e5-base`) to replace TF-IDF |
| `pdfminer.six` | PDF ingestion |
| `ollama` | Local LLM backend |

---

## Supported Laws

Any law structured with numbered articles or principles can be ingested.
Laws tested in development:

| Law | Region | Format |
|-----|--------|--------|
| GDPR 2016/679 | EU | PDF |
| CCPA / CPRA | California, US | TXT / PDF |
| LGPD | Brazil | PDF |
| PIPEDA | Canada | PDF |

To add a new law, provide `LAWNAME=path/to/file` via `--input` and add a
jurisdiction entry to `JURISDICTION_MAP` in `run_pipeline.py` (see ISSUE-4).

---

## License

*(Add your license here)*

## Citation

*(Add your citation here once published)*
