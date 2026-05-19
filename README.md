# PrivacyPolicyMetamodel — LLM Extraction Pipeline

> **A Metamodel-Driven Pipeline for Automated Privacy Policy Extraction and Compliance Gap Analysis**

This pipeline ingests privacy law documents, extracts structured compliance concepts using a two-pass LLM strategy, validates every output against a formal UML metamodel, serializes results as XMI model instances, and produces machine-readable gap analysis reports.

Demonstrated on **PIPEDA** (Personal Information Protection and Electronic Documents Act): 33 articles processed, 28 PolicyStatement instances stored, 28 XMI files generated, 2.1% Pass-1 failure rate.

---

## Table of Contents

- [Research Context](#research-context)
- [Architecture Overview](#architecture-overview)
- [The Metamodel](#the-metamodel)
- [Pipeline Stages](#pipeline-stages)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
- [Demonstration Results — PIPEDA](#demonstration-results--pipeda)
- [Gap Analysis Queries](#gap-analysis-queries)
- [Prompt Architecture](#prompt-architecture)
- [Known Issues & Limitations](#known-issues--limitations)
- [Dependencies](#dependencies)

---

## Research Context

This tool is the implementation artifact for the paper:

> *A Metamodel-Driven Pipeline for Automated Privacy Policy Extraction and Compliance Gap Analysis*

The central design principle is that **the schema IS the prompt** — every extraction target, every enum constraint, and every OCL rule in the metamodel has a direct, traceable counterpart in the prompt system. If a field is removed from the metamodel, the prompt loses it automatically.

---

## Architecture Overview

```
Privacy Law Documents (PDF / TXT)
        │
        ▼
 ┌─────────────┐
 │  INGEST     │  chunk by article → TF-IDF embed → SQLite ChunkStore
 └──────┬──────┘
        │  RAG retrieval (per article × per concept)
        ▼
 ┌─────────────┐
 │  EXTRACT    │  Pass 1 — one LLM call per concept per article
 │  (Pass 1)   │  → raw JSON → Pydantic validation → rejection-loop retry
 └──────┬──────┘
        │  validated concept objects (9 per article)
        ▼
 ┌─────────────┐
 │  ASSEMBLE   │  Pass 2 — compose all Pass-1 objects into PolicyStatement
 │  (Pass 2)   │  → OCL pre-checks → PolicyStatementModel.model_validate()
 └──────┬──────┘
        │
        ▼
 ┌─────────────┐
 │  STORE      │  ModelRepository (SQLite) + XMI serialization (pyecore)
 └──────┬──────┘
        │
        ▼
 ┌─────────────┐
 │  ANALYSE    │  GapAnalyser → coverage matrix + cross-law delta + JSON/text report
 └─────────────┘
```

**Three-Layer Metamodel Mapping**

```
PrivacyPolicyMetamodel  (UML / privacy_metamodel.ecore)
        │  drives
        ▼
Pydantic Schema  (models.py / enums.py)
        │  drives
        ▼
Prompt Templates  (prompts.py)
        │
        ▼
LLM Output → Validation → ModelRepository → XMI
```

---

## The Metamodel

The **PrivacyPolicyMetamodel** is a formal UML metamodel defining the structure of a privacy policy in a domain-independent, jurisdiction-reusable way.

| Dimension | Value |
|---|---|
| Classes | 14 |
| Associations | 29 |
| Enums | 13 (74 literals) |
| OCL constraints | 5 |
| XMI namespace | `http://www.example.org/privacypolicy` |
| Ecore file | `privacy_metamodel.ecore` |

**Five packages:**

- `Core` — `PrivacyPolicy`, `PolicyStatement`
- `Actors` — `Actor`, `ActorRole`
- `Processing` — `ProcessingActivity`, `PersonalData`
- `PolicyRules` — `LegalBasis`, `Constraint`, `Right`, `RetentionPolicy`, `DataTransfer`, `ConsentWithdrawal`, `Purpose`
- `ControlledVocabularies` — all 13 enums

**Five OCL constraints:**

| ID | Rule | Legal Reference |
|---|---|---|
| constraint_1 | Every PolicyStatement must have at least one Purpose | GDPR Art.5(1)(b), PIPEDA Principle 2 |
| constraint_2 | Every LegalBasis must have at least one Jurisdiction | GDPR Art.3, PIPEDA s.4 |
| constraint_3 | High-sensitivity Transfer without DPIA reference triggers warning | GDPR Art.35 |
| constraint_4 | Consent basis without ConsentWithdrawal triggers warning | GDPR Art.7(3), PIPEDA Principle 3 |
| constraint_5 | At least one Constraint must be present per PolicyStatement | GDPR Art.5, PIPEDA Sch.1 |

---

## Pipeline Stages

| # | Stage | Input | Output |
|---|---|---|---|
| 1 | **Ingest** | PDF / TXT law file | SQLite ChunkStore + TF-IDF embedder |
| 2 | **Extract (Pass 1)** | ChunkStore | 9 validated concept JSON objects per article |
| 3 | **Assemble (Pass 2)** | Pass-1 concept objects | Validated `PolicyStatement` instance |
| 4 | **Store** | `PolicyStatement` | ModelRepository (SQLite) + XMI file |
| 5 | **Analyse** | ModelRepository | Coverage matrix, gap report (.txt + .json) |

**Pass 1** makes one LLM call per `(article × concept)` pair. Nine concepts are extracted: `LegalBasis`, `ProcessingActivity`, `Actor`, `Purpose`, `Right`, `Constraint`, `RetentionPolicy`, `DataTransfer`, `ConsentWithdrawal`.

**Pass 2** assembles all nine objects into a single `PolicyStatement`, runs OCL consistency checks, and logs warnings as compliance gap findings.

---

## Project Structure

```
llm-privacy-metamodel-extractor/
├── run_pipeline.py                ← CLI entry point, orchestrates all stages
│
├── rag_pipeline/
│   ├── chunker.py                 ← Hierarchical legal text chunker (article-level)
│   ├── embedder.py                ← TF-IDF embedder (SentenceTransformer stub available)
│   ├── retriever.py               ← Concept-tag pre-filter + cosine similarity rerank
│   └── store.py                   ← SQLite ChunkStore + ingest_file()
│
├── privacy_schema/
│   ├── enums.py                   ← 13 controlled-vocabulary enums (mirrors metamodel)
│   ├── models.py                  ← 14 Pydantic models + 5 OCL validators
│   ├── prompts.py                 ← build_concept_prompt(), build_assembler_prompt()
│   └── prompt_architecture.md    ← Full prompt design rationale
│
├── gap_analyses/
│   ├── gap_analysis.py            ← GapAnalyser — 9 cross-law query functions
│   └── repository.py             ← ModelRepository (SQLite read/write)
│
├── tranform_format/
│   ├── pydantic_to_xmi.py         ← PolicyStatement → XMI 2.1 serializer (pyecore)
│   └── generate_ecore.py          ← Generates privacy_metamodel.ecore from scratch
│
├── laws/                          ← Input law PDFs (not committed — add your own)
├── data/                          ← Runtime data: chunks.db, model_repo.db, embedders
├── output/xmi/                    ← Generated XMI model instances
│
├── privacy_metamodel.ecore        ← Ecore metamodel (required for XMI output)
├── privacy_schema/extraction_demo.py  ← Standalone demo (zero API cost)
└── requirements.txt
```

---

## Quick Start

**1. Install dependencies**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**2. Generate the Ecore metamodel** (required for XMI output)
```bash
python tranform_format/generate_ecore.py privacy_metamodel.ecore
```

**3. Start a local LLM** (Ollama recommended)
```bash
ollama pull mistral-nemo
ollama serve
```

**4. Run the full pipeline on PIPEDA**
```bash
python run_pipeline.py \
    --input PIPEDA=laws/pipeda.pdf \
    --stage all \
    --xmi-out output/xmi \
    --backend local --local-model mistral-nemo:latest
```

**5. Run gap analysis only** (uses existing model_repo.db)
```bash
python run_pipeline.py --stage analyse
```

---

## CLI Reference

```
Input / Output
  --input   LAW=path [LAW=path ...]  Law files to ingest, e.g. PIPEDA=laws/pipeda.pdf
  --db      PATH                     SQLite chunk store   (default: data/chunks.db)
  --repo    PATH                     Model repository     (default: data/model_repo.db)
  --xmi-out DIR                      XMI output directory (default: disabled)

LLM Backend
  --backend   {anthropic|local}      LLM backend (default: anthropic)
  --model     MODEL                  Anthropic model      (default: claude-sonnet-4-20250514)
  --local-url URL                    Local LLM base URL   (default: http://localhost:11434/v1)
  --local-model MODEL                Local model name     (default: llama3.1:8b)

Extraction Control
  --articles  FILTER                 Comma-separated article substrings, e.g. "4.1,4.2,4.3"
  --top-k     N                      RAG chunks per concept per article (default: 3)
  --max-retries N                    Validation retries per LLM call    (default: 2)
  --no-concept-tags                  Disable concept-tag pre-filtering  (~3x more LLM calls)

Run Mode
  --stage  {ingest|extract|analyse|all}  Pipeline stage (default: all)
  --dry-run                              Skip all LLM calls — tests code paths at zero cost
  -v / --verbose                         Enable DEBUG logging
```

**Common patterns:**

```bash
# Ingest only (no LLM calls)
python run_pipeline.py --input PIPEDA=laws/pipeda.pdf --stage ingest

# Filter to specific principles
python run_pipeline.py \
    --input PIPEDA=laws/pipeda.pdf \
    --articles "4.1,4.2,4.3" \
    --backend local --local-model mistral-nemo:latest

# Dry run — test all code paths, zero API cost
python run_pipeline.py --input PIPEDA=laws/pipeda.pdf --dry-run

# Gap analysis from existing repository
python run_pipeline.py --stage analyse
```

---

## Demonstration Results — PIPEDA

Full run on PIPEDA (all 33 articles), `mistral-nemo:latest` (12B) via Ollama:

| Metric | Value |
|---|---|
| Articles processed | 33 / 33 |
| Pass-1 calls | 151 (ok=132, absent=8, failed=3) |
| Pass-1 failure rate | **2.1%** |
| Pass-2 calls | 33 (ok=28, failed=5) |
| Pass-2 success rate | **85%** |
| Statements stored | **28** |
| XMI files written | **28** |
| Total LLM calls | 204 |
| Tokens in / out | 372,581 / 47,612 |

**Gap report summary (PIPEDA, 28 statements):**

| Dimension | Finding |
|---|---|
| Legal basis types | Consent, LegalObligation |
| Rights granted | Access only |
| Constraint types | PurposeLimitation, Security, Temporal |
| Purpose categories | ServiceProvision, LegalCompliance, Analytics |
| Transfer mechanisms | Absent — PIPEDA has no explicit transfer framework |
| Retention rules | 0% — left to organizational discretion |
| DPIA / risk assessment | 0% — no mandatory PIA requirement in PIPEDA |
| Consent withdrawal | 4% (occasional) — found in enforcement articles |

**Sample XMI output** (`PIPEDA_4_3_Principle_3_Consent.xmi`):

```xml
<pp:PrivacyPolicy xmlns:pp="http://www.example.org/privacypolicy" ...>
  <statements description="Organization's transfer of personal data to third parties">
    <actor name="Organization" role="DataController"/>
    <purposes description="..." category="ServiceProvision"/>
    <processingActivity action="Transfer">
      <dataProcessed category="Identifier" sensitivity="Medium" identifiability="Identified"/>
    </processingActivity>
    <legalBasis type="Consent" evidence="The knowledge and consent of the individual...">
      <jurisdiction jurisdictionId="CA" name="Canada"/>
    </legalBasis>
    <constraints type="Temporal" expression="..." enforcementLevel="Mandatory"/>
    <rightImpacted type="Access" triggerCondition="Upon written request by data subject"
                   fulfillmentProcess="Organization must respond within 30 days"/>
  </statements>
</pp:PrivacyPolicy>
```

---

## Gap Analysis Queries

The `GapAnalyser` runs nine structured queries over the model repository. Results are law-agnostic — adding a new law automatically appears in all outputs.

| Query | Description |
|---|---|
| Q1 | Legal basis type coverage per law |
| Q2 | Data subject rights coverage per law |
| Q3 | Retention policy mandate rate |
| Q4 | DPIA / risk assessment mandate rate |
| Q5 | Transfer mechanism coverage |
| Q6 | Consent withdrawal coverage |
| Q7 | Constraint type coverage |
| Q8 | Cross-law delta (obligations in law A absent from law B) |
| Q9 | Purpose category coverage |

---

## Prompt Architecture

Every concept prompt has six sections:

1. **Task header** — one unambiguous class name
2. **Cross-law mapping** — which article in GDPR / LGPD / CCPA / PIPEDA maps to this concept
3. **Output schema** — JSON skeleton using Pydantic alias names; multiplicity encoded as list wrappers or `null` defaults
4. **Enum grammar** — exact vocabulary the LLM may output (case-sensitive)
5. **OCL hints** — natural-language versions of the OCL constraints for this concept
6. **Few-shot example** — one correct input/output pair
7. **Retrieved context** — RAG chunks injected at call time

See [`privacy_schema/prompt_architecture.md`](privacy_schema/prompt_architecture.md) for the full design rationale.

---

## Known Issues & Limitations

| Issue | Severity | Status |
|---|---|---|
| TF-IDF embedder — zero retrieval scores for Actor and Constraint concepts | Medium | Stub for `SentenceTransformerEmbedder` (`multilingual-e5-base`) available in `embedder.py` |
| Article filter substring match — `4.1` can match `4.10` | Low | Fixed: word-boundary regex in `_build_article_filter()` |
| Pass-1 extracts one concept instance per call — under-extracts multi-value concepts | Medium | Increase `--top-k`; future: multi-instance prompt |
| Local 12B model (`mistral-nemo`) produces ~15% Pass-2 failures | Medium | Use `llama3.1:70b-instruct-q4` for better quality |
| Consent withdrawal rarely extracted despite good retrieval scores | Medium | Prompt under-specification; future work |
| DPIA package not yet populated | Low | Placeholder in metamodel; future work |
| Single annotator evaluation | Medium | Inter-annotator agreement study planned |

---

## Dependencies

```
pdfplumber        ← PDF text extraction
pyecore           ← Ecore/XMI serialization
pydantic>=2.0     ← Schema validation + OCL enforcement
numpy             ← TF-IDF cosine similarity
scikit-learn      ← TF-IDF vectorizer
anthropic         ← Anthropic API backend (optional)
```

Install all:
```bash
pip install pdfplumber pyecore "pydantic>=2" numpy scikit-learn anthropic
```

For local LLM support, install [Ollama](https://ollama.com) and pull a model:
```bash
ollama pull mistral-nemo
# or for better quality:
ollama pull llama3.1:70b-instruct-q4
```