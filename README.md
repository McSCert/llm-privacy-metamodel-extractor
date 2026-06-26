# Privacy Policy Metamodel Extractor

A metamodel-driven pipeline for automatically extracting structured, formally validated model instances from statutory privacy law documents.

Built on a formal Ecore metamodel (`PrivacyPolicyMetamodel`), the pipeline compiles the metamodel into a runtime JSON schema and uses constrained LLM token generation to guarantee syntactically valid extractions. Extracted instances are validated against OCL constraints and serialized to XMI 2.1.

---

## Repository Structure

```
├── metamodel/                    # Ecore metamodel definition
│   ├── privacy_metamodel.ecore   # Ecore file used by the pipeline
│   ├── privacy_metamodel.xmi     # Original Rhapsody UML export
│   └── privacy_metamodel.json    # JSON representation
├── privacy_schema/               # Pydantic models compiled from metamodel
│   ├── models.py                 # 14 EClass Pydantic models
│   ├── enums.py                  # 12 controlled vocabulary enumerations
│   └── _ocl_validators.py        # 9 OCL constraint validators
├── rag_pipeline/                 # BM25 retrieval pipeline
│   ├── chunker.py                # Document chunking
│   ├── embedder.py               # BM25 indexing
│   ├── retriever.py              # Chunk retrieval
│   └── store.py                  # Chunk storage
├── gap_analyses/                 # Gap analysis and reporting
│   ├── gap_analysis.py           # Coverage matrix generation
│   └── repository.py             # Model repository interface
├── tranform_format/              # XMI serialization
│   └── pydantic_to_xmi.py        # Pydantic → XMI 2.1 converter
├── schemas/                      # Generated Pydantic JSON schemas
│   ├── all_schemas.json          # Combined schema file
│   └── *.json                    # Individual class schemas (14 files)
├── output/xmi-local/             # XMI output files from pipeline runs
├── evaluation/                   # Evaluation artifacts
│   └── pipeda_gold_standard.csv  # Manually annotated gold standard
├── laws/                         # Input law documents
│   └── pipeda.pdf                # PIPEDA full text
├── run_pipeline.py               # Main pipeline entry point
├── generate_schemas.py           # Schema generation script
└── evaluate.py                   # Evaluation script
```

---

## Installation

**Requirements:** Python 3.9+

```bash
# Clone the repository
git clone https://github.com/McSCert/llm-privacy-metamodel-extractor.git
cd llm-privacy-metamodel-extractor

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Local Configuration (Mistral-Nemo via Ollama)

First, install [Ollama](https://ollama.com) and pull the model:

```bash
ollama pull mistral-nemo:latest
ollama serve
```

Then run the pipeline:

```bash
python run_pipeline.py \
  --input PIPEDA=laws/pipeda.pdf \
  --backend local \
  --local-model mistral-nemo:latest \
  --stage all \
  --report data/gap_report_mistral.txt \
  --db data/chunks_mistral.db \
  --repo data/model_repo_mistral.db
```

### Online Configuration (GPT-4o via OpenAI API)

```bash
export OPENAI_API_KEY=your_api_key_here

python run_pipeline.py \
  --input PIPEDA=laws/pipeda.pdf \
  --backend openai \
  --stage all \
  --report data/gap_report_gpt4o.txt \
  --db data/chunks_gpt4o.db \
  --repo data/model_repo_gpt4o.db
```

---

## Evaluation

To reproduce the evaluation against the PIPEDA gold standard:

```bash
python evaluate.py \
  --gold evaluation/pipeda_gold_standard.csv \
  --repo data/model_repo_gpt4o.db \
  --output evaluation/results_gpt4o.csv
```

---

## Generating Pydantic Schemas

To regenerate the JSON schemas from the Pydantic models:

```bash
python generate_schemas.py
```

Output is saved to `schemas/`.

---

## Metamodel

The `PrivacyPolicyMetamodel` was designed in IBM Rational Rhapsody and exported as XMI. The XMI file was converted to Ecore format for use with `pyecore`. Both files are provided in the `/metamodel` directory.

The metamodel contains:
- **14 EClasses** organized into 6 conceptual layers (Structural, Actors, Operational, Normative, Regulatory, Vocabulary)
- **12 controlled vocabulary enumerations** binding every attribute to a fixed set of typed literals
- **9 OCL constraints** enforcing structural completeness (5 errors) and compliance quality (4 warnings)

| Layer | Classes |
|---|---|
| Structural | PrivacyPolicy, PolicyStatement |
| Actors | Actor |
| Operational | ProcessingActivity, PersonalData, Purpose, DataTransfer |
| Normative | LegalBasis, Constraint, Right, RetentionPolicy, ConsentWithdrawal |
| Regulatory | Regulation, Jurisdiction |

---

## OCL Constraints

| ID | Rule | Severity |
|---|---|---|
| C01 | A `PrivacyPolicy` must contain at least one `PolicyStatement` | Error |
| C02 | Every `Regulation` must reference at least one `Jurisdiction` | Error |
| C03 | Every `DataTransfer` must specify a valid `mechanism` | Error |
| C04 | Every `PolicyStatement` must reference at least one `Purpose` | Error |
| C05 | A `ProcessingActivity` with action `Share` or `Transfer` must have at least one `DataTransfer` | Error |
| C06 | A `ProcessingActivity` involving `High` or `SpecialCategory` data must include a `riskAssessmentReference` | Warning |
| C07 | When `LegalBasis.type` is `Consent`, a `ConsentWithdrawal` must be present | Warning |
| C08 | A `RetentionPolicy` with `unit = Indefinite` must include a justification | Warning |
| C09 | A `Right` of type `Erasure` or `Portability` must include a `fulfillmentProcess` | Warning |

---

## XMI Output

The `/output/xmi-local` directory contains 27 XMI 2.1 files produced by the Mistral-Nemo pipeline run, one per PIPEDA article. These can be imported directly into IBM Rational Rhapsody or Eclipse EMF for further analysis and visualization.

---

## Evaluation Results

The pipeline was evaluated against a manually annotated gold standard covering 10 PIPEDA principles across 9 concept dimensions.

| Configuration | Model | Pass-1 Failure Rate | Structural Failures | Pass-2 Failure Rate | Macro F1 |
|---|---|---|---|---|---|
| Online | GPT-4o | 2.2% | 0% | 0.0% | 52.3% |
| Local | Mistral-Nemo (12B) | 0.0% | 0% | 18.5% | 48.2% |

Both configurations achieve zero structural failures, confirming that constrained token decoding eliminates schema violations regardless of model size.

---
