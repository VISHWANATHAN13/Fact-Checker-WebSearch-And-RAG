# FactCheck RAG

> **Evidence-grounded claim verification powered by Hybrid RAG, OpenAI, and transparent audit metrics.**

FactCheck RAG is a Retrieval-Augmented Generation (RAG) application that verifies factual claims against a local knowledge base and, when necessary, the public web. Every verdict is accompanied by four auditable transparency metrics so users can inspect *why* a claim was classified the way it was.


---

## Table of Contents

- [FactCheck RAG](#factcheck-rag)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Architecture Overview](#architecture-overview)
  - [Tech Stack](#tech-stack)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Configuration](#configuration)
  - [Usage](#usage)
    - [1. Ingest Documents](#1-ingest-documents)
    - [2. Launch the Streamlit UI](#2-launch-the-streamlit-ui)
    - [3. Run Tests](#3-run-tests)
  - [How It Works](#how-it-works)
    - [Hybrid Retrieval Pipeline](#hybrid-retrieval-pipeline)
    - [Fact-Checking Pipeline](#fact-checking-pipeline)
    - [Web Search Fallback](#web-search-fallback)
    - [Evidence Conflict Detection](#evidence-conflict-detection)
  - [Audit Metrics](#audit-metrics)
    - [Verdict Categories](#verdict-categories)
  - [Supported File Formats](#supported-file-formats)
  - [Project Structure (Tree)](#project-structure-tree)

---

## Features

- **Hybrid Retrieval** - Combines semantic (vector) search with BM25 keyword search, merged via Reciprocal Rank Fusion (RRF) for superior recall.
- **Automatic Web Fallback** - When local evidence is insufficient or contradictory, the system automatically searches the public web using DuckDuckGo.
- **Evidence Conflict Detection** - A dedicated LLM pass identifies when local evidence sources disagree about the claim.
- **Four Transparent Audit Metrics** - Every result reports:
  1. Assumptions Made
  2. External Information Added
  3. Portions That Can Be Verified
  4. Portions That Cannot Be Verified
- **Evidence-ID Tracing** - Every statement in the verdict links back to the specific evidence chunks that support it.
- **Multi-format Document Ingestion** - Supports PDF, DOCX, TXT, Markdown, and CSV files.
- **Streamlit Web UI** - Interactive interface with expandable audit panels, color-coded verdicts, and raw JSON inspection.

---

## Architecture Overview

```
+---------------------+
|   Streamlit UI      |
| (streamlit_app.py)  |
+---------+-----------+
          |
          v
+---------------------+
|   Fact Checker      |
| (fact_checker.py)   |
+---------+-----------+
          |
     +----+----+
     |         |
     v         v
+----------+ +----------------+
| Hybrid   | | Web Search     |
| Retrieval| | (DuckDuckGo)   |
| (RRF)    | | (web_search.py)|
+----+-----+ +----------------+
     |
  +--+--+
  |     |
  v     v
+------+ +-----+
|Chroma| |BM25 |
|Vector| |Key- |
|Store | |word |
+------+ +-----+
          |
          v
+---------------------+
|  Document Ingestion |
| (ingestion.py +     |
|  chunking.py)       |
+---------------------+
          |
          v
+---------------------+
|  Source Documents    |
|  (app/data/)        |
+---------------------+
```

---

## Tech Stack

| Layer               | Technology                                          |
| ------------------- | --------------------------------------------------- |
| **LLM**             | OpenAI GPT (via `langchain-openai`)                 |
| **Embeddings**      | OpenAI `text-embedding-3-small`                     |
| **Vector Store**    | ChromaDB (`langchain-chroma`)                       |
| **Keyword Search**  | BM25 (`rank-bm25` via `langchain-community`)        |
| **Fusion**          | Reciprocal Rank Fusion (custom implementation)      |
| **Web Search**      | DuckDuckGo (`ddgs`)                                 |
| **Text Splitting**  | `langchain-text-splitters` (RecursiveCharacter)      |
| **UI**              | Streamlit                                           |
| **Language**        | Python 3.13+                                        |

---

## Prerequisites

- **Python 3.13+**
- **OpenAI API Key** with access to a chat model and an embedding model.
- **pip** (or any Python package manager).

---

## Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd Fact_checker_with_RAG
   ```

2. **Create and activate a virtual environment**

   ```bash
   python -m venv .venv

   # Windows (PowerShell)
   .\.venv\Scripts\Activate.ps1

   # macOS / Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## Configuration

Create a `.env` file in the project root (a template is provided) with the following variables:

```env
OPENAI_API_KEY=sk-your-openai-api-key
OPENAI_MODEL=gpt-4o              # or gpt-4o-mini, gpt-3.5-turbo, etc.
EMBEDDING_MODEL=text-embedding-3-small
```

| Variable          | Description                              | Default                    |
| ----------------- | ---------------------------------------- | -------------------------- |
| `OPENAI_API_KEY`  | Your OpenAI API key                      | *required*                 |
| `OPENAI_MODEL`    | Chat model used for fact-checking        | `gpt-4o`                   |
| `EMBEDDING_MODEL` | Embedding model for vector search        | `text-embedding-3-small`   |

---

## Usage

### 1. Ingest Documents

Place your source documents (PDF, DOCX, TXT, MD, CSV) into the `app/data/` directory, then run the ingestion pipeline:

```bash
python -m app.rag.run_ingest
```

This will:
- Load all supported documents from `app/data/`.
- Split them into overlapping chunks (default: 400 characters, 80 overlap).
- Generate OpenAI embeddings and store them in ChromaDB at `app/db/`.

### 2. Launch the Streamlit UI

```bash
streamlit run app/streamlit_app.py
```

Then open the URL shown in your terminal (typically `http://localhost:8501`).

1. Type or paste a factual claim into the text area.
2. Click **Fact-check claim**.
3. Review the verdict, explanation, four audit metric panels, and retrieved evidence.

### 3. Run Tests

**Test semantic retrieval:**

```bash
python -m app.tests.test_retrieval
```

**Test hybrid (semantic + BM25) retrieval:**

```bash
python -m app.tests.test_hybrid
```

---

## How It Works

### Hybrid Retrieval Pipeline

The retrieval stage combines two complementary search strategies:

1. **Semantic Search** - ChromaDB performs approximate nearest-neighbor search using OpenAI embeddings to find contextually similar chunks.
2. **BM25 Keyword Search** - A BM25 retriever scores chunks by term-frequency relevance for precise keyword matching.
3. **Reciprocal Rank Fusion (RRF)** - Both ranked lists are merged using RRF (`score = 1 / (k + rank)`), which promotes chunks that appear in both lists while preserving diversity.

### Fact-Checking Pipeline

The 7-stage fact-checking pipeline (`fact_checker.py`) processes every claim:

| Stage | Description |
| ----- | ----------- |
| 1     | **Local Retrieval** - Hybrid search retrieves the top-k evidence chunks from ChromaDB. |
| 2     | **Local Analysis** - The LLM evaluates the claim against local evidence and produces a structured JSON verdict. |
| 3     | **Conflict Detection** - A separate LLM call checks whether local evidence sources contradict each other. |
| 4     | **Web Fallback Decision** - The system triggers an external web search if local evidence is missing, the verdict is "Insufficient Evidence", unverifiable portions exist, or conflicts are detected. |
| 5     | **Reassessment** - When web evidence is found, the claim is re-evaluated against the combined local + web evidence. |
| 6     | **Evidence-ID Validation** - All cited evidence IDs in the final audit are validated against the actual evidence pool; invalid IDs are removed. |
| 7     | **Result Assembly** - The final result object is returned with verdict, audit metrics, evidence, conflict report, and search metadata. |

### Web Search Fallback

When local evidence is insufficient, the system searches the public web using **DuckDuckGo** (`ddgs` library). Web results are assigned `W`-prefixed evidence IDs (e.g., `W1`, `W2`) and merged with local evidence (`L`-prefixed IDs) for a combined reassessment.

### Evidence Conflict Detection

A dedicated LLM pass examines all local evidence chunks to determine whether they contain materially contradictory information about the claim. The report classifies each evidence chunk as *supporting* or *contradicting* and flags conflicts that warrant external verification.

---

## Audit Metrics

Every fact-check result includes four mandatory transparency metrics:

| # | Metric | Description |
|---|--------|-------------|
| 1 | **Assumptions Made** | Unstated premises that the analysis relied on but that are not directly established by the evidence. |
| 2 | **External Information Added** | Factual details introduced in the explanation that go beyond the original claim, with their evidence source identified. |
| 3 | **Portions That Can Be Verified** | Sub-claims that the evidence can confirm (*Supported*) or refute (*Contradicted*), with cited evidence IDs. |
| 4 | **Portions That Cannot Be Verified** | Sub-claims for which the available evidence is insufficient to determine truth or falsehood. |

### Verdict Categories

| Verdict                 | Meaning |
| ----------------------- | ------- |
| **Fully True**          | All material parts of the claim are supported by evidence. |
| **Partially True**      | At least one material part is supported while another is contradicted or unresolved. |
| **False**               | Evidence contradicts a material part of the claim. |
| **Insufficient Evidence** | The available evidence cannot establish the verdict. Missing evidence alone is **not** proof of falsehood. |

---

## Supported File Formats

| Format   | Extension | Loader Used        |
| -------- | --------- | ------------------ |
| PDF      | `.pdf`    | `PyPDFLoader`      |
| Word     | `.docx`   | `Docx2txtLoader`   |
| Text     | `.txt`    | `TextLoader`       |
| Markdown | `.md`     | `TextLoader`       |
| CSV      | `.csv`    | `CSVLoader`        |

---

## Project Structure (Tree)

```
Fact_checker_with_RAG/
|
|-- .env                          # Environment variables (API keys, model names)
|-- .gitignore                    # Git ignore rules
|-- requirements.txt              # Python dependencies
|-- structure.py                  # Project scaffolding script
|-- README.md                     # This file
|
|-- app/
|   |-- __init__.py
|   |-- main.py                   # Application entry point
|   |-- streamlit_app.py          # Streamlit web UI
|   |
|   |-- data/
|   |   |-- .gitkeep
|   |   |-- cricket.txt           # Sample knowledge-base document
|   |
|   |-- db/
|   |   |-- chroma.sqlite3        # ChromaDB persistent storage
|   |
|   |-- llm_config/
|   |   |-- __init__.py
|   |   |-- llm.py                # LLM and embedding model setup
|   |
|   |-- rag/
|   |   |-- __init__.py
|   |   |-- ingestion.py          # Multi-format document loading
|   |   |-- chunking.py           # Recursive text splitting
|   |   |-- vectorstore.py        # ChromaDB indexing and search
|   |   |-- hybrid_retrieval.py   # Semantic + BM25 + RRF fusion
|   |   |-- fact_checker.py       # 7-stage fact-checking pipeline
|   |   |-- web_search.py         # DuckDuckGo web search fallback
|   |   |-- run_ingest.py         # CLI ingestion runner
|   |
|   |-- tests/
|   |   |-- test_retrieval.py     # Semantic retrieval test
|   |   |-- test_hybrid.py        # Hybrid retrieval test
|   |
|   |-- main.py
|   |-- streamlit_app.py      # Alternate UI display helpers
|       
```

---
