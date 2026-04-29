# TL;DR Legal

A RAG (Retrieval-Augmented Generation) CLI that analyses Terms of Service files against two legal knowledge bases and summarises risks in plain English using Llama 3.2.

## What It Does

- Builds a FAISS vector index from two datasets: **Claudette** (unfairness detector) and **ToS;DR** (cases + topics).
- Chunks input `.txt` files into paragraphs and runs semantic search against the index.
- Groups matches by category, keeping only the highest-scoring hit per category per paragraph.
- Assigns severity: `Critical / High / Medium / Low` based on dataset labels.
- Passes the top 5 flags to **Llama 3.2:1b** (via Ollama) for a plain-English 3-bullet summary.
- Prints a colour-coded table to the console and optionally exports JSON or a formatted text report.

## Project Structure

- [main.py](main.py) - RAG pipeline, FAISS index, analysis engine, LLM summarisation, CLI.
- [train-00000-of-00001-a8de7efe0da36666.parquet](train-00000-of-00001-a8de7efe0da36666.parquet) - Claudette unfairness dataset.
- [15012282/cases.csv](15012282/cases.csv) - ToS;DR cases.
- [15012282/topics.csv](15012282/topics.csv) - ToS;DR topics (joined to cases on `topic_id`).
- [.embeddings_cache/](.embeddings_cache) - Auto-generated FAISS embedding cache (first run only).
- [requirements.txt](requirements.txt) - Python dependencies.

## Setup

```powershell
pip install -r requirements.txt
```

To enable LLM summaries, install and start [Ollama](https://ollama.com/download), then pull the model:

```powershell
ollama pull llama3.2:1b
ollama serve
```

## Usage

```powershell
# Analyse a single file
python main.py text\Twitter_TermsofService.txt

# Analyse a whole folder
python main.py text\

# Custom threshold and JSON output
python main.py text\Uber_TermsofService.txt --threshold 0.65 --output report.json

# Human-readable text report
python main.py text\Uber_TermsofService.txt --output report.txt
```

## CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `input` | required | Path to a `.txt` file or folder of `.txt` files |
| `--threshold` | `0.70` | Minimum cosine similarity for a match |
| `--output` | none | Export path — `.json` for JSON, `.txt` for formatted report |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model name |
| `LLM_MODEL` | `llama3.2:1b` | Ollama model used for summarisation |
| `FORCE_HF_OFFLINE` | `true` | Prevents Hugging Face network access at runtime |

## Severity Levels

| Colour | Severity | Source |
|--------|----------|--------|
| RED | Critical | ToS;DR `blocker` |
| RED | High | ToS;DR `bad` / Claudette label `0` |
| YELLOW | Medium | ToS;DR `neutral` |
| GREEN | Low | ToS;DR `good` / Claudette label `1` |

## Notes

- The FAISS index is built once and cached to `.embeddings_cache/`. Subsequent runs load in under a second.
- If Ollama is not running, the LLM summary is skipped gracefully and the rest of the report is unaffected.
- The embedding model must be cached locally when `FORCE_HF_OFFLINE=true`. Download it once while online, then it works fully offline.
