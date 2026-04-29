# TL;DR Legal

TL;DR Legal is a FastAPI app that semantically compares Terms of Service text against a local legal case dataset and optional text corpus files. It uses `sentence-transformers` with `all-MiniLM-L6-v2`, cosine similarity, and cached local models, so it does not call the Hugging Face inference API at runtime.

## What It Does

- Loads `legal_text_classification.csv` once at startup.
- Uses the columns `case_id`, `case_outcome`, `case_title`, and `case_text`.
- Optionally loads `.txt` files from the `text/` folder and includes them in the search index.
- Splits user input into sentences and finds semantic matches above the similarity threshold.
- Returns matching cases plus a `risk_score` based on configured high-risk outcomes.

## Project Structure

- [main.py](main.py) - FastAPI backend, data loading, embeddings, and similarity search.
- [static/index.html](static/index.html) - Single-page Tailwind UI.
- [legal_text_classification.csv](legal_text_classification.csv) - Local dataset used for matching.
- [text/](text) - Optional local `.txt` corpus files that are also embedded.
- [requirements.txt](requirements.txt) - Python dependencies.

## Setup

Install dependencies inside your virtual environment:

```powershell
pip install -r requirements.txt
```

## Run

Start the app with:

```powershell
uvicorn main:app --reload
```

Open the app at `http://127.0.0.1:8000`.

## Environment Variables

- `LEGAL_CSV_PATH` - Path to the CSV file. Defaults to `legal_text_classification.csv`.
- `TEXT_CORPUS_DIR` - Folder containing optional `.txt` files. Defaults to `text`.
- `TEXT_CORPUS_ENABLE` - Set to `false` to skip loading the text folder.
- `TEXT_CORPUS_MAX_FILES` - Maximum number of `.txt` files to read.
- `TEXT_CORPUS_MAX_CHARS` - Maximum characters to read from each text file.
- `EMBED_MODEL_NAME` - Sentence-transformers model name. Defaults to `all-MiniLM-L6-v2`.
- `SIMILARITY_THRESHOLD` - Minimum cosine similarity for a match. Defaults to `0.65`.
- `HIGH_RISK_OUTCOMES` - Comma-separated list of `case_outcome` values counted as high risk.
- `FORCE_HF_OFFLINE` - Defaults to `true`. Prevents Hugging Face network access at runtime.

## API

### `POST /analyze`

Request body:

```json
{
	"text": "paste terms of service here"
}
```

Response body:

```json
{
	"found_flags": [
		{
			"user_sentence": "...",
			"case_id": "...",
			"case_title": "...",
			"case_outcome": "...",
			"case_text": "..."
		}
	],
	"risk_score": 0
}
```

## Notes

- If the embedding model is not already cached locally and `FORCE_HF_OFFLINE=true`, startup will fail fast.
- The app uses the Hugging Face model hub only to load model files, not the hosted inference API.
- The `risk_score` logic depends on your `HIGH_RISK_OUTCOMES` mapping. If you do not set it, every match counts as high risk.

