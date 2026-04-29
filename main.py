import asyncio
import hashlib
import logging
import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

FORCE_HF_OFFLINE = os.getenv("FORCE_HF_OFFLINE", "true").lower() in {"1", "true", "yes"}

if FORCE_HF_OFFLINE:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import torch
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tldr_legal")

ANALYZE_TIMEOUT_SECONDS = int(os.getenv("ANALYZE_TIMEOUT_SECONDS", "45"))
CSV_PATH = os.getenv("LEGAL_CSV_PATH", "legal_text_classification.csv")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.65"))
BATCH_SIZE = int(os.getenv("SIMILARITY_BATCH_SIZE", "64"))
TEXT_CORPUS_DIR = os.getenv("TEXT_CORPUS_DIR", "text")
TEXT_CORPUS_MAX_FILES = int(os.getenv("TEXT_CORPUS_MAX_FILES", "300"))
TEXT_CORPUS_MAX_CHARS = int(os.getenv("TEXT_CORPUS_MAX_CHARS", "2000"))
TEXT_CORPUS_ENABLE = os.getenv("TEXT_CORPUS_ENABLE", "true").lower() in {"1", "true", "yes"}
EMBEDDINGS_CACHE_DIR = os.getenv("EMBEDDINGS_CACHE_DIR", ".embeddings_cache")
HIGH_RISK_OUTCOMES = {
    item.strip().lower()
    for item in os.getenv("HIGH_RISK_OUTCOMES", "").split(",")
    if item.strip()
}

def resolve_local_model_path(model_name: str) -> Optional[Path]:
    cache_root = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface"))
    
    # Transform model name to Hugging Face cache format (e.g., sentence-transformers/all-MiniLM-L6-v2 -> models--sentence-transformers--all-MiniLM-L6-v2)
    formatted_name = model_name.replace("/", "--")
    if "--" not in formatted_name:
        formatted_name = f"sentence-transformers--{formatted_name}"
    
    snapshot_root = cache_root / "hub" / f"models--{formatted_name}" / "snapshots"
    if not snapshot_root.exists():
        return None

    snapshots = [path for path in snapshot_root.iterdir() if path.is_dir()]
    if not snapshots:
        return None

    for snapshot in sorted(snapshots, reverse=True):
        if (snapshot / "modules.json").exists():
            return snapshot

    return sorted(snapshots, reverse=True)[0]

SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?\n]?")

app = FastAPI(title="TL;DR Legal", version="0.2.0")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)


data_store = None
load_error: Optional[str] = None


class AnalyzeRequest(BaseModel):
    text: str = Field(..., description="Full Terms of Service text")


class FoundFlag(BaseModel):
    user_sentence: str
    case_id: str
    case_title: str
    case_outcome: str
    case_text: str


class AnalyzeResponse(BaseModel):
    found_flags: List[FoundFlag]
    risk_score: int


@dataclass
class DataStore:
    dataframe: pd.DataFrame
    model: SentenceTransformer
    embeddings: np.ndarray


def split_text_units(text: str) -> List[str]:
    blocks = [block.strip() for block in re.split(r"\n{2,}", text) if block.strip()]
    units: List[str] = []
    for block in blocks:
        sentences = [sentence.strip() for sentence in SENTENCE_RE.findall(block) if sentence.strip()]
        if sentences:
            units.extend(sentences)
        else:
            units.append(block)
    return units


def load_data_store() -> DataStore:
    csv_path = Path(CSV_PATH)
    if not csv_path.is_absolute():
        csv_path = BASE_DIR / csv_path

    logger.info("Loading CSV dataset: %s", csv_path)
    dataframe = pd.read_csv(csv_path)
    required_columns = ["case_id", "case_outcome", "case_title", "case_text"]
    missing = [column for column in required_columns if column not in dataframe.columns]
    if missing:
        raise RuntimeError(f"CSV missing columns: {', '.join(missing)}")

    dataframe = dataframe[required_columns].dropna()
    dataframe["case_id"] = dataframe["case_id"].astype(str)
    dataframe["case_outcome"] = dataframe["case_outcome"].astype(str)
    dataframe["case_title"] = dataframe["case_title"].astype(str)
    dataframe["case_text"] = dataframe["case_text"].astype(str)

    if TEXT_CORPUS_ENABLE:
        corpus_dir = BASE_DIR / TEXT_CORPUS_DIR
        if corpus_dir.exists():
            corpus_rows = []
            for path in sorted(corpus_dir.glob("*.txt"))[:TEXT_CORPUS_MAX_FILES]:
                try:
                    with path.open("r", encoding="utf-8", errors="ignore") as handle:
                        content = handle.read(TEXT_CORPUS_MAX_CHARS + 1).strip()
                except Exception:
                    logger.warning("Failed to read text file: %s", path)
                    continue

                if not content:
                    continue

                corpus_rows.append(
                    {
                        "case_id": f"text:{path.stem}",
                        "case_outcome": "text-corpus",
                        "case_title": path.stem,
                        "case_text": content[:TEXT_CORPUS_MAX_CHARS],
                    }
                )

            if corpus_rows:
                dataframe = pd.concat([dataframe, pd.DataFrame(corpus_rows)], ignore_index=True)
                logger.info("Added %d text corpus files.", len(corpus_rows))
        else:
            logger.warning("Text corpus directory not found: %s", corpus_dir)

    model_source: str | Path = EMBED_MODEL_NAME
    if FORCE_HF_OFFLINE:
        local_model_path = resolve_local_model_path(EMBED_MODEL_NAME)
        if local_model_path is None:
            raise RuntimeError(
                "Offline mode is enabled, but the embedding model is not cached locally. "
                "Download it once while online, then restart with offline mode enabled."
            )
        model_source = local_model_path

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Embedding device: %s", device)
    model = SentenceTransformer(str(model_source), device=device)

    descriptions = dataframe["case_text"].tolist()
    
    cache_dir = BASE_DIR / EMBEDDINGS_CACHE_DIR
    cache_dir.mkdir(exist_ok=True)
    
    data_hash = hashlib.sha256(
        "".join(descriptions).encode("utf-8") + EMBED_MODEL_NAME.encode("utf-8")
    ).hexdigest()[:16]
    cache_file = cache_dir / f"embeddings_{data_hash}.pkl"
    
    if cache_file.exists():
        logger.info("Loading cached embeddings from %s", cache_file)
        with cache_file.open("rb") as f:
            embeddings = pickle.load(f)
    else:
        logger.info("Generating embeddings for %d rows...", len(descriptions))
        embeddings = model.encode(
            descriptions,
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        with cache_file.open("wb") as f:
            pickle.dump(embeddings, f)
        logger.info("Cached embeddings to %s", cache_file)

    logger.info("Loaded %d red-flag rows.", len(dataframe))
    return DataStore(dataframe=dataframe, model=model, embeddings=embeddings)


def outcome_is_high(outcome: str) -> bool:
    if not HIGH_RISK_OUTCOMES:
        return True
    normalized = outcome.strip().lower()
    return normalized in HIGH_RISK_OUTCOMES


def run_analysis(text: str) -> AnalyzeResponse:
    if data_store is None:
        raise RuntimeError("Dataset is not ready.")

    units = split_text_units(text)
    if not units:
        return AnalyzeResponse(found_flags=[], risk_score=0)

    found_flags: List[FoundFlag] = []
    risk_score = 0

    for start in range(0, len(units), BATCH_SIZE):
        batch_units = units[start : start + BATCH_SIZE]
        batch_embeddings = data_store.model.encode(
            batch_units,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        similarity = cosine_similarity(batch_embeddings, data_store.embeddings)

        for sentence_index, sentence in enumerate(batch_units):
            match_indices = np.where(similarity[sentence_index] >= SIMILARITY_THRESHOLD)[0]
            for match_idx in match_indices:
                row = data_store.dataframe.iloc[int(match_idx)]
                outcome = str(row["case_outcome"])
                if outcome_is_high(outcome):
                    risk_score += 1
                found_flags.append(
                    FoundFlag(
                        user_sentence=sentence,
                        case_id=str(row["case_id"]),
                        case_title=str(row["case_title"]),
                        case_outcome=outcome,
                        case_text=str(row["case_text"]),
                    )
                )

    return AnalyzeResponse(found_flags=found_flags, risk_score=risk_score)


@app.on_event("startup")
async def startup_event() -> None:
    global data_store, load_error

    try:
        data_store = await asyncio.to_thread(load_data_store)
        load_error = None
    except Exception as exc:
        data_store = None
        load_error = str(exc)
        logger.exception("Failed to load dataset or embeddings.")


@app.get("/", response_class=HTMLResponse)
def root() -> HTMLResponse:
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found.")
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(payload: AnalyzeRequest) -> AnalyzeResponse:
    if not payload.text or not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text must not be empty.")
    if data_store is None:
        detail = load_error or "Dataset is still loading or failed to load."
        raise HTTPException(status_code=503, detail=detail)

    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(run_analysis, payload.text),
            timeout=ANALYZE_TIMEOUT_SECONDS,
        )
        return result
    except asyncio.TimeoutError as exc:
        raise HTTPException(status_code=504, detail="Analysis timed out.") from exc
    except Exception as exc:
        logger.exception("Analysis failed.")
        raise HTTPException(status_code=500, detail="Analysis failed.") from exc
