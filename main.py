import asyncio
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

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
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.75"))
BATCH_SIZE = int(os.getenv("SIMILARITY_BATCH_SIZE", "64"))

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
    kaggle_title: str
    kaggle_rating: str
    kaggle_explanation: str


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
    required_columns = ["title", "rating", "point_description"]
    missing = [column for column in required_columns if column not in dataframe.columns]
    if missing:
        raise RuntimeError(f"CSV missing columns: {', '.join(missing)}")

    dataframe = dataframe[required_columns].dropna()
    dataframe["title"] = dataframe["title"].astype(str)
    dataframe["rating"] = dataframe["rating"].astype(str)
    dataframe["point_description"] = dataframe["point_description"].astype(str)

    model = SentenceTransformer(EMBED_MODEL_NAME)
    descriptions = dataframe["point_description"].tolist()
    embeddings = model.encode(
        descriptions,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    logger.info("Loaded %d red-flag rows.", len(dataframe))
    return DataStore(dataframe=dataframe, model=model, embeddings=embeddings)


def rating_is_high(rating: str) -> bool:
    normalized = rating.strip().lower()
    return normalized.startswith("high")


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
                rating = str(row["rating"])
                if rating_is_high(rating):
                    risk_score += 1
                found_flags.append(
                    FoundFlag(
                        user_sentence=sentence,
                        kaggle_title=str(row["title"]),
                        kaggle_rating=rating,
                        kaggle_explanation=str(row["point_description"]),
                    )
                )

    return AnalyzeResponse(found_flags=found_flags, risk_score=risk_score)


@app.on_event("startup")
async def startup_event() -> None:
    global data_store, load_error

    try:
        data_store = load_data_store()
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
