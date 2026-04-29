"""
TL;DR Legal — RAG pipeline CLI
Usage:
    python main.py <input.txt|folder/> [--threshold 0.70] [--output report.json|report.txt]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import pickle
import io

class _Unpickler(pickle.Unpickler):
    """Remaps __main__.KBEntry -> main.KBEntry so cache works whether
    the file was saved via CLI (__main__) or imported as a module."""
    def find_class(self, module, name):
        if module == "__main__":
            module = "main"
        return super().find_class(module, name)

import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# ── offline guard ────────────────────────────────────────────────────────────
FORCE_HF_OFFLINE = os.getenv("FORCE_HF_OFFLINE", "true").lower() in {"1", "true", "yes"}
if FORCE_HF_OFFLINE:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import faiss
import numpy as np
import ollama
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pypdf import PdfReader
from rich.console import Console
from rich.table import Table
from rich import box
from sentence_transformers import SentenceTransformer

console = Console()

# ── paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
PARQUET_PATH = BASE_DIR / "train-00000-of-00001-a8de7efe0da36666.parquet"
CASES_PATH   = BASE_DIR / "15012282" / "cases.csv"
TOPICS_PATH  = BASE_DIR / "15012282" / "topics.csv"
CACHE_DIR    = BASE_DIR / ".embeddings_cache"
CACHE_DIR.mkdir(exist_ok=True)

EMBED_MODEL  = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
LLM_MODEL    = os.getenv("LLM_MODEL",   "llama3.2:1b")

# Claudette label → severity mapping
# label 0 = unfair clause (HIGH), label 1 = fair/neutral (LOW)
CLAUDETTE_SEVERITY = {0: "High", 1: "Low"}

# ToS;DR classification → severity
TOSDR_SEVERITY = {
    "blocker": "Critical",
    "bad":     "High",
    "neutral": "Medium",
    "good":    "Low",
}


# ── data classes ─────────────────────────────────────────────────────────────
@dataclass
class KBEntry:
    source: str          # "claudette" | "tosdr"
    text: str            # text used for embedding
    label: str           # severity label string
    severity: str        # Critical / High / Medium / Low
    category: str        # topic title or "Unfairness"
    description: str     # plain-English description


@dataclass
class Match:
    paragraph: str
    entry: KBEntry
    score: float


@dataclass
class AnalysisResult:
    file: str
    matches: list[Match]
    summary: str
    elapsed: float


# ── knowledge base ────────────────────────────────────────────────────────────
def _cache_key(tag: str) -> Path:
    return CACHE_DIR / f"kb_{tag}_{EMBED_MODEL.replace('/', '_')}.pkl"


def _resolve_model() -> str | Path:
    """Return local snapshot path when offline, else model name."""
    if not FORCE_HF_OFFLINE:
        return EMBED_MODEL
    cache_root = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface"))
    name = EMBED_MODEL.replace("/", "--")
    if "--" not in name:
        name = f"sentence-transformers--{name}"
    snap_root = cache_root / "hub" / f"models--{name}" / "snapshots"
    if not snap_root.exists():
        raise RuntimeError(f"Model not cached locally: {EMBED_MODEL}")
    snaps = sorted(snap_root.iterdir(), reverse=True)
    for s in snaps:
        if (s / "modules.json").exists():
            return s
    return snaps[0]


def load_knowledge_base() -> tuple[list[KBEntry], faiss.IndexFlatIP]:
    cache = _cache_key("full")
    if cache.exists():
        console.log("[dim]Loading cached knowledge base...[/dim]")
        with cache.open("rb") as f:
            entries, embeddings = _Unpickler(f).load()
        index = _build_index(embeddings)
        console.log(f"[green]KB ready[/green] — {len(entries)} entries")
        return entries, index

    console.log("[bold]Building knowledge base from datasets...[/bold]")
    entries: list[KBEntry] = []

    # ── Source A: Claudette ──────────────────────────────────────────────────
    df_c = pd.read_parquet(PARQUET_PATH)
    for _, row in df_c.iterrows():
        lbl = int(row["label"])
        entries.append(KBEntry(
            source="claudette",
            text=str(row["text"]),
            label=str(lbl),
            severity=CLAUDETTE_SEVERITY.get(lbl, "Medium"),
            category="Unfairness Detector",
            description=str(row["text"]),
        ))

    # ── Source B: ToS;DR ─────────────────────────────────────────────────────
    df_cases  = pd.read_csv(CASES_PATH)
    df_topics = pd.read_csv(TOPICS_PATH)[["id", "title"]].rename(
        columns={"id": "topic_id", "title": "topic_title"}
    )
    df_tosdr = df_cases.merge(df_topics, on="topic_id", how="left")
    df_tosdr["topic_title"] = df_tosdr["topic_title"].fillna("General")
    df_tosdr["description"] = df_tosdr["description"].fillna("").str.replace(r"\r\n", " ", regex=True).str.strip()

    for _, row in df_tosdr.iterrows():
        clf = str(row.get("classification", "neutral")).lower()
        embed_text = str(row["title"])
        entries.append(KBEntry(
            source="tosdr",
            text=embed_text,
            label=clf,
            severity=TOSDR_SEVERITY.get(clf, "Medium"),
            category=str(row["topic_title"]),
            description=str(row["description"]) or embed_text,
        ))

    # ── Embed ────────────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    console.log(f"Embedding {len(entries)} entries on [cyan]{device}[/cyan]...")
    model = SentenceTransformer(str(_resolve_model()), device=device)
    texts = [e.text for e in entries]
    embeddings = model.encode(
        texts, batch_size=64, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=True,
    )

    with cache.open("wb") as f:
        pickle.dump((entries, embeddings), f, protocol=4)

    index = _build_index(embeddings)
    console.log(f"[green]KB ready[/green] — {len(entries)} entries")
    return entries, index


def _build_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)          # inner product == cosine on normalised vecs
    index.add(embeddings.astype("float32"))
    return index


# ── chunking ──────────────────────────────────────────────────────────────────
def chunk_text(text: str) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    # split very long paragraphs at sentence boundaries
    result: list[str] = []
    for para in paragraphs:
        if len(para) > 800:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            result.extend(s.strip() for s in sentences if s.strip())
        else:
            result.append(para)
    return result


# ── analysis engine ───────────────────────────────────────────────────────────
def analyse_text(
    text: str,
    model: SentenceTransformer,
    index: faiss.IndexFlatIP,
    entries: list[KBEntry],
    threshold: float,
) -> list[Match]:
    paragraphs = chunk_text(text)
    if not paragraphs:
        return []

    para_embs = model.encode(
        paragraphs, batch_size=64, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=False,
    ).astype("float32")

    # search top-20 candidates per paragraph
    scores_matrix, idx_matrix = index.search(para_embs, 20)

    matches: list[Match] = []
    for p_idx, paragraph in enumerate(paragraphs):
        # group by category, keep best score per category
        best: dict[str, Match] = {}
        for rank in range(scores_matrix.shape[1]):
            score = float(scores_matrix[p_idx, rank])
            if score < threshold:
                break
            kb_idx = int(idx_matrix[p_idx, rank])
            entry = entries[kb_idx]
            cat = entry.category
            if cat not in best or score > best[cat].score:
                best[cat] = Match(paragraph=paragraph, entry=entry, score=score)
        matches.extend(best.values())

    # sort by score descending
    matches.sort(key=lambda m: m.score, reverse=True)
    return matches


# ── LLM summarisation ─────────────────────────────────────────────────────────
def summarise_with_llm(matches: list[Match]) -> str:
    if not matches:
        return "No significant legal risks detected."

    top = matches[:5]
    snippets = "\n".join(
        f"- [{m.entry.severity}] {m.entry.category}: {m.entry.description[:200]}"
        for m in top
    )
    prompt = (
        "You are a legal risk analyst. Summarize these specific legal risks "
        "into exactly 3 bullet points in plain English for a regular user. "
        "Be brief and blunt. No preamble.\n\n"
        f"{snippets}"
    )
    try:
        resp = ollama.chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"num_predict": 256},
        )
        return resp["message"]["content"].strip()
    except Exception as exc:
        return f"[LLM unavailable: {exc}]"


# ── rich console output ───────────────────────────────────────────────────────
SEVERITY_COLOR = {
    "Critical": "bold red",
    "High":     "red",
    "Medium":   "yellow",
    "Low":      "green",
}


def print_results(result: AnalysisResult) -> None:
    console.rule(f"[bold]{result.file}[/bold]")

    if not result.matches:
        console.print("[green]No risks found above threshold.[/green]")
        return

    table = Table(box=box.ROUNDED, show_lines=True, expand=True)
    table.add_column("Sev",      style="bold", width=8)
    table.add_column("Category", width=20)
    table.add_column("Source",   width=10)
    table.add_column("Score",    width=6)
    table.add_column("Matched paragraph (truncated)", no_wrap=False)
    table.add_column("KB description (truncated)",    no_wrap=False)

    for m in result.matches[:20]:
        color = SEVERITY_COLOR.get(m.entry.severity, "white")
        table.add_row(
            f"[{color}]{m.entry.severity}[/{color}]",
            m.entry.category,
            m.entry.source,
            f"{m.score:.2f}",
            m.paragraph[:120] + ("…" if len(m.paragraph) > 120 else ""),
            m.entry.description[:120] + ("…" if len(m.entry.description) > 120 else ""),
        )

    console.print(table)
    console.print(f"\n[bold cyan]Plain-English Summary[/bold cyan]")
    console.print(result.summary)
    console.print(f"\n[dim]Analysed in {result.elapsed:.2f}s | {len(result.matches)} flags[/dim]\n")


# ── output serialisation ──────────────────────────────────────────────────────
def export_json(results: list[AnalysisResult], path: Path) -> None:
    data = []
    for r in results:
        data.append({
            "file": r.file,
            "elapsed_s": round(r.elapsed, 2),
            "summary": r.summary,
            "flags": [
                {
                    "severity":    m.entry.severity,
                    "category":    m.entry.category,
                    "source":      m.entry.source,
                    "score":       round(m.score, 4),
                    "paragraph":   m.paragraph,
                    "description": m.entry.description,
                }
                for m in r.matches
            ],
        })
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"[green]JSON report saved:[/green] {path}")


def export_txt(results: list[AnalysisResult], path: Path) -> None:
    lines: list[str] = []
    for r in results:
        lines.append(f"{'='*70}")
        lines.append(f"FILE: {r.file}  ({len(r.matches)} flags, {r.elapsed:.2f}s)")
        lines.append(f"{'='*70}")
        lines.append("PLAIN-ENGLISH SUMMARY:")
        lines.append(r.summary)
        lines.append("")
        lines.append("DETAILED FLAGS:")
        for m in r.matches:
            lines.append(f"  [{m.entry.severity}] {m.entry.category} (score={m.score:.2f})")
            lines.append(f"    Paragraph : {m.paragraph[:200]}")
            lines.append(f"    KB entry  : {m.entry.description[:200]}")
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    console.print(f"[green]Text report saved:[/green] {path}")


# ── FastAPI web server ────────────────────────────────────────────────────────
app = FastAPI(title="TL;DR Legal", version="0.3.0")
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)

_kb: tuple[list[KBEntry], faiss.IndexFlatIP] | None = None
_model: SentenceTransformer | None = None


class WebRequest(BaseModel):
    text: str
    threshold: float = 0.70


@app.on_event("startup")
async def _startup() -> None:
    global _kb, _model
    _kb = await asyncio.to_thread(load_knowledge_base)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _model = SentenceTransformer(str(_resolve_model()), device=device)


@app.get("/", response_class=HTMLResponse)
def _root() -> HTMLResponse:
    p = STATIC_DIR / "index.html"
    if not p.exists():
        raise HTTPException(404, "index.html not found")
    return HTMLResponse(p.read_text(encoding="utf-8"))


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.post("/upload-pdf")
async def _upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted.")
    contents = await file.read()
    if len(contents) > 20 * 1024 * 1024:  # 20 MB cap
        raise HTTPException(413, "PDF exceeds 20 MB limit.")
    try:
        reader = PdfReader(io.BytesIO(contents))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n\n".join(pages).strip()
    except Exception as exc:
        raise HTTPException(422, f"Could not parse PDF: {exc}")
    if not text:
        raise HTTPException(422, "No extractable text found in this PDF.")
    return {"text": text, "pages": len(reader.pages)}


@app.post("/analyze")
async def _analyze(payload: WebRequest):
    if not payload.text.strip():
        raise HTTPException(400, "Text must not be empty.")
    if _kb is None:
        raise HTTPException(503, "Knowledge base is still loading.")
    entries, index = _kb

    def _run():
        matches = analyse_text(payload.text, _model, index, entries, payload.threshold)
        summary = summarise_with_llm(matches)
        return matches, summary

    matches, summary = await asyncio.to_thread(_run)
    return {
        "summary": summary,
        "risk_score": sum(1 for m in matches if m.entry.severity in {"Critical", "High"}),
        "flags": [
            {
                "severity":    m.entry.severity,
                "category":    m.entry.category,
                "source":      m.entry.source,
                "score":       round(m.score, 4),
                "paragraph":   m.paragraph,
                "description": m.entry.description,
            }
            for m in matches
        ],
    }


# ── CLI entry point ────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="TL;DR Legal — RAG-based Terms of Service analyser"
    )
    parser.add_argument("input", help="Path to a .txt file or a folder of .txt files")
    parser.add_argument("--threshold", type=float, default=0.70,
                        help="Cosine similarity threshold (default: 0.70)")
    parser.add_argument("--output", default=None,
                        help="Output file path (.json or .txt). Omit to print only.")
    args = parser.parse_args()

    input_path = Path(args.input)
    if input_path.is_dir():
        files = sorted(input_path.glob("*.txt"))
    elif input_path.is_file():
        files = [input_path]
    else:
        console.print(f"[red]Input not found:[/red] {input_path}")
        sys.exit(1)

    if not files:
        console.print("[yellow]No .txt files found.[/yellow]")
        sys.exit(0)

    # load KB
    entries, index = load_knowledge_base()

    # load embed model (reuse same instance for analysis)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(str(_resolve_model()), device=device)

    results: list[AnalysisResult] = []

    for txt_file in files:
        t0 = time.perf_counter()
        text = txt_file.read_text(encoding="utf-8", errors="ignore")
        matches = analyse_text(text, model, index, entries, args.threshold)
        summary = summarise_with_llm(matches)
        elapsed = time.perf_counter() - t0

        result = AnalysisResult(
            file=txt_file.name,
            matches=matches,
            summary=summary,
            elapsed=elapsed,
        )
        results.append(result)
        print_results(result)

    # export
    if args.output:
        out_path = Path(args.output)
        if out_path.suffix == ".txt":
            export_txt(results, out_path)
        else:
            export_json(results, out_path)


if __name__ == "__main__":
    main()
