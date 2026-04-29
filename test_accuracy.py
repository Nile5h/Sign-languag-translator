"""
Accuracy test: runs the analysis pipeline directly (no HTTP server needed)
against 5 sample files from the text/ folder and prints a summary report.
"""
import time
from pathlib import Path

# Bootstrap the app module so data_store is populated synchronously
import asyncio
import sys

sys.path.insert(0, str(Path(__file__).parent))

import main as app_module

TEST_FILES = [
    "Twitter_TermsofService.txt",
    "YouTube_TermsofService.txt",
    "Uber_TermsofService.txt",
    "WhatsApp_TermsofService.txt",
    "Zoom VideoCommunications_TermsofService.txt",
]

# Fallback: pick first 5 .txt files if any of the above are missing
TEXT_DIR = Path(__file__).parent / "text"


def pick_test_files() -> list[Path]:
    chosen = []
    for name in TEST_FILES:
        p = TEXT_DIR / name
        if p.exists():
            chosen.append(p)
    if len(chosen) < 5:
        extras = [f for f in sorted(TEXT_DIR.glob("*_TermsofService.txt")) if f not in chosen]
        chosen.extend(extras[: 5 - len(chosen)])
    return chosen[:5]


def load() -> None:
    print("Loading data store (may take a while on first run)...\n")
    asyncio.run(app_module.startup_event())
    if app_module.data_store is None:
        print(f"ERROR: {app_module.load_error}")
        sys.exit(1)
    print(f"Data store ready. Index size: {len(app_module.data_store.dataframe)} rows\n")
    print("-" * 70)


def test_file(path: Path, threshold: float = None) -> dict:
    text = path.read_text(encoding="utf-8", errors="ignore")
    original = app_module.SIMILARITY_THRESHOLD
    if threshold is not None:
        app_module.SIMILARITY_THRESHOLD = threshold
    t0 = time.perf_counter()
    result = app_module.run_analysis(text)
    elapsed = time.perf_counter() - t0
    app_module.SIMILARITY_THRESHOLD = original

    outcomes: dict[str, int] = {}
    for flag in result.found_flags:
        outcomes[flag.case_outcome] = outcomes.get(flag.case_outcome, 0) + 1

    return {
        "file": path.name,
        "chars": len(text),
        "flags": len(result.found_flags),
        "risk_score": result.risk_score,
        "outcomes": outcomes,
        "elapsed_s": elapsed,
    }


THRESHOLDS = [0.75, 0.65, 0.55]


def main() -> None:
    load()
    files = pick_test_files()
    if not files:
        print("No test files found in text/ folder.")
        sys.exit(1)

    for threshold in THRESHOLDS:
        print(f"\n{'='*70}")
        print(f"THRESHOLD: {threshold}")
        print(f"{'='*70}")
        total_flags = 0
        total_risk = 0
        files_with_matches = 0

        for path in files:
            r = test_file(path, threshold=threshold)
            total_flags += r["flags"]
            total_risk += r["risk_score"]
            if r["flags"] > 0:
                files_with_matches += 1

            print(f"  {r['file']}")
            print(f"    size={r['chars']:,}  flags={r['flags']}  risk={r['risk_score']}  time={r['elapsed_s']:.2f}s")
            if r["outcomes"]:
                print(f"    outcomes: {r['outcomes']}")

        print(f"\n  SUMMARY | flags={total_flags} | risk={total_risk} | match_rate={files_with_matches}/{len(files)} ({100*files_with_matches//len(files)}%)")
        print(f"  Avg flags/file: {total_flags/len(files):.1f}")


if __name__ == "__main__":
    main()
