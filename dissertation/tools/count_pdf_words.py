"""Estimate the WMG-counted dissertation words from the compiled PDF."""

from __future__ import annotations

import re
from pathlib import Path

from pypdf import PdfReader


PDF_PATH = Path(__file__).resolve().parents[1] / "main.pdf"
WORD_RE = re.compile(r"\b[A-Za-z0-9][A-Za-z0-9'’\-]*\b")


def count(text: str) -> int:
    return len(WORD_RE.findall(text))


def main() -> None:
    pages = [page.extract_text() or "" for page in PdfReader(str(PDF_PATH)).pages]
    starts = {
        "Abstract": next(i for i, text in enumerate(pages) if text.lstrip().startswith("Abstract")),
        "References": next(
            i for i, text in enumerate(pages) if text.lstrip().startswith("References")
        ),
    }
    starts.update(
        {
            f"Chapter {number}": next(
                i for i, text in enumerate(pages) if f"Chapter {number}:" in text
            )
            for number in range(1, 8)
        }
    )

    counts = {"Abstract": count(pages[starts["Abstract"]])}
    for number in range(1, 8):
        start = starts[f"Chapter {number}"]
        end = starts[f"Chapter {number + 1}"] if number < 7 else starts["References"]
        counts[f"Chapter {number}"] = count("\n".join(pages[start:end]))

    for label, value in counts.items():
        print(f"{label}: {value:,}")
    print(f"Total: {sum(counts.values()):,}")


if __name__ == "__main__":
    main()
