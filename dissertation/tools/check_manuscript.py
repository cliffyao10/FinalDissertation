"""Deterministic checks for the dissertation source tree."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    tex_files = sorted(ROOT.rglob("*.tex"))
    combined = "\n".join(path.read_text(encoding="utf-8") for path in tex_files)
    bib_text = (ROOT / "references.bib").read_text(encoding="utf-8")

    cited: set[str] = set()
    for match in re.finditer(r"\\(?:paren|text)?cite\{([^}]+)\}", combined):
        cited.update(key.strip() for key in match.group(1).split(","))

    entries = list(
        re.finditer(
            r"@[A-Za-z]+\{(?P<key>[^,]+),(?P<body>.*?)(?=\n@|\Z)",
            bib_text,
            flags=re.DOTALL,
        )
    )
    defined = {match.group("key").strip() for match in entries}
    missing = sorted(cited - defined)
    if missing:
        raise SystemExit(f"Missing bibliography entries: {missing}")

    without_link = []
    for match in entries:
        body = match.group("body")
        if not re.search(r"\n\s*(?:doi|url)\s*=", body, flags=re.IGNORECASE):
            without_link.append(match.group("key").strip())
    if without_link:
        raise SystemExit(f"Bibliography entries without DOI/URL: {without_link}")

    for path in tex_files:
        text = path.read_text(encoding="utf-8")
        stripped = re.sub(r"(?m)%.*$", "", text)
        if stripped.count("{") != stripped.count("}"):
            raise SystemExit(f"Unbalanced braces: {path.relative_to(ROOT)}")

    introduction = (ROOT / "chapters" / "01_introduction.tex").read_text(
        encoding="utf-8"
    )
    prose = re.sub(r"(?m)%.*$", "", introduction)
    prose = re.sub(r"\\[A-Za-z]+(?:\[[^]]*\])?", " ", prose)
    prose = re.sub(r"[{}]", " ", prose)
    words = re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)*", prose)

    placeholders = sum(text.count("[To be drafted") for text in [combined])
    print(f"TeX files: {len(tex_files)}")
    print(f"Bibliography entries: {len(entries)}")
    print(f"Citation keys used: {len(cited)}")
    print(f"Approximate Chapter 1 words: {len(words)}")
    print(f"Expected future-chapter placeholders: {placeholders}")
    print("Manuscript source checks passed.")


if __name__ == "__main__":
    main()
