# Cove dissertation source

This directory is the working LaTeX source for the WMG MSc dissertation. It
reproduces the structure of the supplied 2025--26 WMG Word template while
applying the published presentation rules: 12-point text, 1.5 line spacing,
lower-right page numbering and a new page for every chapter. WMG does not
mandate Verdana. It is currently selected only because it is the default font
found in the supplied Word template and can be changed in `main.tex`.

## Build

The project has a local portable MiKTeX installation under `../.tools/miktex`.
From the repository root, build the dissertation with:

```text
powershell -ExecutionPolicy Bypass -File .\dissertation\build.ps1
```

The script runs XeLaTeX, Biber and the repeated passes required to update the
contents pages, figure and table lists, citations and cross-references. The PDF
is written to `dissertation/main.pdf`.

The main editable identity fields and provisional title are near the top of
`main.tex`. Chapters are separate files under `chapters/`; bibliography records
belong in `references.bib` and are cited with `\parencite{citation-key}`.

## Source policy

- English dissertation text is stored under `chapters/`.
- Non-submission Chinese explanations are stored under `chinese-notes/`.
- New dissertation prose follows a review gate: first present matching English
  and Chinese drafts in the conversation, revise them with the student, and
  edit the submission source only after the student explicitly approves them.
- Every factual research claim must be supported by a cited primary source or
  a versioned project result.
- `references.bib` is cumulative. Metadata must be verified against Crossref,
  the publisher or the official paper page before an entry is added.
- Generated academic-result images are prohibited. Result charts must be
  derived from versioned files under `../results/`.

## Overleaf hand-off

The source uses repository-relative paths and is portable. Run
`export_overleaf.ps1` to create `Cove-dissertation-overleaf.zip`, upload that
ZIP as a new Overleaf project, set `main.tex` as the main document and select
XeLaTeX as the compiler. The export intentionally excludes local build files
and the non-submission Chinese notes.

## Provisional word budget

The managed count includes titles, the abstract, summaries, quotations,
in-text citations, footnotes, endnotes, tables, figures and captions. It
excludes the contents/list pages, reference list, bibliography and appendices.

| Counted material | Target words |
|---|---:|
| Abstract | 250 |
| Chapter 1: Introduction | 1,450 |
| Chapter 2: Literature Review | 3,400 |
| Chapter 3: Methodology | 2,800 |
| Chapter 4: Results | 1,500 |
| Chapter 5: Analysis | 1,700 |
| Chapter 6: Discussion | 2,200 |
| Chapter 7: Conclusion | 650 |
| Titles, tables, figures, captions and citation overhead reserve | 750 |
| Target counted total | 14,700 |

The permitted 10% range around 15,000 words is 13,500--16,500 words. The
14,700-word working target leaves a small safety margin for corrections while
remaining close to the indicative limit.

## Figure rule

WMG states that text within an image is not permitted. Tables and diagrams
should therefore be produced as native LaTeX content where possible. Mermaid
source may be used to plan a diagram, but the submission version should be
implemented in TikZ/PGF or another form that keeps labels as selectable text.
UI screenshots require specific confirmation because interface labels are
unavoidably embedded in the screenshot.
