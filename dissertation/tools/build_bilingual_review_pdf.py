"""Build a paragraph-aligned English-Chinese review PDF from the compiled thesis.

The English side is extracted from the already compiled submission PDF so that
citations and reported values match the rendered dissertation.  Chinese text is
an aid for the author's review and is never written back to the submission TeX.
"""

from __future__ import annotations

import html
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE_PDF = ROOT / "dissertation" / "main.pdf"
GUIDE_MD = ROOT / "dissertation" / "review_drafts" / "figures_and_tables_bilingual_guide.md"
OUTPUT_DIR = ROOT / "output" / "pdf"
OUTPUT_PDF = OUTPUT_DIR / "Cove_full_bilingual_review.pdf"
CACHE_PATH = ROOT / "tmp" / "pdfs" / "bilingual_translation_cache_nllb.json"
AUDIT_PATH = ROOT / "tmp" / "pdfs" / "bilingual_review_audit.json"

# The desktop runtime provides PDF authoring packages, while the project venv
# provides transformers and torch.
BUNDLED_PYTHON = Path(
    r"C:\Users\Cliff\.cache\codex-runtimes\codex-primary-runtime\dependencies\python"
)
for dependency_path in (BUNDLED_PYTHON / "Lib" / "site-packages", BUNDLED_PYTHON):
    if str(dependency_path) not in sys.path:
        sys.path.insert(0, str(dependency_path))

import pdfplumber  # noqa: E402
import torch  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_CENTER, TA_LEFT  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    BaseDocTemplate,
    Frame,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
)
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # noqa: E402


@dataclass
class Block:
    english: str
    kind: str = "paragraph"
    chinese: str = ""


HEADING_RE = re.compile(
    r"^(?:Chapter\s+\d+:|\d+(?:\.\d+)+\s+|Abstract$|Declaration$|"
    r"Acknowledgements$|References$|Appendix\s+[A-Z]|List of |Contents$)",
    re.IGNORECASE,
)


def clean_join(lines: list[str]) -> str:
    text = ""
    for raw in lines:
        line = re.sub(r"\s+", " ", raw.strip())
        if not line:
            continue
        if text.endswith("-") and line[:1].islower():
            text += line
        else:
            text += (" " if text else "") + line
    text = text.replace(";", ";").replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", text).strip()


def extract_blocks() -> tuple[list[Block], list[Block]]:
    blocks: list[Block] = []
    references: list[Block] = []
    with pdfplumber.open(SOURCE_PDF) as pdf:
        # Pro-forma through acknowledgements, then dissertation body through appendices.
        selected = list(range(0, 5)) + list(range(11, len(pdf.pages)))
        in_references = False
        for page_index in selected:
            lines = pdf.pages[page_index].extract_text_lines()
            lines = [x for x in lines if not re.fullmatch(r"[ivxlcdm]+|\d+", x["text"].strip(), re.I)]
            groups: list[list[str]] = []
            current: list[str] = []
            previous_top: float | None = None
            for entry in lines:
                text = entry["text"].strip()
                top = float(entry["top"])
                is_heading = bool(HEADING_RE.match(text))
                gap = top - previous_top if previous_top is not None else 99.0
                if current and (gap > 22.0 or is_heading):
                    groups.append(current)
                    current = []
                current.append(text)
                previous_top = top
            if current:
                groups.append(current)

            for group in groups:
                text = clean_join(group)
                if not text or text in {"Cove", "WMG"}:
                    continue
                # PDF text extraction flattens TikZ labels and longtable cells.
                # Keep their captions and prose descriptions, but omit the
                # unreadable raw drawing/table text; the bilingual guide at the
                # end explains each visual in full.
                if (
                    len(text) > 1500
                    or re.search(r"ID Feature Userstory Acceptancecriteria", text, re.I)
                    or re.search(r"WardrobeStore version pieces\[\] outfits\[\]", text, re.I)
                    or re.search(r"last_search_stats float slot_dim", text, re.I)
                    or re.search(r"artifacts_available|occupied_slots_for|score_preprojected|FEATURE_WEIGHTS", text, re.I)
                    or re.search(r"Candidate\s*search.*Trained|Modelcheckpoint.*ExternalOpen", text, re.I)
                ):
                    continue
                kind = "heading" if HEADING_RE.match(text) else "paragraph"
                if text.lower() == "references":
                    in_references = True
                    references.append(Block(text, "heading"))
                    continue
                if in_references and re.match(r"^(?:Appendix|Chapter)\s+[A-Z]", text, re.I):
                    in_references = False
                target = references if in_references else blocks
                target.append(Block(text, kind))
    return merge_page_continuations(blocks), references


def merge_page_continuations(blocks: list[Block]) -> list[Block]:
    merged: list[Block] = []
    for block in blocks:
        if (
            merged
            and block.kind == "paragraph"
            and merged[-1].kind == "paragraph"
            and not re.search(r"[.!?:;\]\)]$", merged[-1].english)
        ):
            merged[-1].english = clean_join([merged[-1].english, block.english])
        else:
            merged.append(block)
    return merged


def split_for_model(text: str, max_chars: int = 720) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(])", text)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= max_chars:
            current = f"{current} {sentence}".strip()
        else:
            if current:
                chunks.append(current)
            if len(sentence) <= max_chars:
                current = sentence
            else:
                pieces = [sentence[i : i + max_chars] for i in range(0, len(sentence), max_chars)]
                chunks.extend(pieces[:-1])
                current = pieces[-1]
    if current:
        chunks.append(current)
    return chunks


def postprocess_chinese(text: str) -> str:
    replacements = {
        "装备": "穿搭",
        "候选人": "候选",
        "建议": "推荐",
        "推荐书": "推荐结果",
        "服装制成": "服装表示",
        "各州": "这些状态",
        "二级多式培训数据": "二手 Polyvore 训练数据",
        "独立于产品": "与具体产品无关",
        "确切搜索": "精确搜索",
        "确切的": "精确的",
        "艺术事实": "研究产物",
        "年龄": "管理",
        "考古方法": "架构",
        "完全适合的建议": "完整穿搭推荐",
        "审查": "综述",
        "一般审计": "泛化审计",
        "服装兼容": "穿搭兼容",
        "衣服兼容": "穿搭兼容",
        "填空任务": "填空（FITB）任务",
        "视觉语言": "视觉-语言",
        "服装范围限制": "服装范围约束",
        "候选搜索": "候选搜索",
        "西格利普": "SigLIP",
        "Siglip": "SigLIP",
        "Polyvore服装": "Polyvore 穿搭",
        "二级Polyvore": "二手 Polyvore",
        "结的视觉": "冻结的视觉",
        "推系统": "推荐系统",
        "完整服装推荐": "完整穿搭推荐",
        "服装推荐": "穿搭推荐",
        "服装推": "穿搭推荐",
        "冷的": "冻结的",
        "重流": "循环",
        "类型意识": "类型感知",
        "减排": "消融",
        "过度适应": "过拟合",
        "档案结构": "架构",
        "排放": "消融",
        "通用化": "泛化",
        "文本背景": "模型产物",
        "外国密钥": "外键",
        "插槽": "位置",
        "作物": "裁剪图",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"\s+([，。；：！？])", r"\1", text)
    return text.strip()


def preserve_evidence_tokens(english: str, chinese: str) -> str:
    """Keep reported values and citations visible if a translation drops them."""
    citations = re.findall(r"\([^()]{0,180}(?:19|20)\d{2}[^()]{0,80}\)", english)
    missing_citations = [item for item in citations if item not in chinese]
    numeric_source = re.sub(r"\([^()]{0,180}(?:19|20)\d{2}[^()]{0,80}\)", "", english)
    numbers = sorted(set(re.findall(r"(?<![A-Za-z])\d+(?:[.,]\d+)*%?", numeric_source)))
    missing_numbers = [item for item in numbers if item not in chinese]
    extras = []
    if missing_citations:
        extras.append("原文引用：" + "；".join(missing_citations))
    if missing_numbers:
        extras.append("原文数值：" + "、".join(missing_numbers))
    if extras:
        chinese = chinese.rstrip("。；") + "（" + "；".join(extras) + "。）"
    return chinese


def translate_blocks(blocks: list[Block]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    cache = json.loads(CACHE_PATH.read_text(encoding="utf-8")) if CACHE_PATH.exists() else {}
    pending: list[tuple[Block, list[str]]] = []
    for block in blocks:
        if block.english in cache:
            translated = postprocess_chinese(cache[block.english])
            block.chinese = preserve_evidence_tokens(block.english, translated) if block.kind == "paragraph" else translated
        else:
            pending.append((block, split_for_model(block.english)))

    if not pending:
        return

    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="eng_Latn", local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, local_files_only=True)
    model.eval()

    flat: list[str] = []
    ownership: list[tuple[Block, int]] = []
    for block, chunks in pending:
        ownership.append((block, len(chunks)))
        flat.extend(chunks)

    translated: list[str] = []
    batch_size = 16
    with torch.inference_mode():
        for start in range(0, len(flat), batch_size):
            batch = flat[start : start + batch_size]
            encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            generated = model.generate(
                **encoded,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids("zho_Hans"),
                max_new_tokens=220,
                num_beams=1,
            )
            translated.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
            done = min(start + batch_size, len(flat))
            print(f"translated_chunks={done}/{len(flat)}", flush=True)

    cursor = 0
    for block, count in ownership:
        translated = postprocess_chinese("".join(translated[cursor : cursor + count]))
        block.chinese = preserve_evidence_tokens(block.english, translated) if block.kind == "paragraph" else translated
        cache[block.english] = block.chinese
        cursor += count
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def diagram_guide_blocks() -> list[Block]:
    if not GUIDE_MD.exists():
        return []
    blocks: list[Block] = [Block("Bilingual guide to figures and tables", "heading", "图表双语说明")]
    heading = ""
    english = ""
    for line in GUIDE_MD.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            heading = line[3:].strip()
            if heading not in {"Review note / 审阅提示"}:
                en, _, zh = heading.partition(" / ")
                blocks.append(Block(en, "heading", zh))
        elif line.startswith("**English.**"):
            english = line.replace("**English.**", "", 1).strip()
        elif line.startswith("**中文。**") and english:
            chinese = line.replace("**中文。**", "", 1).strip()
            blocks.append(Block(english, "paragraph", chinese))
            english = ""
    return blocks


def safe_markup(text: str) -> str:
    return html.escape(text).replace("\n", "<br/>")


class NumberedDocTemplate(BaseDocTemplate):
    def __init__(self, filename: str, **kwargs):
        super().__init__(filename, **kwargs)
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="normal")
        self.addPageTemplates(PageTemplate(id="all", frames=frame, onPage=self.draw_page))

    def draw_page(self, canvas, doc):
        canvas.saveState()
        canvas.setFont("Arial", 8)
        canvas.setFillColor(colors.HexColor("#6F7378"))
        canvas.drawString(18 * mm, 10 * mm, "Cove bilingual review copy")
        canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, str(doc.page))
        canvas.restoreState()


def build_pdf(blocks: list[Block], references: list[Block]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdfmetrics.registerFont(TTFont("Arial", r"C:\Windows\Fonts\arial.ttf"))
    pdfmetrics.registerFont(TTFont("ArialBold", r"C:\Windows\Fonts\arialbd.ttf"))
    pdfmetrics.registerFont(TTFont("SimHei", r"C:\Windows\Fonts\simhei.ttf"))

    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "Title", parent=styles["Title"], fontName="ArialBold", fontSize=21,
        leading=27, alignment=TA_CENTER, textColor=colors.HexColor("#1F4D78"), spaceAfter=12 * mm,
    )
    subtitle = ParagraphStyle(
        "Subtitle", parent=styles["Normal"], fontName="SimHei", fontSize=13,
        leading=21, alignment=TA_CENTER, textColor=colors.HexColor("#45606F"), spaceAfter=5 * mm,
    )
    heading = ParagraphStyle(
        "Heading", parent=styles["Heading2"], fontName="ArialBold", fontSize=14,
        leading=18, textColor=colors.HexColor("#2E74B5"), spaceBefore=7 * mm, spaceAfter=3 * mm,
    )
    heading_zh = ParagraphStyle(
        "HeadingZH", parent=heading, fontName="SimHei", fontSize=13,
        textColor=colors.HexColor("#375E55"), spaceBefore=0, spaceAfter=4 * mm,
    )
    english_style = ParagraphStyle(
        "English", parent=styles["BodyText"], fontName="Arial", fontSize=9.4,
        leading=14, alignment=TA_LEFT, textColor=colors.HexColor("#1D2730"),
        borderColor=colors.HexColor("#D9E4EA"), borderWidth=0.5, borderPadding=7,
        backColor=colors.HexColor("#F8FBFC"), spaceAfter=2.5 * mm,
    )
    chinese_style = ParagraphStyle(
        "Chinese", parent=styles["BodyText"], fontName="SimHei", fontSize=9.6,
        leading=16, alignment=TA_LEFT, textColor=colors.HexColor("#263C35"),
        borderColor=colors.HexColor("#E5DCCB"), borderWidth=0.5, borderPadding=7,
        backColor=colors.HexColor("#FBF7EF"), spaceAfter=5 * mm,
        wordWrap="CJK",
    )
    note_style = ParagraphStyle(
        "Note", parent=chinese_style, fontSize=10.5, leading=18,
        backColor=colors.HexColor("#EEF4F1"), borderColor=colors.HexColor("#B9CCC4"),
    )
    reference_style = ParagraphStyle(
        "Reference", parent=english_style, fontSize=8.3, leading=12,
        backColor=colors.white, borderWidth=0, borderPadding=0, spaceAfter=2 * mm,
    )

    story = [Spacer(1, 26 * mm)]
    story.append(Paragraph("Cove Full Dissertation Bilingual Review", title))
    story.append(Paragraph("Cove 论文全文中英逐段通读版", subtitle))
    story.append(Spacer(1, 8 * mm))
    story.append(Paragraph(
        "This is a review copy. The English submission PDF remains authoritative. "
        "Chinese paragraphs are local machine-assisted translations prepared for comprehension and checking; "
        "they are not part of the submitted dissertation.", english_style))
    story.append(Paragraph(
        "这是用于通读和核对的审阅版本。正式英文提交 PDF 仍是唯一权威文本。中文段落由本地机器翻译辅助生成，"
        "用于理解和检查，不属于最终提交论文。引用、数字和专业结论应以相邻英文原文为准。", note_style))
    story.append(PageBreak())

    for block in blocks:
        if block.kind == "heading":
            if block.english.lower().startswith("chapter ") and story:
                story.append(PageBreak())
            story.append(Paragraph(safe_markup(block.english), heading))
            story.append(Paragraph(safe_markup(block.chinese), heading_zh))
        else:
            pair = [
                Paragraph("<b>English</b><br/>" + safe_markup(block.english), english_style),
                Paragraph("<b>中文</b><br/>" + safe_markup(block.chinese), chinese_style),
            ]
            story.append(KeepTogether(pair))

    story.append(PageBreak())
    story.append(Paragraph("References", heading))
    story.append(Paragraph("参考文献（书目信息保留原文，不作翻译）", heading_zh))
    story.append(Paragraph(
        "Bibliographic titles, journal names, DOIs and URLs are preserved in their published form to avoid altering source information.",
        english_style,
    ))
    story.append(Paragraph(
        "为避免改变文献来源信息，题名、期刊名、DOI 与链接均保留正式英文形式。", chinese_style))
    for block in references:
        if block.english.lower() == "references":
            continue
        story.append(Paragraph(safe_markup(block.english), reference_style))

    doc = NumberedDocTemplate(
        str(OUTPUT_PDF), pagesize=A4, leftMargin=17 * mm, rightMargin=17 * mm,
        topMargin=16 * mm, bottomMargin=17 * mm,
        title="Cove Full Dissertation Bilingual Review",
        author="Cove dissertation review artefact",
    )
    doc.build(story)


def number_tokens(text: str) -> set[str]:
    return set(re.findall(r"(?<![A-Za-z])\d+(?:[.,]\d+)*%?", text))


def main() -> None:
    if not SOURCE_PDF.exists():
        raise FileNotFoundError(f"Build the dissertation first: {SOURCE_PDF}")
    blocks, references = extract_blocks()
    translate_blocks(blocks)
    blocks.extend(diagram_guide_blocks())
    build_pdf(blocks, references)

    missing_numbers = []
    for index, block in enumerate(blocks, start=1):
        if block.kind != "paragraph":
            continue
        missing = sorted(number_tokens(block.english) - number_tokens(block.chinese))
        if missing:
            missing_numbers.append({"block": index, "missing_in_chinese": missing, "english": block.english[:180]})
    audit = {
        "source_pdf": str(SOURCE_PDF),
        "output_pdf": str(OUTPUT_PDF),
        "bilingual_blocks": sum(1 for block in blocks if block.chinese),
        "paragraph_blocks": sum(1 for block in blocks if block.kind == "paragraph"),
        "heading_blocks": sum(1 for block in blocks if block.kind == "heading"),
        "reference_blocks_english_only": len(references),
        "numeric_review_flags": missing_numbers,
    }
    AUDIT_PATH.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in audit.items() if k != "numeric_review_flags"}, ensure_ascii=False, indent=2))
    print(f"numeric_review_flags={len(missing_numbers)}")


if __name__ == "__main__":
    main()
