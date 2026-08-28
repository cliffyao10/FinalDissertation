"""Create temporary contact sheets from rendered dissertation pages."""

from __future__ import annotations

import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw


def main() -> None:
    source = Path(sys.argv[1])
    output = Path(sys.argv[2])
    output.mkdir(parents=True, exist_ok=True)
    pages = sorted(source.glob("page-*.png"), key=lambda path: int(path.stem.split("-")[-1]))
    per_sheet = 8
    thumb_width = 240
    margin = 18
    label_height = 24

    for sheet_index in range(math.ceil(len(pages) / per_sheet)):
        batch = pages[sheet_index * per_sheet : (sheet_index + 1) * per_sheet]
        thumbs: list[tuple[Image.Image, str]] = []
        for page in batch:
            image = Image.open(page).convert("RGB")
            height = round(image.height * thumb_width / image.width)
            image.thumbnail((thumb_width, height), Image.Resampling.LANCZOS)
            thumbs.append((image.copy(), page.stem))

        cell_height = max(image.height for image, _ in thumbs) + label_height
        sheet = Image.new(
            "RGB",
            (margin * 3 + thumb_width * 2, margin * 5 + cell_height * 4),
            "#d8d8d8",
        )
        draw = ImageDraw.Draw(sheet)
        for index, (image, label) in enumerate(thumbs):
            row, column = divmod(index, 2)
            x = margin + column * (thumb_width + margin)
            y = margin + row * (cell_height + margin)
            sheet.paste(image, (x, y + label_height))
            draw.text((x, y + 4), label, fill="black")
        sheet.save(output / f"sheet-{sheet_index + 1:02d}.png")


if __name__ == "__main__":
    main()
