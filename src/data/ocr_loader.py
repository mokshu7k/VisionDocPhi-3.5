"""Parse Azure Computer Vision OCR JSON for SP-DocVQA."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from config.settings import OCR_DIR


@dataclass
class OcrLine:
    line_id: int
    text: str
    bbox: List[int]  # [x1, y1, x2, y2]


def bounding_box_to_xyxy(bounding_box: List[int]) -> List[int]:
    """Convert 8-point Azure boundingBox to [x1, y1, x2, y2]."""
    if len(bounding_box) < 8:
        return [0, 0, 0, 0]
    xs = [bounding_box[i] for i in range(0, 8, 2)]
    ys = [bounding_box[i] for i in range(1, 8, 2)]
    return [min(xs), min(ys), max(xs), max(ys)]


def get_ocr_path(ucsf_id: str, page_no: str) -> Path:
    return OCR_DIR / f"{ucsf_id}_{page_no}.json"


def load_ocr_lines(ucsf_id: str, page_no: str) -> List[OcrLine]:
    """Load OCR lines from Azure JSON file."""
    path = get_ocr_path(ucsf_id, page_no)
    if not path.exists():
        return []

    with open(path, "r") as f:
        data = json.load(f)

    lines: List[OcrLine] = []
    line_id = 0
    for result in data.get("recognitionResults", []):
        for line in result.get("lines", []):
            text = line.get("text", "").strip()
            if not text:
                continue
            bbox = bounding_box_to_xyxy(line.get("boundingBox", []))
            lines.append(OcrLine(line_id=line_id, text=text, bbox=bbox))
            line_id += 1

    return lines


def ocr_available(ucsf_id: str, page_no: str) -> bool:
    path = get_ocr_path(ucsf_id, page_no)
    return path.exists() and path.stat().st_size > 0
