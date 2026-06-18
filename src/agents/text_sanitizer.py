"""Sanitize OCR line text before prompt assembly."""

import re
from typing import Optional

from config.settings import OCR_BOILERPLATE_DENYLIST

BOILERPLATE_START_RE = re.compile(
    r"^(" + "|".join(re.escape(w) for w in OCR_BOILERPLATE_DENYLIST) + r")\b",
    re.IGNORECASE,
)
SAMPLE_NUM_RE = re.compile(r"Sample\s*#", re.IGNORECASE)


def sanitize_line_text(text: str) -> Optional[str]:
    """Clean a single OCR line; return None to drop the line."""
    t = text.strip()
    if not t:
        return None

    t = re.sub(r"::+", ":", t)
    t = SAMPLE_NUM_RE.sub("Sample Num", t)

    if t.startswith("#"):
        t = t.lstrip("#").strip()
        if not t:
            return None

    if BOILERPLATE_START_RE.match(t):
        return None

    return t
