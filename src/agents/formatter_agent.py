"""Layout-preserving snippet formatter."""

from typing import List

from src.agents.text_sanitizer import sanitize_line_text
from src.agents.types import FormattedSnippet, ScoredLine


class LayoutFormatterAgent:
    def format(self, scored_lines: List[ScoredLine]) -> FormattedSnippet:
        if not scored_lines:
            return FormattedSnippet(
                text_block="",
                line_ids=[],
                bboxes=[],
                char_count=0,
                presentation_order=[],
                sanitized=True,
            )

        # Sanitize then re-sort by reading order: y1 then x1
        sanitized_lines: List[ScoredLine] = []
        for line in scored_lines:
            clean = sanitize_line_text(line.text)
            if clean is None:
                continue
            sanitized_lines.append(
                ScoredLine(
                    line_id=line.line_id,
                    text=clean,
                    bbox=line.bbox,
                    dense_score=line.dense_score,
                    sparse_score=line.sparse_score,
                    final_score=line.final_score,
                )
            )

        if not sanitized_lines:
            return FormattedSnippet(
                text_block="",
                line_ids=[],
                bboxes=[],
                char_count=0,
                presentation_order=[],
                sanitized=True,
            )

        indexed = list(enumerate(sanitized_lines))
        indexed.sort(key=lambda x: (x[1].bbox[1], x[1].bbox[0]))

        presentation_order = [idx for idx, _ in indexed]
        sorted_lines = [line for _, line in indexed]

        lines_text = []
        line_ids = []
        bboxes = []
        for line in sorted_lines:
            x1, y1, x2, y2 = line.bbox
            lines_text.append(f"[{x1},{y1},{x2},{y2}] {line.text}")
            line_ids.append(line.line_id)
            bboxes.append(line.bbox)

        body = "\n".join(lines_text)

        return FormattedSnippet(
            text_block=body,
            line_ids=line_ids,
            bboxes=bboxes,
            char_count=len(body),
            presentation_order=presentation_order,
            sanitized=True,
        )
