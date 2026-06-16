"""Layout-preserving snippet formatter."""

from typing import List

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
            )

        # Re-sort selected lines by reading order: y1 then x1
        indexed = list(enumerate(scored_lines))
        indexed.sort(key=lambda x: (x[1].bbox[1], x[1].bbox[0]))

        presentation_order = [idx for idx, _ in indexed]
        sorted_lines = [line for _, line in indexed]

        lines_text = []
        line_ids = []
        bboxes = []
        for line in sorted_lines:
            y1, x1, y2, x2 = line.bbox
            lines_text.append(f"[{y1},{x1},{y2},{x2}] {line.text}")
            line_ids.append(line.line_id)
            bboxes.append(line.bbox)

        header = "Relevant OCR excerpts (page coordinates, top-to-bottom):\n"
        body = "\n".join(lines_text)
        text_block = header + body

        return FormattedSnippet(
            text_block=text_block,
            line_ids=line_ids,
            bboxes=bboxes,
            char_count=len(text_block),
            presentation_order=presentation_order,
        )
