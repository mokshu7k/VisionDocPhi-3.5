"""Router agent: text rules + image density fallback."""

from typing import List

from PIL import Image

from config.settings import LAYOUT_HEAVY_TYPES, LAYOUT_KEYWORDS
from src.agents.image_density import analyze_image_density
from src.agents.types import RoutingDecision
from src.data.ocr_loader import ocr_available


def _text_route(question: str, question_types: List[str]) -> str:
    q_lower = question.lower()
    if any(t in LAYOUT_HEAVY_TYPES for t in question_types):
        return "ocr_infused"
    if any(kw in q_lower for kw in LAYOUT_KEYWORDS):
        return "ocr_infused"
    return "vision_only"


class RouterAgent:
    def decide(
        self,
        question: str,
        question_types: List[str],
        image: Image.Image,
        ucsf_id: str = "",
        page_no: str = "",
    ) -> RoutingDecision:
        text_rule = _text_route(question, question_types)
        density = analyze_image_density(image)

        if ucsf_id and page_no and not ocr_available(ucsf_id, page_no):
            return RoutingDecision(
                route="vision_only",
                reason="ocr_unavailable",
                text_rule=text_rule,
                density_override=False,
                edge_density=density.edge_density,
                resolution_flag=density.resolution_flag,
                low_contrast_flag=density.low_contrast_flag,
                ui_tag="Native Vision",
            )

        if density.density_override:
            route = "ocr_infused"
            reason = "image_density_fallback"
        elif text_rule == "ocr_infused":
            route = "ocr_infused"
            reason = "question_type_or_keyword"
        else:
            route = "vision_only"
            reason = "visual_question_low_density"

        ui_tag = "OCR Enhanced" if route == "ocr_infused" else "Native Vision"

        return RoutingDecision(
            route=route,
            reason=reason,
            text_rule=text_rule,
            density_override=density.density_override,
            edge_density=density.edge_density,
            resolution_flag=density.resolution_flag,
            low_contrast_flag=density.low_contrast_flag,
            ui_tag=ui_tag,
        )
