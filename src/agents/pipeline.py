"""Orchestrates Router → Retriever → Formatter for ocr_adaptive mode."""

from typing import Any, Dict, List

from PIL import Image

from src.agents.formatter_agent import LayoutFormatterAgent
from src.agents.retriever_agent import HybridRetrieverAgent
from src.agents.router_agent import RouterAgent
from src.agents.types import PipelineResult
from src.data.ocr_loader import load_ocr_lines


class AgenticDocVQAPipeline:
    def __init__(self):
        self.router = RouterAgent()
        self.retriever = HybridRetrieverAgent()
        self.formatter = LayoutFormatterAgent()

    def prepare(
        self,
        image: Image.Image,
        question: str,
        question_types: List[str],
        ucsf_id: str,
        page_no: str,
    ) -> PipelineResult:
        routing = self.router.decide(
            question, question_types, image, ucsf_id, page_no
        )

        if routing.route != "ocr_infused":
            return PipelineResult(
                answer="",
                routing=routing,
                used_ocr=False,
                ocr_snippet=None,
            )

        ocr_lines = load_ocr_lines(ucsf_id, page_no)
        scored = self.retriever.retrieve(
            question, ocr_lines, ucsf_id=ucsf_id, page_no=page_no
        )

        if not scored:
            routing.route = "vision_only"
            routing.reason = "ocr_empty_after_retrieval"
            routing.ui_tag = "Native Vision"
            return PipelineResult(
                answer="",
                routing=routing,
                used_ocr=False,
                ocr_snippet=None,
            )

        formatted = self.formatter.format(scored)

        retrieval = {
            "line_ids": [s.line_id for s in scored],
            "dense_scores": [s.dense_score for s in scored],
            "sparse_scores": [s.sparse_score for s in scored],
            "final_scores": [s.final_score for s in scored],
        }

        return PipelineResult(
            answer="",
            routing=routing,
            retrieval=retrieval,
            formatting=formatted.to_dict(),
            used_ocr=True,
            ocr_snippet=formatted.text_block,
        )

    def build_audit(self, result: PipelineResult, sample: Dict[str, Any]) -> Dict[str, Any]:
        audit = result.to_audit_dict()
        audit.update({
            "question_id": sample.get("question_id"),
            "doc_id": sample.get("doc_id"),
            "cohort": sample.get("cohort", ""),
            "question_types": sample.get("question_types", []),
            "question": sample.get("question"),
        })
        return audit
