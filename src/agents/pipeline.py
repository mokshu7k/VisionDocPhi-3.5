"""Orchestrates Router → Retriever → Expander → Formatter for ocr_adaptive mode."""

from typing import Any, Dict, List

from PIL import Image

from config.settings import ENABLE_CONTEXT_EXPANSION
from src.agents.context_expander import expand_context
from src.agents.formatter_agent import LayoutFormatterAgent
from src.agents.quality_gate import passes_quality_gate
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
                prompt_meta={"used_xml_wrapper": False, "ocr_in_prompt": False},
            )

        routing.pre_gate_route = routing.route

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
                prompt_meta={"used_xml_wrapper": False, "ocr_in_prompt": False},
            )

        retrieved_ids = [s.line_id for s in scored]
        expanded = scored
        if ENABLE_CONTEXT_EXPANSION:
            expanded = expand_context(scored, ocr_lines)

        formatted = self.formatter.format(expanded)

        if not formatted.text_block.strip():
            routing.route = "vision_only"
            routing.reason = "sanitizer_dropped_all_lines"
            routing.ui_tag = "Native Vision"
            return PipelineResult(
                answer="",
                routing=routing,
                retrieval={
                    "line_ids": retrieved_ids,
                    "dense_scores": [s.dense_score for s in scored],
                    "sparse_scores": [s.sparse_score for s in scored],
                    "final_scores": [s.final_score for s in scored],
                    "expanded_line_ids": [s.line_id for s in expanded],
                },
                formatting=formatted.to_dict(),
                used_ocr=False,
                ocr_snippet=None,
                prompt_meta={"used_xml_wrapper": False, "ocr_in_prompt": False},
            )

        if not passes_quality_gate(scored, formatted, question):
            routing.route = "vision_only"
            routing.reason = "low_retrieval_confidence"
            routing.ui_tag = "Native Vision"
            return PipelineResult(
                answer="",
                routing=routing,
                retrieval={
                    "line_ids": retrieved_ids,
                    "dense_scores": [s.dense_score for s in scored],
                    "sparse_scores": [s.sparse_score for s in scored],
                    "final_scores": [s.final_score for s in scored],
                    "expanded_line_ids": [s.line_id for s in expanded],
                },
                formatting=formatted.to_dict(),
                used_ocr=False,
                ocr_snippet=None,
                prompt_meta={"used_xml_wrapper": False, "ocr_in_prompt": False},
            )

        retrieval = {
            "line_ids": retrieved_ids,
            "dense_scores": [s.dense_score for s in scored],
            "sparse_scores": [s.sparse_score for s in scored],
            "final_scores": [s.final_score for s in scored],
            "expanded_line_ids": [s.line_id for s in expanded],
        }

        prompt_meta = {
            "used_xml_wrapper": True,
            "ocr_in_prompt": True,
        }

        return PipelineResult(
            answer="",
            routing=routing,
            retrieval=retrieval,
            formatting=formatted.to_dict(),
            used_ocr=True,
            ocr_snippet=formatted.text_block,
            prompt_meta=prompt_meta,
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
