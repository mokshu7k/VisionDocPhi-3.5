"""Shared types for the agentic DocVQA pipeline."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RoutingDecision:
    route: str  # "vision_only" | "ocr_infused"
    reason: str
    text_rule: str  # "vision_only" | "ocr_infused"
    density_override: bool = False
    edge_density: float = 0.0
    resolution_flag: bool = False
    low_contrast_flag: bool = False
    ui_tag: str = ""
    pre_gate_route: str = ""

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "route": self.route,
            "reason": self.reason,
            "text_rule": self.text_rule,
            "density_override": self.density_override,
            "edge_density": self.edge_density,
            "resolution_flag": self.resolution_flag,
            "low_contrast_flag": self.low_contrast_flag,
            "ui_tag": self.ui_tag,
        }
        if self.pre_gate_route:
            out["pre_gate_route"] = self.pre_gate_route
        return out


@dataclass
class ScoredLine:
    line_id: int
    text: str
    bbox: List[int]
    dense_score: float
    sparse_score: float
    final_score: float


@dataclass
class FormattedSnippet:
    text_block: str
    line_ids: List[int]
    bboxes: List[List[int]]
    char_count: int
    presentation_order: List[int] = field(default_factory=list)
    sanitized: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "char_count": self.char_count,
            "presentation_order": self.presentation_order,
            "bboxes": self.bboxes,
            "line_ids": self.line_ids,
            "sanitized": self.sanitized,
        }


@dataclass
class PipelineResult:
    answer: str
    routing: Optional[RoutingDecision] = None
    retrieval: Optional[Dict[str, Any]] = None
    formatting: Optional[Dict[str, Any]] = None
    used_ocr: bool = False
    ocr_snippet: Optional[str] = None
    prompt_meta: Optional[Dict[str, Any]] = None

    def to_audit_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "used_ocr": self.used_ocr,
            "prediction": self.answer,
        }
        if self.routing:
            out["routing"] = self.routing.to_dict()
        if self.retrieval:
            out["retrieval"] = self.retrieval
        if self.formatting:
            out["formatting"] = self.formatting
        if self.prompt_meta:
            out["prompt_meta"] = self.prompt_meta
        return out
