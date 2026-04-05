from dataclasses import dataclass, field
from typing import List, Optional, Literal
from pydantic import BaseModel, field_validator


# ─── Core Data Classes ───────────────────────────────────────────────

@dataclass
class ClaimItem:
    item_id: str
    code_type: str  # e.g., "ICD-10", "CPT", "NONE"
    code: str
    description: str
    qty: int
    unit_cost: float
    total: float
    # Enriched by the pipeline
    matched_clause_id: Optional[str] = None
    decision: Optional[str] = None  # "APPROVE", "PARTIAL", "REJECT"
    approved_amount: float = 0.0
    reason: Optional[str] = None
    decision_source: Optional[str] = None  # "DETERMINISTIC" or "LLM"


@dataclass
class PolicyClause:
    clause_id: str
    page_num: int
    para_num: int
    text: str
    coverage_type: str  # e.g., "general", "exclusion", "limit"
    limit_amount: Optional[float] = None
    is_exclusion: bool = False


@dataclass
class PolicyRuleDSL:
    rule_id: str
    operation: str  # "EXCLUSION", "LIMIT_CAP", "PROPORTIONAL_DEDUCTION", "COPAY", "GLOBAL_CAP"
    semantic_targets: List[str] = field(default_factory=list)
    limit_value: Optional[float] = None
    percentage: Optional[float] = None
    trigger_condition: Optional[str] = None
    confidence: Optional[float] = None


@dataclass
class DecisionResult:
    claim_id: str
    patient_id: str
    admission_date: str
    total_claimed: float
    total_approved: float
    overall_status: str  # "APPROVED", "PARTIAL", "REJECTED"
    items: List[ClaimItem]
    citations: List[str]
    pipeline_metadata: Optional[dict] = None  # For evaluation metrics


# ─── Pydantic Validation Models (LLM Output Guardrails) ─────────────

class LLMRuleOutput(BaseModel):
    """Validates a single rule extracted by the MicroCompiler LLM."""
    rule_id: str = "UNKNOWN"
    operation: str
    limit_value: Optional[float] = None
    percentage: Optional[float] = None
    confidence: Optional[float] = None

    @field_validator("operation")
    @classmethod
    def validate_operation(cls, v: str) -> str:
        valid_ops = {"EXCLUSION", "LIMIT_CAP", "PROPORTIONAL_DEDUCTION", "COPAY", "GLOBAL_CAP"}
        normalized = v.upper().strip().replace(" ", "_")
        # Common LLM typos
        typo_map = {
            "EXCLUDE": "EXCLUSION",
            "EXCLUDED": "EXCLUSION",
            "EXCLUSIONS": "EXCLUSION",
            "CAP": "LIMIT_CAP",
            "LIMIT": "LIMIT_CAP",
            "CO_PAY": "COPAY",
            "CO-PAY": "COPAY",
            "DEDUCTION": "PROPORTIONAL_DEDUCTION",
            "PROPORTIONAL": "PROPORTIONAL_DEDUCTION",
        }
        if normalized in valid_ops:
            return normalized
        if normalized in typo_map:
            return typo_map[normalized]
        return "EXCLUSION"  # Safe fallback

    @field_validator("percentage")
    @classmethod
    def normalize_percentage(cls, v):
        if v is not None and v > 1.0:
            return v / 100.0  # LLM returned 10 instead of 0.10
        return v


class LLMRulesResponse(BaseModel):
    """Validates the full MicroCompiler JSON response."""
    rules: List[LLMRuleOutput] = []


class LLMBillItem(BaseModel):
    """Validates a single bill item extracted by the BillParser LLM."""
    item_id: Optional[str] = None
    code_type: Optional[str] = "NONE"
    code: Optional[str] = "N/A"
    description: str = "Unknown"
    qty: Optional[int] = 1
    unit_cost: Optional[float] = 0.0
    total: Optional[float] = 0.0

    @field_validator("qty", mode="before")
    @classmethod
    def coerce_qty(cls, v):
        if v is None:
            return 1
        try:
            return int(v)
        except (ValueError, TypeError):
            return 1

    @field_validator("unit_cost", "total", mode="before")
    @classmethod
    def coerce_float(cls, v):
        if v is None:
            return 0.0
        try:
            return float(v)
        except (ValueError, TypeError):
            return 0.0


class LLMBillResponse(BaseModel):
    """Validates the full BillParser JSON response."""
    patient_id: Optional[str] = "UNKNOWN"
    admission_date: Optional[str] = "UNKNOWN"
    items: List[LLMBillItem] = []
