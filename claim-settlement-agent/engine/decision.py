from typing import List
from data.schema import ClaimItem, PolicyClause, DecisionResult


class DecisionMaker:
    def __init__(self, clauses: List[PolicyClause]):
        self.clauses_dict = {c.clause_id: c for c in clauses}

    def generate_result(
        self,
        claim_id: str,
        patient_id: str,
        admission_date: str,
        items: List[ClaimItem],
        pipeline_metadata: dict = None
    ) -> DecisionResult:
        total_claimed = sum(i.total for i in items)
        total_approved = sum(i.approved_amount for i in items)

        citations = []

        # Determine overall status
        if total_approved <= 0 and total_claimed > 0:
            overall_status = "REJECTED"
        elif total_approved < total_claimed:
            overall_status = "PARTIAL"
        else:
            overall_status = "APPROVED"

        for item in items:
            if item.decision in ["REJECT", "PARTIAL"]:
                source_tag = ""
                if item.decision_source == "DETERMINISTIC":
                    source_tag = " [Deterministic Match]"
                elif item.decision_source == "LLM":
                    source_tag = " [LLM Adjudicated]"

                citation = (
                    f"Item '{item.description}' ({item.decision}). "
                    f"Reason: {item.reason}{source_tag}"
                )
                citations.append(citation)

        return DecisionResult(
            claim_id=claim_id,
            patient_id=patient_id,
            admission_date=admission_date,
            total_claimed=round(total_claimed, 2),
            total_approved=round(total_approved, 2),
            overall_status=overall_status,
            items=items,
            citations=citations,
            pipeline_metadata=pipeline_metadata
        )
