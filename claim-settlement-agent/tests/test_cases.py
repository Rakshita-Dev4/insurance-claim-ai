"""
Real pytest test suite for the Insurance Claim Settlement Agent.
Tests each component in isolation with deterministic inputs.
"""
import pytest
import json
from data.schema import (
    ClaimItem, PolicyClause, PolicyRuleDSL,
    LLMRuleOutput, LLMRulesResponse, LLMBillItem, LLMBillResponse
)
from engine.rule_engine import RuleEngine
from engine.decision import DecisionMaker
from engine.exclusion_matcher import ExclusionMatcher


# ─── Schema Validation Tests ──────────────────────────────────────────

class TestPydanticValidation:
    """Tests the Pydantic guardrails that validate LLM JSON output."""

    def test_valid_rule_output(self):
        rule = LLMRuleOutput(rule_id="Code-Excl01", operation="EXCLUSION")
        assert rule.operation == "EXCLUSION"

    def test_operation_typo_correction(self):
        """LLM might return 'EXCLUDE' instead of 'EXCLUSION'."""
        rule = LLMRuleOutput(rule_id="test", operation="EXCLUDE")
        assert rule.operation == "EXCLUSION"

    def test_operation_typo_copay(self):
        rule = LLMRuleOutput(rule_id="test", operation="CO-PAY")
        assert rule.operation == "COPAY"

    def test_percentage_normalization(self):
        """LLM might return 10 instead of 0.10."""
        rule = LLMRuleOutput(rule_id="test", operation="COPAY", percentage=10.0)
        assert rule.percentage == 0.10

    def test_percentage_already_decimal(self):
        rule = LLMRuleOutput(rule_id="test", operation="COPAY", percentage=0.15)
        assert rule.percentage == 0.15

    def test_bill_item_null_coercion(self):
        """LLM might return null for qty."""
        item = LLMBillItem(description="Test", qty=None, unit_cost=None, total=None)
        assert item.qty == 1
        assert item.unit_cost == 0.0
        assert item.total == 0.0

    def test_bill_response_validation(self):
        data = {
            "patient_id": "John Doe",
            "admission_date": "2024-01-15",
            "items": [
                {"description": "Consultation", "qty": 1, "unit_cost": 500, "total": 500},
                {"description": "Discount", "total": -200}
            ]
        }
        validated = LLMBillResponse(**data)
        assert len(validated.items) == 2
        assert validated.items[1].total == -200.0


# ─── ExclusionMatcher Tests ──────────────────────────────────────────

class TestExclusionMatcher:
    """Tests the deterministic fuzzy matching engine."""

    @pytest.fixture
    def matcher_with_data(self):
        matcher = ExclusionMatcher()
        clauses = [
            PolicyClause(
                clause_id="C1", page_num=29, para_num=1,
                text="List I - Items for which coverage is not available in the Policy:\n1. File Charges\n2. Registration Charges\n3. Admission Kit\n4. Service Charges\n5. Nebulizer Kit\n6. Gloves\n7. Gown\n8. Cap",
                coverage_type="exclusion"
            ),
            PolicyClause(
                clause_id="C2", page_num=31, para_num=1,
                text="List II - Items that are to be subsumed into Room Charges:\n1. Bed pan\n2. Blanket\n3. Pillow cover\n4. Linen charges",
                coverage_type="exclusion"
            ),
            PolicyClause(
                clause_id="C3", page_num=10, para_num=1,
                text="General coverage for hospitalization including consultation fees and emergency treatment.",
                coverage_type="general"
            ),

        ]
        matcher.extract_exclusions_from_clauses(clauses)
        return matcher

    def test_exact_match(self, matcher_with_data):
        result = matcher_with_data.match("File Charges")
        assert result is not None
        assert result["score"] >= 80

    def test_compound_name_no_false_match(self, matcher_with_data):
        """Compound names like 'FILE CHARGE & REGISTRATION CHARGES' should
        NOT match unrelated items just because they share the word 'charges'."""
        result = matcher_with_data.match("Room & Nursing Charges")
        assert result is None  # Must NOT match any exclusion

    def test_no_match_for_legitimate_item(self, matcher_with_data):
        """Consultation Fee should NOT match any exclusion."""
        result = matcher_with_data.match("Consultation Fee")
        assert result is None

    def test_no_match_for_emergency(self, matcher_with_data):
        """Emergency Charge should NOT match any exclusion."""
        result = matcher_with_data.match("Emergency Charge")
        assert result is None

    def test_no_match_for_lab(self, matcher_with_data):
        """Laboratory Charges should NOT match any exclusion."""
        result = matcher_with_data.match("Laboratory Charges")
        assert result is None

    def test_discount_skip(self, matcher_with_data):
        """Discounts should never be matched against exclusions."""
        result = matcher_with_data.match("Special Discount")
        assert result is None

    def test_nebulizer_exact(self, matcher_with_data):
        """Exact match for Nebulizer Kit (the equipment, not the service)."""
        result = matcher_with_data.match("Nebulizer Kit")
        assert result is not None


# ─── RuleEngine Tests ─────────────────────────────────────────────────

class TestRuleEngine:
    """Tests the pure mathematical evaluation layer."""

    @pytest.fixture
    def engine(self):
        return RuleEngine()

    def test_no_rules_full_approval(self, engine):
        items = [
            ClaimItem(item_id="B1", code_type="NONE", code="N/A",
                      description="Consultation", qty=1, unit_cost=500, total=500)
        ]
        result = engine.evaluate(items, [], {"B1": []}, {"ytd_approved": 0})
        assert result[0].decision == "APPROVE"
        assert result[0].approved_amount == 500

    def test_exclusion_zeroes_amount(self, engine):
        items = [
            ClaimItem(item_id="B1", code_type="NONE", code="N/A",
                      description="File Charges", qty=1, unit_cost=500, total=500)
        ]
        rules = [{"rule_id": "R1", "operation": "EXCLUSION", "semantic_targets": ["file charges"]}]
        mapping = {"B1": ["R1"]}
        result = engine.evaluate(items, rules, mapping, {"ytd_approved": 0})
        assert result[0].decision == "REJECT"
        assert result[0].approved_amount == 0.0

    def test_limit_cap(self, engine):
        items = [
            ClaimItem(item_id="B1", code_type="NONE", code="N/A",
                      description="ICU Room", qty=3, unit_cost=8000, total=24000)
        ]
        rules = [{"rule_id": "R1", "operation": "LIMIT_CAP",
                  "semantic_targets": ["icu room"], "limit_value": 5000}]
        mapping = {"B1": ["R1"]}
        result = engine.evaluate(items, rules, mapping, {"ytd_approved": 0})
        assert result[0].decision == "PARTIAL"
        assert result[0].approved_amount == 15000  # 5000 * 3

    def test_copay_reduces_amount(self, engine):
        items = [
            ClaimItem(item_id="B1", code_type="NONE", code="N/A",
                      description="Surgery", qty=1, unit_cost=10000, total=10000)
        ]
        rules = [{"rule_id": "R1", "operation": "COPAY",
                  "semantic_targets": ["surgery"], "percentage": 0.20}]
        mapping = {"B1": ["R1"]}
        result = engine.evaluate(items, rules, mapping, {"ytd_approved": 0})
        assert result[0].decision == "PARTIAL"
        assert result[0].approved_amount == 8000.0

    def test_discount_passthrough(self, engine):
        """Negative totals (discounts) should always pass through."""
        items = [
            ClaimItem(item_id="B1", code_type="NONE", code="N/A",
                      description="Special Discount", qty=1, unit_cost=-500, total=-500)
        ]
        rules = [{"rule_id": "R1", "operation": "EXCLUSION",
                  "semantic_targets": ["special discount"]}]
        mapping = {"B1": ["R1"]}
        result = engine.evaluate(items, rules, mapping, {"ytd_approved": 0})
        assert result[0].decision == "APPROVE"
        assert result[0].approved_amount == -500

    def test_global_cap(self, engine):
        items = [
            ClaimItem(item_id="B1", code_type="NONE", code="N/A",
                      description="Surgery A", qty=1, unit_cost=150000, total=150000),
            ClaimItem(item_id="B2", code_type="NONE", code="N/A",
                      description="Surgery B", qty=1, unit_cost=100000, total=100000),
        ]
        rules = [{"rule_id": "GCAP", "operation": "GLOBAL_CAP",
                  "semantic_targets": ["global"], "limit_value": 200000}]
        mapping = {"B1": [], "B2": []}
        result = engine.evaluate(items, rules, mapping, {"ytd_approved": 0})
        assert result[0].approved_amount == 150000
        assert result[1].approved_amount == 50000  # Capped

    def test_exclusion_precedence_over_cap(self, engine):
        """EXCLUSION should override any CAP rules."""
        items = [
            ClaimItem(item_id="B1", code_type="NONE", code="N/A",
                      description="File Charges", qty=1, unit_cost=500, total=500)
        ]
        rules = [
            {"rule_id": "R1", "operation": "EXCLUSION", "semantic_targets": ["file charges"]},
            {"rule_id": "R2", "operation": "LIMIT_CAP", "semantic_targets": ["file charges"], "limit_value": 300}
        ]
        mapping = {"B1": ["R1", "R2"]}
        result = engine.evaluate(items, rules, mapping, {"ytd_approved": 0})
        assert result[0].decision == "REJECT"
        assert result[0].approved_amount == 0.0


# ─── DecisionMaker Tests ─────────────────────────────────────────────

class TestDecisionMaker:
    """Tests the final decision aggregation."""

    @pytest.fixture
    def maker(self):
        return DecisionMaker([])

    def test_full_approval(self, maker):
        items = [
            ClaimItem(item_id="B1", code_type="NONE", code="N/A",
                      description="Consultation", qty=1, unit_cost=500, total=500,
                      decision="APPROVE", approved_amount=500)
        ]
        result = maker.generate_result("test-1", "patient-1", "2024-01-01", items)
        assert result.overall_status == "APPROVED"
        assert result.total_claimed == 500
        assert result.total_approved == 500
        assert len(result.citations) == 0

    def test_partial_status(self, maker):
        items = [
            ClaimItem(item_id="B1", code_type="NONE", code="N/A",
                      description="Consultation", qty=1, unit_cost=1000, total=1000,
                      decision="APPROVE", approved_amount=1000),
            ClaimItem(item_id="B2", code_type="NONE", code="N/A",
                      description="File Charges", qty=1, unit_cost=500, total=500,
                      decision="REJECT", approved_amount=0, reason="Excluded",
                      decision_source="DETERMINISTIC"),
        ]
        result = maker.generate_result("test-2", "patient-1", "2024-01-01", items)
        assert result.overall_status == "PARTIAL"
        assert result.total_approved == 1000
        assert len(result.citations) == 1
        assert "[Deterministic Match]" in result.citations[0]
