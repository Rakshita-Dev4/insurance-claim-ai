import logging
from typing import List, Dict
from data.schema import ClaimItem, PolicyRuleDSL

logger = logging.getLogger(__name__)

# Minimum LLM confidence to apply a rule.
# Deterministic rules always set confidence=1.0 and are never skipped.
CONFIDENCE_THRESHOLD = 0.65


class RuleEngine:
    """
    Pure mathematical evaluator. Zero LLM calls.
    
    Applies DSL rules to bill items in strict precedence order:
    Pass 1: Detect Room Rent limits and calculate proportional ratios
    Pass 2: Apply rules per item (EXCLUSION → LIMIT_CAP → PROPORTIONAL → COPAY)
    Pass 3: Apply Global Cap across all items
    
    Special handling:
    - Negative totals (discounts) are always passed through, never excluded
    - Room rent detection uses exact semantic_target binding, not substring matching
    """

    def __init__(self):
        pass

    def evaluate(
        self,
        items: List[ClaimItem],
        dsl_rules: List[Dict],
        mapping: Dict[str, List[str]],
        claim_metadata: Dict
    ) -> List[ClaimItem]:
        
        ytd_approved = claim_metadata.get("ytd_approved", 0.0)

        # Convert raw dicts to PolicyRuleDSL dataclass objects
        rules_list = []
        for r in dsl_rules:
            try:
                if isinstance(r, dict):
                    rules_list.append(PolicyRuleDSL(**r))
                else:
                    rules_list.append(r)
            except Exception as e:
                logger.warning(f"Skipping malformed rule: {r} — {e}")

        rules_dict = {r.rule_id: r for r in rules_list if hasattr(r, 'rule_id')}

        # ── Pass 1: Room Rent Proportional Ratio Detection ──
        proportion_ratio = 1.0
        room_rent_keywords = {"room rent", "room charges", "ward charges", "bed charges"}

        for item in items:
            matched_rule_ids = mapping.get(item.item_id, [])
            for rid in matched_rule_ids:
                if rid not in rules_dict:
                    continue
                rule = rules_dict[rid]

                if rule.operation == "LIMIT_CAP":
                    # Check if this rule's semantic targets are specifically room-rent related
                    targets_lower = {t.lower() for t in (rule.semantic_targets or [])}
                    is_room_rent = bool(targets_lower & room_rent_keywords)

                    if is_room_rent and rule.limit_value and rule.limit_value > 0:
                        daily_cost = item.unit_cost if item.unit_cost > 0 else (item.total / max(1, item.qty))
                        if daily_cost > rule.limit_value:
                            proportion_ratio = rule.limit_value / daily_cost
                            logger.info(f"Room Rent ratio: {proportion_ratio:.2f} (limit={rule.limit_value}, actual={daily_cost})")

        # ── Pass 2: Per-Item Evaluation ──
        for item in items:
            # Discounts (negative totals) are always passed through as-is
            if item.total < 0:
                item.approved_amount = item.total
                item.decision = "APPROVE"
                item.reason = "Discount/deduction passed through."
                continue

            item.approved_amount = item.total
            item.decision = "APPROVE"

            matched_rule_ids = mapping.get(item.item_id, [])
            applied_rules = []

            # Resolve rules and sort by execution precedence
            precedence = {
                "EXCLUSION": 0,
                "LIMIT_CAP": 1,
                "PROPORTIONAL_DEDUCTION": 2,
                "COPAY": 3,
                "GLOBAL_CAP": 4
            }
            matched_rules = []
            for rid in matched_rule_ids:
                if rid in rules_dict:
                    matched_rules.append(rules_dict[rid])
            matched_rules.sort(key=lambda x: precedence.get(x.operation, 9))

            for rule in matched_rules:

                # Skip low-confidence LLM rules (deterministic rules set confidence=1.0)
                rule_confidence = rule.confidence if rule.confidence is not None else 1.0
                if rule_confidence < CONFIDENCE_THRESHOLD:
                    logger.info(
                        f"Skipping low-confidence rule '{rule.rule_id}' "
                        f"(confidence={rule_confidence:.2f} < {CONFIDENCE_THRESHOLD}) for '{item.description}'"
                    )
                    continue

                if rule.operation == "EXCLUSION":
                    item.approved_amount = 0.0
                    item.decision = "REJECT"
                    item.reason = (
                        f"Excluded under {rule.rule_id}."
                    )
                    applied_rules.append("EXCLUSION")
                    break  # No further rules needed after exclusion

                elif rule.operation == "LIMIT_CAP" and "EXCLUSION" not in applied_rules:
                    limit = rule.limit_value
                    if limit and limit > 0:
                        daily_cost = item.unit_cost if item.unit_cost > 0 else (item.total / max(1, item.qty))
                        if daily_cost > limit:
                            item.approved_amount = limit * item.qty
                            item.decision = "PARTIAL"
                            item.reason = f"Capped at {limit}/unit (limit from {rule.rule_id})."
                            applied_rules.append("CAP")

                elif rule.operation == "PROPORTIONAL_DEDUCTION" and proportion_ratio < 1.0 and "EXCLUSION" not in applied_rules:
                    item.approved_amount = round(item.approved_amount * proportion_ratio, 2)
                    item.decision = "PARTIAL"
                    item.reason = f"Proportionally reduced (ratio: {proportion_ratio:.2f}) due to Room Rent limit breach."
                    applied_rules.append("PROPORTIONAL")

                elif rule.operation == "COPAY" and "EXCLUSION" not in applied_rules:
                    pct = rule.percentage
                    if pct and pct > 0:
                        item.approved_amount = round(item.approved_amount * (1 - pct), 2)
                        item.decision = "PARTIAL"
                        item.reason = f"Co-payment of {pct * 100:.0f}% applied ({rule.rule_id})."

        # ── Pass 3: Global Cap ──
        global_cap = None
        for rule in rules_list:
            if rule.operation == "GLOBAL_CAP" and rule.limit_value and rule.limit_value > 0:
                global_cap = rule.limit_value
                break

        if global_cap:
            running_total = ytd_approved
            for item in items:
                if running_total + item.approved_amount > global_cap:
                    remaining = max(0, global_cap - running_total)
                    item.approved_amount = round(remaining, 2)
                    item.decision = "PARTIAL" if remaining > 0 else "REJECT"
                    if remaining == 0:
                        item.reason = f"Annual limit of {global_cap} fully exhausted."
                running_total += item.approved_amount

        return items
