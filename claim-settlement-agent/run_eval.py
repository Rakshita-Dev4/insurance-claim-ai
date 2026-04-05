"""
Live evaluation runner for Bill1 + policy1 using the real pipeline.
Calls the actual API endpoint and compares against ground truth.
"""
import sys, os, warnings, json, time
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
sys.path.insert(0, '.')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tests'))

from evaluate import (
    compute_item_level_metrics,
    compute_claim_level_metrics,
    compute_pipeline_distribution,
    compute_determinism_score,
    print_evaluation_report,
)
from ingestion.bill_parser import BillParser
from parser.policy_indexer import PolicyIndexer
from engine.exclusion_matcher import ExclusionMatcher
from engine.micro_compiler import MicroCompiler
from engine.rule_engine import RuleEngine
from engine.decision import DecisionMaker

BILL_PATH   = os.path.join(os.path.dirname(__file__), "data", "sample_bills", "Bill1.pdf")
POLICY_PATH = os.path.join(os.path.dirname(__file__), "data", "sample_policies", "policy1.pdf")

# ── Gold Standard ────────────────────────────────────────────────────
# Based on ICICI Lombard policy + Mata Laung Shree Hospital Bill1
gold_items = [
    {"description": "FILE CHARGE & REGISTRATION CHARGES", "expected_decision": "REJECT"},
    {"description": "PROFESSIONAL FEES CHARGES",           "expected_decision": "APPROVE"},
    {"description": "CONSULTATION CHARGE",                 "expected_decision": "APPROVE"},
    {"description": "EMERGENCY CHARGE",                    "expected_decision": "APPROVE"},
    {"description": "NEBULIZATION CHARGES",                "expected_decision": "APPROVE"},
    {"description": "MEDICINES & DRUGS",                   "expected_decision": "APPROVE"},
    {"description": "MEDICAL CONS. & DISPOSABLES",         "expected_decision": "APPROVE"},
    {"description": "OTHER CHARGES",                       "expected_decision": "APPROVE"},
    {"description": "Special Discount",                    "expected_decision": "APPROVE"},
]
gold_claims = [{"expected_status": "PARTIAL", "expected_total_approved": 36400}]

# ── Run Pipeline ─────────────────────────────────────────────────────
print("Step 1: Parsing bill...")
bp = BillParser()
bill_data = bp.parse(BILL_PATH)
items = bill_data["items"]
print(f"  → {len(items)} items extracted")
for it in items:
    print(f"    {it.item_id}: {it.description} = ₹{it.total:,.2f}")

print("\nStep 2: Indexing policy...")
idx = PolicyIndexer()
idx.parse_and_index(POLICY_PATH)
print(f"  → {len(idx.clauses)} clauses indexed")

print("\nStep 3: Building exclusion list...")
em = ExclusionMatcher()
em.extract_exclusions_from_clauses(idx.clauses)
print(f"  → {len(em.exclusion_items)} exclusion items extracted")

print("\nStep 4: Running MicroCompiler (2-tier)...")
mc = MicroCompiler(idx, em)
rules, mapping = mc.compile_rules_for_bill(items)
print(f"  → {len(rules)} rules generated")
print(f"  → Stats: {mc.pipeline_stats}")

print("\nStep 5: Applying RuleEngine...")
rule_engine = RuleEngine()
evaluated = rule_engine.evaluate(items, rules, mapping, {"ytd_approved": 0})

print("\nStep 6: Generating decision...")
dm = DecisionMaker([])
result = dm.generate_result("EVAL-001", bill_data["patient_id"], bill_data["admission_date"], evaluated)

print(f"\n  Status: {result.overall_status}")
print(f"  Claimed:  ₹{result.total_claimed:,.2f}")
print(f"  Approved: ₹{result.total_approved:,.2f}")

# ── Build Prediction Objects ─────────────────────────────────────────
predicted_items = []
for item in evaluated:
    predicted_items.append({
        "description": item.description,
        "decision": item.decision or "APPROVE",
        "decision_source": item.decision_source or "UNKNOWN",
        "approved_amount": item.approved_amount,
        "reason": item.reason or "",
    })

predicted_claims = [{
    "overall_status": result.overall_status,
    "total_approved": result.total_approved,
}]

# ── Evaluate ─────────────────────────────────────────────────────────
print("\n" + "="*60)
item_metrics   = compute_item_level_metrics(gold_items, predicted_items)
claim_metrics  = compute_claim_level_metrics(gold_claims, predicted_claims)
pipeline_dist  = compute_pipeline_distribution([{"items": predicted_items}])
determinism    = compute_determinism_score([result.total_approved])

print_evaluation_report(item_metrics, claim_metrics, pipeline_dist, determinism)

print("\n📋 ITEM-BY-ITEM BREAKDOWN:")
pred_lookup = {p["description"].lower().strip(): p for p in predicted_items}
for g in gold_items:
    key = g["description"].lower().strip()
    pred = pred_lookup.get(key)
    if pred:
        match = "✅" if pred["decision"].upper() == g["expected_decision"].upper() else "❌"
        print(f"  {match} {g['description'][:45]:45s} | Exp:{g['expected_decision']:7s} Got:{pred['decision']:7s} | Src:{pred['decision_source']}")
    else:
        print(f"  ❓ {g['description']:45s} | NOT FOUND IN PREDICTION")

print("\n📋 ALL PIPELINE ITEMS:")
for item in evaluated:
    print(f"  {item.item_id}: {item.description[:40]:40s} → {item.decision:7s} ₹{item.approved_amount:>9,.2f}  [{item.decision_source}]")

print(f"\nErrors: {mc.pipeline_stats['error_details']}")
