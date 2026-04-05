"""
Model Evaluation & Pipeline Metrics for the Insurance Claim Settlement Agent.

Computes:
- Per-item accuracy (did each item get the correct APPROVE/REJECT decision?)
- Claim-level status accuracy
- Precision, Recall, F1 (binary: should-be-rejected vs was-rejected)
- Determinism score (run same input N times, measure output variance)
- Pipeline layer distribution (what % of decisions were deterministic vs LLM?)
"""
from typing import List, Dict, Any


def compute_item_level_metrics(
    gold_items: List[Dict[str, str]],
    predicted_items: List[Dict[str, str]]
) -> Dict[str, float]:
    """
    Per-item evaluation.
    gold_items: [{"description": "File Charges", "expected_decision": "REJECT"}, ...]
    predicted_items: [{"description": "File Charges", "decision": "REJECT"}, ...]
    """
    if not gold_items or not predicted_items:
        return {"item_accuracy": 0.0, "item_precision": 0.0, "item_recall": 0.0, "item_f1": 0.0}

    # Build lookup from predicted
    pred_lookup = {p["description"].lower().strip(): p.get("decision", "APPROVE") for p in predicted_items}

    tp, fp, fn, tn = 0, 0, 0, 0
    correct = 0

    for gold in gold_items:
        g_desc = gold["description"].lower().strip()
        g_expected = gold["expected_decision"].upper()
        p_actual = pred_lookup.get(g_desc, "APPROVE").upper()

        if g_expected == p_actual:
            correct += 1

        # Binary: REJECT = Positive, APPROVE/PARTIAL = Negative
        g_binary = 1 if g_expected == "REJECT" else 0
        p_binary = 1 if p_actual == "REJECT" else 0

        if g_binary == 1 and p_binary == 1:
            tp += 1
        elif g_binary == 0 and p_binary == 1:
            fp += 1
        elif g_binary == 1 and p_binary == 0:
            fn += 1
        else:
            tn += 1

    accuracy = correct / len(gold_items) if gold_items else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "item_accuracy": round(accuracy, 4),
        "item_precision": round(precision, 4),
        "item_recall": round(recall, 4),
        "item_f1": round(f1, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def compute_claim_level_metrics(
    gold_standards: List[Dict],
    predictions: List[Dict]
) -> Dict[str, float]:
    """
    Claim-level evaluation.
    gold_standards: [{"expected_status": "PARTIAL", "expected_total_approved": 36400}, ...]
    predictions: [{"overall_status": "PARTIAL", "total_approved": 36400}, ...]
    """
    correct_status = 0
    total_absolute_error = 0.0

    for gold, pred in zip(gold_standards, predictions):
        if gold["expected_status"] == pred["overall_status"]:
            correct_status += 1

        expected_amount = gold.get("expected_total_approved", 0)
        actual_amount = pred.get("total_approved", 0)
        total_absolute_error += abs(expected_amount - actual_amount)

    n = len(gold_standards)
    return {
        "claim_status_accuracy": round(correct_status / n, 4) if n > 0 else 0,
        "mean_absolute_error": round(total_absolute_error / n, 2) if n > 0 else 0,
    }


def compute_pipeline_distribution(predictions: List[Dict]) -> Dict[str, float]:
    """
    Analyzes what percentage of decisions came from deterministic vs LLM paths.
    """
    total_items = 0
    deterministic_count = 0
    llm_count = 0
    error_count = 0

    for pred in predictions:
        for item in pred.get("items", []):
            total_items += 1
            source = item.get("decision_source", "UNKNOWN")
            if source == "DETERMINISTIC":
                deterministic_count += 1
            elif source == "LLM":
                llm_count += 1
            elif source == "LLM_ERROR":
                error_count += 1

    if total_items == 0:
        return {"deterministic_pct": 0, "llm_pct": 0, "error_pct": 0}

    return {
        "total_items": total_items,
        "deterministic_pct": round(deterministic_count / total_items * 100, 1),
        "llm_pct": round(llm_count / total_items * 100, 1),
        "error_pct": round(error_count / total_items * 100, 1),
    }


def compute_determinism_score(run_results: List[float]) -> Dict[str, float]:
    """
    Measures variance across multiple runs of the same input.
    run_results: list of total_approved values from N runs
    Perfect determinism = 0.0 variance
    """
    if len(run_results) < 2:
        return {"determinism_variance": 0.0, "determinism_score": 1.0}

    mean = sum(run_results) / len(run_results)
    variance = sum((x - mean) ** 2 for x in run_results) / len(run_results)
    # Normalize: if all identical, score = 1.0
    max_possible_variance = mean ** 2 if mean > 0 else 1
    score = max(0, 1.0 - (variance / max_possible_variance))

    return {
        "determinism_variance": round(variance, 2),
        "determinism_score": round(score, 4),
        "run_values": run_results,
    }


def print_evaluation_report(
    item_metrics: Dict,
    claim_metrics: Dict,
    pipeline_dist: Dict,
    determinism: Dict
):
    """Pretty-prints the full evaluation report."""
    print("=" * 60)
    print("   INSURANCE AI ADJUDICATOR — EVALUATION REPORT")
    print("=" * 60)

    print("\n📊 ITEM-LEVEL METRICS")
    print(f"   Accuracy:  {item_metrics['item_accuracy']:.2%}")
    print(f"   Precision: {item_metrics['item_precision']:.2%}")
    print(f"   Recall:    {item_metrics['item_recall']:.2%}")
    print(f"   F1 Score:  {item_metrics['item_f1']:.2%}")
    print(f"   Confusion: TP={item_metrics['tp']} FP={item_metrics['fp']} FN={item_metrics['fn']} TN={item_metrics['tn']}")

    print("\n📋 CLAIM-LEVEL METRICS")
    print(f"   Status Accuracy:      {claim_metrics['claim_status_accuracy']:.2%}")
    print(f"   Mean Absolute Error:  ₹{claim_metrics['mean_absolute_error']:,.2f}")

    print("\n⚙️  PIPELINE DISTRIBUTION")
    print(f"   Total Items Processed:  {pipeline_dist.get('total_items', 'N/A')}")
    print(f"   Deterministic Decisions: {pipeline_dist.get('deterministic_pct', 0)}%")
    print(f"   LLM Decisions:          {pipeline_dist.get('llm_pct', 0)}%")
    print(f"   LLM Errors:             {pipeline_dist.get('error_pct', 0)}%")

    print("\n🎯 DETERMINISM SCORE")
    print(f"   Score:    {determinism.get('determinism_score', 'N/A')}")
    print(f"   Variance: {determinism.get('determinism_variance', 'N/A')}")
    if determinism.get("run_values"):
        print(f"   Runs:     {determinism['run_values']}")

    print("\n" + "=" * 60)


# ─── Example Usage ───────────────────────────────────────────────────

if __name__ == "__main__":
    # Gold standard for Bill1.pdf + policy1.pdf
    gold_items = [
        {"description": "FILE CHARGE & REGISTRATION CHARGES", "expected_decision": "REJECT"},
        {"description": "PROFESSIONAL FEES CHARGES", "expected_decision": "APPROVE"},
        {"description": "CONSULTATION CHARGE", "expected_decision": "APPROVE"},
        {"description": "EMERGENCY CHARGE", "expected_decision": "APPROVE"},
        {"description": "NEBULIZATION CHARGES", "expected_decision": "APPROVE"},
        {"description": "MEDICINES & DRUGS", "expected_decision": "APPROVE"},
        {"description": "MEDICAL CONS. & DISPOSABLES", "expected_decision": "APPROVE"},
        {"description": "OTHER CHARGES", "expected_decision": "APPROVE"},
        {"description": "Special Discount", "expected_decision": "APPROVE"},
    ]

    # Simulated prediction from a pipeline run
    predicted_items = [
        {"description": "FILE CHARGE & REGISTRATION CHARGES", "decision": "REJECT", "decision_source": "DETERMINISTIC"},
        {"description": "PROFESSIONAL FEES CHARGES", "decision": "APPROVE", "decision_source": "LLM"},
        {"description": "CONSULTATION CHARGE", "decision": "APPROVE", "decision_source": "LLM"},
        {"description": "EMERGENCY CHARGE", "decision": "APPROVE", "decision_source": "LLM"},
        {"description": "NEBULIZATION CHARGES", "decision": "APPROVE", "decision_source": "LLM"},
        {"description": "MEDICINES & DRUGS", "decision": "APPROVE", "decision_source": "LLM"},
        {"description": "MEDICAL CONS. & DISPOSABLES", "decision": "APPROVE", "decision_source": "LLM"},
        {"description": "OTHER CHARGES", "decision": "APPROVE", "decision_source": "LLM"},
        {"description": "Special Discount", "decision": "APPROVE", "decision_source": "LLM"},
    ]

    gold_claims = [{"expected_status": "PARTIAL", "expected_total_approved": 36400}]
    predicted_claims = [{"overall_status": "PARTIAL", "total_approved": 36400}]

    item_metrics = compute_item_level_metrics(gold_items, predicted_items)
    claim_metrics = compute_claim_level_metrics(gold_claims, predicted_claims)
    pipeline_dist = compute_pipeline_distribution([{"items": predicted_items}])
    determinism = compute_determinism_score([36400, 36400, 36400])

    print_evaluation_report(item_metrics, claim_metrics, pipeline_dist, determinism)
