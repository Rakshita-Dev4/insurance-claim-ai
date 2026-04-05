import json
import logging
import time
from typing import List, Dict, Tuple, Optional
from data.schema import ClaimItem, PolicyRuleDSL, LLMRulesResponse
from parser.policy_indexer import PolicyIndexer
from engine.exclusion_matcher import ExclusionMatcher
from groq import Groq
from config import settings

logger = logging.getLogger(__name__)

MODEL = "llama-3.3-70b-versatile"
MAX_RETRIES = 3
RETRY_WAIT_SEC = 8  # Wait between retries on rate limit
THROTTLE_SEC = 5    # Wait between successive LLM calls to stay under TPM


class MicroCompiler:
    """
    Task-Driven RAG Micro-Compiler with Deterministic Pre-filtering.
    
    Pipeline:
    1. ExclusionMatcher checks each item deterministically (no LLM)
    2. Items that match exclusions are tagged immediately
    3. Only ambiguous items are sent to the LLM via targeted BM25+Semantic retrieval
    4. LLM output is validated through Pydantic schemas
    5. Inter-call throttling to respect Groq TPM limits
    """

    def __init__(self, indexer: PolicyIndexer, exclusion_matcher: ExclusionMatcher):
        self.indexer = indexer
        self.exclusion_matcher = exclusion_matcher
        self.groq_client = Groq(api_key=settings.groq_api_key) if settings.groq_api_key else None
        self.rule_counter = 1
        self._call_count = 0
        self.pipeline_stats = {
            "deterministic_exclusions": 0,
            "llm_exclusions": 0,
            "llm_approvals": 0,
            "llm_calls_made": 0,
            "llm_errors": 0,
            "llm_model_used": MODEL,
            "error_details": [],
        }

    def _call_llm(self, prompt: str) -> Optional[str]:
        """
        Calls llama-3.3-70b with retry + throttling.
        Waits between calls to avoid hitting Groq TPM limits.
        """
        # Throttle: wait between successive calls
        if self._call_count > 0:
            time.sleep(THROTTLE_SEC)

        self._call_count += 1
        last_error = None

        for attempt in range(MAX_RETRIES):
            try:
                self.pipeline_stats["llm_calls_made"] += 1
                response = self.groq_client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=MODEL,
                    temperature=0.0,
                    response_format={"type": "json_object"}
                )
                return response.choices[0].message.content

            except Exception as e:
                error_str = str(e)
                last_error = error_str
                logger.warning(f"LLM call failed (attempt {attempt+1}/{MAX_RETRIES}): {error_str[:200]}")

                if "rate_limit" in error_str.lower() or "429" in error_str:
                    wait = RETRY_WAIT_SEC * (attempt + 1)  # Exponential backoff
                    logger.info(f"Rate limited. Waiting {wait}s before retry...")
                    time.sleep(wait)
                elif attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_WAIT_SEC)

        self.pipeline_stats["error_details"].append(f"Failed after {MAX_RETRIES} retries: {last_error[:150] if last_error else 'unknown'}")
        return None

    def compile_rules_for_bill(self, bill_items: List[ClaimItem]) -> Tuple[List[Dict], Dict[str, List[str]]]:
        """
        Executes a 2-tier adjudication pipeline:
        Tier 1: Deterministic fuzzy matching against extracted exclusion lists
        Tier 2: LLM-driven RAG for items that pass Tier 1
        """
        if not bill_items:
            return [], {}

        all_rules = []
        item_mapping = {item.item_id: [] for item in bill_items}

        # Deduplicate descriptions to save API calls
        unique_queries = {}
        for item in bill_items:
            desc_key = item.description.lower().strip()
            if desc_key not in unique_queries:
                unique_queries[desc_key] = []
            unique_queries[desc_key].append(item)

        for query_desc, items_group in unique_queries.items():
            item_ids = [item.item_id for item in items_group]

            # ── TIER 1: Deterministic Exclusion Matching ──
            det_match = self.exclusion_matcher.match(query_desc)
            if det_match:
                rule = {
                    "rule_id": det_match["rule_id"],
                    "operation": "EXCLUSION",
                    "semantic_targets": [query_desc],
                    "limit_value": None,
                    "percentage": None,
                    "confidence": 1.0,   # Deterministic — always trusted
                }
                all_rules.append(rule)
                for iid in item_ids:
                    item_mapping[iid].append(rule["rule_id"])
                self.rule_counter += 1
                self.pipeline_stats["deterministic_exclusions"] += 1

                for item in items_group:
                    item.decision_source = "DETERMINISTIC"

                logger.info(f"TIER-1 DETERMINISTIC: '{query_desc}' excluded via {det_match['rule_id']}")
                continue

            # ── TIER 2: LLM-Driven RAG (only for ambiguous items) ──
            if not self.groq_client:
                self.pipeline_stats["error_details"].append("No Groq API key configured")
                continue

            retrieved_context = self.indexer.retrieve_relevant_clauses(query_desc, top_k=5)

            prompt = f"""You are a precise medical insurance underwriter.
Your task is to analyze ONLY the provided subset of policy clauses to determine HOW to adjudicate the specific hospital charge: '{query_desc}'.

Retrieved Policy Context (Schedules + Top 5 relevant paragraphs via Hybrid BM25+Semantic Retrieval):
{retrieved_context}

Determine if this specific charge ({query_desc}) is mathematically capped, proportionally deducted, explicitly excluded, or subject to copay according to the context provided.

IMPORTANT RULES:
- Ignore any clause that specifies "Worldwide", "International", or geographies outside domestic territory.
- ANTI-CONTAMINATION: You are ONLY evaluating '{query_desc}'. If the retrieved text describes limits on a totally different concept (e.g., 'Emergency charges' when '{query_desc}' is 'Consultation'), you MUST ignore that text.
- THE ADMISSIBILITY FUNNEL:
   1. DEFINITION GATE: Does '{query_desc}' qualify as a "Medical Expense" or "Medically Necessary Treatment"? If clearly non-medical (e.g., Aromatherapy, Spa), create an EXCLUSION rule.
   2. EXCLUSION GATE: Is '{query_desc}' explicitly listed in Permanent Exclusions, Standard Exclusions (Code-ExclXX), or Non-Payables Lists (List I, II, III, IV)? If yes, create an EXCLUSION rule and cite the exact code/list.
   3. DEDUCTION GATE: If it passes Gate 1 and 2, it is ADMISSIBLE. Check if subject to Co-Payments or Proportionate Deductions (Room Rent limits), and generate those rules if so.

Return STRICTLY a JSON object:
{{
  "rules": [
    {{
       "rule_id": "Cite the exact legal reference (e.g., 'Code-Excl01', 'List IV, Page 29'). If none found, use 'Page X, Para Y'.",
       "operation": "EXCLUSION" or "LIMIT_CAP" or "PROPORTIONAL_DEDUCTION" or "COPAY" or "GLOBAL_CAP",
       "limit_value": null or float,
       "percentage": null or float (as decimal, e.g. 0.10 for 10%),
       "confidence": float between 0.0 and 1.0 indicating how confident you are
    }}
  ]
}}
If no special limits or exclusions apply, return {{"rules": []}}.
Generate a GLOBAL_CAP rule if you see an annual/sum-insured limit.
"""
            raw_content = self._call_llm(prompt)

            if raw_content is None:
                self.pipeline_stats["llm_errors"] += 1
                for item in items_group:
                    item.decision_source = "LLM_ERROR"
                logger.error(f"All LLM models failed for '{query_desc}'")
                continue

            try:
                parsed = json.loads(raw_content)

                # ── Pydantic Validation ──
                validated = LLMRulesResponse(**parsed)

                for validated_rule in validated.rules:
                    rule_id = validated_rule.rule_id
                    if not rule_id or rule_id == "UNKNOWN" or rule_id.lower().startswith("cite") or rule_id.lower().startswith("extract"):
                        rule_id = f"LLM-Rule-{self.rule_counter}"

                    rule = {
                        "rule_id": rule_id,
                        "operation": validated_rule.operation,
                        "semantic_targets": [query_desc],
                        "limit_value": validated_rule.limit_value,
                        "percentage": validated_rule.percentage,
                        "confidence": validated_rule.confidence,
                    }

                    all_rules.append(rule)
                    for iid in item_ids:
                        item_mapping[iid].append(rule_id)
                    self.rule_counter += 1

                    if validated_rule.operation == "EXCLUSION":
                        self.pipeline_stats["llm_exclusions"] += 1
                    else:
                        self.pipeline_stats["llm_approvals"] += 1

                # If LLM returned empty rules, item is approved (no restrictions found)
                if len(validated.rules) == 0:
                    self.pipeline_stats["llm_approvals"] += 1

                for item in items_group:
                    item.decision_source = "LLM"

                logger.info(f"TIER-2 LLM: Compiled {len(validated.rules)} rules for '{query_desc}'.")

            except json.JSONDecodeError as e:
                logger.error(f"LLM returned invalid JSON for '{query_desc}': {raw_content[:200]}")
                self.pipeline_stats["llm_errors"] += 1
                self.pipeline_stats["error_details"].append(f"Invalid JSON for '{query_desc}': {str(e)}")
                for item in items_group:
                    item.decision_source = "LLM_ERROR"

            except Exception as e:
                logger.error(f"MicroCompiler processing failed for '{query_desc}': {e}")
                self.pipeline_stats["llm_errors"] += 1
                self.pipeline_stats["error_details"].append(f"Processing error for '{query_desc}': {str(e)}")
                for item in items_group:
                    item.decision_source = "LLM_ERROR"

        logger.info(f"Pipeline Stats: {self.pipeline_stats}")
        return all_rules, item_mapping
