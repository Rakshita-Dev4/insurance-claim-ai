import logging
from typing import List
from data.schema import ClaimItem, PolicyClause

logger = logging.getLogger(__name__)


class Reconciler:
    """
    Lightweight item reconciler.
    
    In the current architecture, the heavy lifting is done by:
    - ExclusionMatcher (deterministic fuzzy matching)
    - MicroCompiler (LLM-driven RAG for ambiguous items)
    
    The Reconciler serves as a clean integration point for any
    future pre-processing logic (code normalization, deduplication, etc.)
    """

    def __init__(self, policy_clauses: List[PolicyClause]):
        self.clauses = policy_clauses

    def reconcile_item(self, item: ClaimItem) -> ClaimItem:
        """Normalize and prepare item for downstream processing."""
        # Trim whitespace from description
        if item.description:
            item.description = item.description.strip()

        # Ensure total is consistent
        if item.total == 0 and item.unit_cost > 0 and item.qty > 0:
            item.total = item.unit_cost * item.qty
            logger.info(f"Reconciler: Computed total for '{item.description}': {item.total}")

        return item
