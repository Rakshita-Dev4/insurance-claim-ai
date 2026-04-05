import logging
import re
from typing import List, Dict, Optional
from rapidfuzz import fuzz, process
from data.schema import PolicyClause

logger = logging.getLogger(__name__)

# Minimum fuzzy match score (using token_sort_ratio — full content comparison)
FUZZY_THRESHOLD = 80

# Words that are too common in hospital bills to be discriminating
# If the ONLY matching tokens are these, we reject the match
NOISE_WORDS = {
    "charges", "charge", "fees", "fee", "cost", "costs",
    "service", "services", "other", "others", "misc", "miscellaneous",
    "total", "amount", "payment", "billing",
}


class ExclusionMatcher:
    """
    Deterministic exclusion detection using fuzzy string matching.
    
    2-Stage matching with anti-noise filtering:
    Stage 1: token_sort_ratio must exceed threshold (full content comparison)
    Stage 2: Verify that the match isn't driven purely by noise words like "charges"
    
    This prevents false positives like:
    - "Room & Nursing Charges" matching "Nebulization Charges" (both contain "charges")
    - "Laboratory Charges" matching "Laundry Charges" (similar words + "charges")
    """

    def __init__(self):
        self.exclusion_items: List[Dict] = []
        self.exclusion_names: List[str] = []  # lowercase, for fuzzy matching
        self._seen_names: set = set()          # O(1) deduplication

    def extract_exclusions_from_clauses(self, clauses: List[PolicyClause]) -> None:
        """
        Scans all policy clauses to extract structured exclusion list items.
        Handles real PDF formats: "NUMBER SPACE ITEM_NAME" tables,
        numbered lists, bulleted lists, exclusion codes.
        """
        self.exclusion_items = []

        section_markers = [
            r"(?i)list\s*[iI]{1,4}\b",
            r"(?i)non[\s-]?payable",
            r"(?i)not\s+payable",
            r"(?i)items?\s+(?:which|that|for\s+which)\s+(?:are\s+)?(?:not|non)",
            r"(?i)permanent\s+exclusion",
            r"(?i)standard\s+exclusion",
        ]

        excl_code_pattern = r"(?i)(Code[\s-]?Excl\d+)"

        skip_items = {
            "sr", "no", "items", "list of non payable items", "sr. no items",
            "uin", "cin", "non payables", "below are",
        }

        for clause in clauses:
            text = clause.text
            is_exclusion_section = any(re.search(p, text) for p in section_markers)

            for match in re.finditer(excl_code_pattern, text):
                code = match.group(1).strip()
                desc_match = re.search(
                    re.escape(code) + r"[\s:\-]+(.+?)(?:\n|$)", text
                )
                desc = desc_match.group(1).strip() if desc_match else code
                self._add_exclusion(desc, f"Exclusion {code}", clause.page_num)

            if not is_exclusion_section:
                continue

            list_match = re.search(r"(?i)list\s*([iI]{1,4})\b", text)
            list_label = f"List {list_match.group(1).upper()}" if list_match else "Non-Payable"

            # PDF table format: "38 NEBULIZER KIT"
            pdf_table_pattern = r"(?:^|\n)\s*(\d{1,3})\s+([A-Z][^\n]{3,})"
            table_matches = re.findall(pdf_table_pattern, text)

            if len(table_matches) >= 3:
                for num, item_text in table_matches:
                    item_text = item_text.strip()
                    item_text = re.sub(r"[,;\.]+$", "", item_text).strip()
                    if item_text.lower() in skip_items or len(item_text) < 4:
                        continue
                    if item_text.startswith("UIN") or item_text.startswith("CIN"):
                        continue
                    if len(item_text) > 200:
                        continue
                    self._add_exclusion(item_text, list_label, clause.page_num)
            else:
                # Fallback: standard numbered items with punctuation
                std_pattern = r"(?:^|\n)\s*(?:\d{1,3}[\.\)\:]|\(\d{1,3}\))\s*(.+?)(?=\n\s*(?:\d{1,3}[\.\)\:]|\(\d{1,3}\))|\n\n|$)"
                for match in re.finditer(std_pattern, text, re.DOTALL):
                    item_text = match.group(1).strip()
                    item_text = re.sub(r"[,;\.]+$", "", item_text).strip()
                    if len(item_text) > 5 and len(item_text) < 200:
                        if item_text.lower() not in skip_items:
                            self._add_exclusion(item_text, list_label, clause.page_num)

        logger.info(f"ExclusionMatcher: Extracted {len(self.exclusion_items)} exclusion items from policy.")
        self.exclusion_names = [e["name"].lower() for e in self.exclusion_items]

    def _add_exclusion(self, name: str, source: str, page: int) -> None:
        """Adds a unique exclusion item. O(1) dedup via set."""
        normalized = name.lower().strip()
        if normalized in self._seen_names:
            return
        self._seen_names.add(normalized)
        self.exclusion_items.append({
            "name": name.strip(),
            "source": source,
            "page": page
        })

    def _extract_meaningful_tokens(self, text: str) -> set:
        """Extracts tokens that are NOT noise words."""
        tokens = set(re.findall(r'[a-z]+', text.lower()))
        return tokens - NOISE_WORDS

    def _validate_match(self, query: str, matched_name: str) -> bool:
        """
        Anti-noise validation: Ensures the match is based on meaningful
        content words, not just common terms like 'charges' or 'fees'.
        
        Returns True only if the meaningful tokens overlap sufficiently.
        """
        query_meaningful = self._extract_meaningful_tokens(query)
        match_meaningful = self._extract_meaningful_tokens(matched_name)

        if not query_meaningful or not match_meaningful:
            return False

        # At least one meaningful (non-noise) token must overlap
        overlap = query_meaningful & match_meaningful
        if not overlap:
            return False

        # If both sides have 2+ meaningful tokens, require at least 2 overlapping
        if len(query_meaningful) >= 2 and len(match_meaningful) >= 2:
            if len(overlap) < 2:
                return False

        # The overlap must cover at least 60% of the shorter set's meaningful tokens
        min_tokens = min(len(query_meaningful), len(match_meaningful))
        if min_tokens > 0 and len(overlap) / min_tokens < 0.6:
            return False

        return True

    def match(self, bill_item_description: str) -> Optional[Dict]:
        """
        2-stage deterministic matching:
        Stage 1: token_sort_ratio score check (full-string comparison)
        Stage 2: Anti-noise validation (meaningful token overlap)
        """
        if not self.exclusion_names:
            return None

        query = bill_item_description.lower().strip()

        # Skip items that are clearly not excludable
        skip_patterns = ["discount", "deduction", "rebate", "adjustment"]
        if any(s in query for s in skip_patterns):
            return None

        # Stage 1: Get best match using token_set_ratio
        # token_set_ratio handles subset relationships:
        # "FILE CHARGE & REGISTRATION CHARGES" contains "REGISTRATION CHARGES"
        # → token_set gives 100% because all of the exclusion's tokens are present
        result = process.extractOne(
            query,
            self.exclusion_names,
            scorer=fuzz.token_set_ratio,
            score_cutoff=FUZZY_THRESHOLD
        )

        if result is None:
            return None

        matched_name, score, idx = result

        # Stage 2: Anti-noise validation
        if not self._validate_match(query, matched_name):
            logger.info(
                f"ExclusionMatcher: REJECTED noise match '{bill_item_description}' → "
                f"'{matched_name}' (score={score:.1f} but failed noise filter)"
            )
            return None

        exclusion_info = self.exclusion_items[idx]

        logger.info(
            f"ExclusionMatcher: DETERMINISTIC MATCH '{bill_item_description}' → "
            f"'{exclusion_info['name']}' (score={score:.1f}, "
            f"source={exclusion_info['source']}, page={exclusion_info['page']})"
        )

        return {
            "matched_exclusion": exclusion_info["name"],
            "source": exclusion_info["source"],
            "page": exclusion_info["page"],
            "score": score,
            "rule_id": f"{exclusion_info['source']}, Page {exclusion_info['page']}, Item: {exclusion_info['name']}",
        }

    def match_all(self, bill_items_descriptions: List[str]) -> Dict[str, Optional[Dict]]:
        """Batch match all bill items."""
        results = {}
        for desc in bill_items_descriptions:
            results[desc.lower().strip()] = self.match(desc)
        return results
