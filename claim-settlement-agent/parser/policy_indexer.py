import logging
import numpy as np
import pdfplumber
from typing import List, Optional
from data.schema import PolicyClause
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Singleton embedding model to avoid loading 80MB per request
_embedding_model: Optional[SentenceTransformer] = None


def get_embedding_model() -> SentenceTransformer:
    """Lazy singleton for the sentence transformer model."""
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading sentence-transformers model (one-time initialization)...")
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Sentence-transformers model loaded successfully.")
    return _embedding_model


class PolicyIndexer:
    """
    Hybrid BM25 + Semantic retrieval with Reciprocal Rank Fusion (RRF).
    
    Architecture:
    1. Extracts policy PDF text and chunks by paragraph
    2. Builds a BM25Okapi lexical index (exact token matching)
    3. Builds a dense semantic index via sentence-transformers (concept matching)
    4. At retrieval time, merges both result sets via RRF for best-of-both-worlds
    
    This ensures that "Consultation Fee" still retrieves "Doctor's Professional Charges"
    (semantic match) while "List II" still retrieves the exact exclusion list (lexical match).
    """

    def __init__(self):
        self.clauses: List[PolicyClause] = []
        self.bm25_index: Optional[BM25Okapi] = None
        self.schedule_text: str = ""
        self.clause_embeddings: Optional[np.ndarray] = None
        self.embedding_model: Optional[SentenceTransformer] = None

    def parse_and_index(self, file_path: str, index_dir: str = "") -> None:
        """
        Extracts raw text from the Policy PDF, chunks by paragraph,
        and builds both BM25 and semantic indexes.
        """
        logger.info(f"Indexing policy document: {file_path}")
        self.clauses = []
        clause_id_counter = 1

        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text()
                    if not text:
                        continue

                    # Split into paragraphs
                    paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 30]
                    if not paragraphs:
                        paragraphs = [p.strip() for p in text.split('\n') if len(p.strip()) > 30]

                    # First 3 pages are always schedule/base limits
                    if page_num <= 3:
                        self.schedule_text += text + "\n"

                    for para_num, para_text in enumerate(paragraphs, start=1):
                        self.clauses.append(PolicyClause(
                            clause_id=f"C{clause_id_counter}",
                            page_num=page_num,
                            para_num=para_num,
                            text=para_text,
                            coverage_type="general",
                            limit_amount=None,
                            is_exclusion=False
                        ))
                        clause_id_counter += 1

            logger.info(f"Extracted {len(self.clauses)} clause chunks from policy.")

            # ── Build BM25 Lexical Index ──
            tokenized_corpus = [c.text.lower().split() for c in self.clauses]
            if tokenized_corpus:
                self.bm25_index = BM25Okapi(tokenized_corpus)
                logger.info("BM25 lexical index built.")

            # ── Build Semantic Embedding Index ──
            self.embedding_model = get_embedding_model()
            corpus_texts = [c.text for c in self.clauses]
            if corpus_texts:
                self.clause_embeddings = self.embedding_model.encode(
                    corpus_texts, show_progress_bar=False, normalize_embeddings=True
                )
                logger.info(f"Semantic embedding index built ({self.clause_embeddings.shape}).")

        except Exception as e:
            logger.error(f"Error indexing policy PDF: {e}")

    def retrieve_relevant_clauses(self, query: str, top_k: int = 3) -> str:
        """
        Hybrid retrieval using Reciprocal Rank Fusion (RRF).
        
        1. BM25 retrieves top-N by lexical match
        2. Semantic search retrieves top-N by cosine similarity
        3. RRF merges both rankings into a single fused ranking
        
        Returns: Formatted string of schedule summary + top retrieved paragraphs.
        """
        if not self.clauses:
            return self.schedule_text[:800]

        rrf_k = 60  # RRF smoothing constant

        # ── BM25 Retrieval ──
        bm25_scores = {}
        if self.bm25_index:
            tokenized_query = query.lower().split()
            bm25_raw_scores = self.bm25_index.get_scores(tokenized_query)
            bm25_ranked = sorted(
                range(len(bm25_raw_scores)),
                key=lambda i: bm25_raw_scores[i],
                reverse=True
            )
            for rank, idx in enumerate(bm25_ranked[:top_k * 2]):
                bm25_scores[idx] = 1.0 / (rrf_k + rank + 1)

        # ── Semantic Retrieval ──
        semantic_scores = {}
        if self.clause_embeddings is not None and self.embedding_model is not None:
            query_embedding = self.embedding_model.encode(
                [query], normalize_embeddings=True
            )
            cosine_scores = np.dot(self.clause_embeddings, query_embedding.T).flatten()
            semantic_ranked = sorted(
                range(len(cosine_scores)),
                key=lambda i: cosine_scores[i],
                reverse=True
            )
            for rank, idx in enumerate(semantic_ranked[:top_k * 2]):
                semantic_scores[idx] = 1.0 / (rrf_k + rank + 1)

        # ── Reciprocal Rank Fusion ──
        all_indices = set(bm25_scores.keys()) | set(semantic_scores.keys())
        fused_scores = {}
        for idx in all_indices:
            fused_scores[idx] = bm25_scores.get(idx, 0.0) + semantic_scores.get(idx, 0.0)

        top_indices = sorted(fused_scores.keys(), key=lambda i: fused_scores[i], reverse=True)[:top_k]

        # ── Format Output (compact to save tokens) ──
        schedule_summary = self.schedule_text[:800] if self.schedule_text else ""
        payload = f"--- Policy Schedule (Summary) ---\n{schedule_summary}\n\n"
        payload += f"--- Retrieved Policy Clauses for '{query}' ---\n"

        for idx in top_indices:
            c = self.clauses[idx]
            payload += f"[Pg {c.page_num}, Para {c.para_num}]: {c.text[:500]}\n\n"

        return payload
