"""
BM25 Retrieval Engine.

Builds a query from Goal Terms + Inventory Terms + Entity Terms,
then retrieves Top-K entries from both Step Memory (Mstep)
and Semantic Memory (Msem).

Includes a self-contained BM25Okapi implementation (no external dependency).
"""
import math
from typing import List, Tuple
from collections import Counter

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import BM25_TOP_K
from memory.step_memory import StepMemory
from memory.semantic_memory import SemanticMemory


# ─── Built-in BM25 (no pip dependency) ────────────────────────────

class BM25Okapi:
    """
    Okapi BM25 ranking function. Pure Python, zero dependencies.
    Drop-in replacement for rank_bm25.BM25Okapi.
    """
    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.doc_len = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_len) / len(corpus) if corpus else 1.0
        self.n_docs = len(corpus)
        
        # Document frequency for each term
        self.df = Counter()
        for doc in corpus:
            for term in set(doc):
                self.df[term] += 1
        
        # Term frequencies per document
        self.tf = [Counter(doc) for doc in corpus]
    
    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)
    
    def get_scores(self, query: List[str]) -> List[float]:
        scores = [0.0] * self.n_docs
        for term in query:
            idf = self._idf(term)
            for i in range(self.n_docs):
                tf = self.tf[i].get(term, 0)
                dl = self.doc_len[i]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[i] += idf * (tf * (self.k1 + 1)) / denom if denom > 0 else 0
        return scores


class BM25Retriever:
    """
    BM25-based retrieval over heterogeneous memory stores.
    
    Query construction (from slides):
        query = Goal Terms + Inventory Terms + Entity Terms
    
    Retrieves Top-K from:
        - Mstep (step memory): recent trajectory entries
        - Msem (semantic memory): consolidated rules
    """
    
    def __init__(self, top_k: int = BM25_TOP_K):
        self.top_k = top_k
    
    def build_query(self, goal: str, inventory_terms: List[str],
                    entity_terms: List[str]) -> List[str]:
        """
        Construct BM25 query tokens.
        
        Query = Goal Terms + Inventory Terms + Entity Terms
        """
        tokens = []
        
        # Goal terms (tokenized)
        tokens.extend(goal.lower().split())
        
        # Inventory terms
        for item in inventory_terms:
            tokens.extend(item.lower().replace("_", " ").split())
        
        # Entity terms
        for entity in entity_terms:
            tokens.extend(entity.lower().replace("_", " ").split())
        
        # Deduplicate while preserving order
        seen = set()
        unique = []
        for t in tokens:
            if t not in seen and len(t) > 1:  # skip single chars
                seen.add(t)
                unique.append(t)
        
        return unique
    
    def retrieve_from_step_memory(
        self, query_tokens: List[str], step_memory: StepMemory
    ) -> List[str]:
        """Retrieve top-K relevant entries from step memory."""
        if len(step_memory) == 0 or not query_tokens:
            return []
        
        corpus = step_memory.get_all_tokenized()
        texts = step_memory.get_all_texts()
        
        if not corpus:
            return []
        
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(query_tokens)
        
        # Get top-K indices
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:self.top_k]
        
        # Filter out zero-score entries
        return [texts[i] for i in top_indices if scores[i] > 0]
    
    def retrieve_from_semantic_memory(
        self, query_tokens: List[str], semantic_memory: SemanticMemory
    ) -> List[str]:
        """Retrieve top-K relevant rules from semantic memory."""
        if len(semantic_memory) == 0 or not query_tokens:
            return []
        
        corpus = semantic_memory.get_all_tokenized()
        texts = semantic_memory.get_all_texts()
        
        if not corpus:
            return []
        
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(query_tokens)
        
        # Get top-K indices
        k = min(self.top_k, len(scores))
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:k]
        
        return [texts[i] for i in top_indices if scores[i] > 0]
    
    def retrieve(
        self,
        goal: str,
        inventory_terms: List[str],
        entity_terms: List[str],
        step_memory: StepMemory,
        semantic_memory: SemanticMemory,
    ) -> Tuple[List[str], List[str]]:
        """
        Full retrieval pipeline.
        
        Returns:
            (step_results, semantic_results): Retrieved texts from each store.
        """
        query_tokens = self.build_query(goal, inventory_terms, entity_terms)
        
        step_results = self.retrieve_from_step_memory(query_tokens, step_memory)
        semantic_results = self.retrieve_from_semantic_memory(
            query_tokens, semantic_memory
        )
        
        return step_results, semantic_results
