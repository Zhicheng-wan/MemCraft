"""
retrieval.py - BM25-based memory retrieval.

No embedding API calls needed = free retrieval!
Query = Goal Terms + Inventory Terms + Entity Terms
Retrieves Top-K from both Step Memory and Semantic Memory.
"""

import re
from rank_bm25 import BM25Okapi
from typing import List, Tuple


def tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer."""
    text = text.lower()
    tokens = re.findall(r'[a-z0-9_]+', text)
    return tokens


class BM25Retriever:
    """
    BM25-based retrieval over memory entries.
    Rebuilds index on each query (memories are small enough for this).
    """

    def retrieve(self, query_terms: List[str], entries: List[dict],
                 top_k: int = 5) -> List[Tuple[dict, float]]:
        """
        Retrieve top-k entries most relevant to query terms.

        Args:
            query_terms: List of query terms (goal + inventory + entity)
            entries: List of memory entry dicts, each with 'text' field
            top_k: Number of results to return

        Returns:
            List of (entry, score) tuples, sorted by relevance
        """
        if not entries or not query_terms:
            return []

        # Tokenize all entries
        corpus = [tokenize(e.get("text", "")) for e in entries]

        # Filter out empty docs
        valid = [(i, tokens) for i, tokens in enumerate(corpus) if tokens]
        if not valid:
            return []

        valid_indices, valid_corpus = zip(*valid)
        valid_entries = [entries[i] for i in valid_indices]

        # Build BM25 index
        bm25 = BM25Okapi(list(valid_corpus))

        # Query
        query_tokens = []
        for term in query_terms:
            query_tokens.extend(tokenize(term))

        if not query_tokens:
            return []

        scores = bm25.get_scores(query_tokens)

        # Rank and return top-k
        scored = list(zip(valid_entries, scores))
        scored.sort(key=lambda x: -x[1])

        return scored[:top_k]


def build_query_terms(goal: str, inventory_terms: List[str],
                      entity_terms: List[str]) -> List[str]:
    """
    Build BM25 query from goal + inventory + entity terms.
    Matching the slide: Query = Goal Terms + Inventory Terms + Entity Terms
    """
    terms = []

    # Goal terms (highest priority)
    terms.extend(tokenize(goal))

    # Inventory terms
    for item in inventory_terms:
        terms.extend(tokenize(item))

    # Entity terms
    for entity in entity_terms:
        terms.extend(tokenize(entity))

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for t in terms:
        if t not in seen:
            seen.add(t)
            unique.append(t)

    return unique
