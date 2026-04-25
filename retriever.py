"""
retriever.py
────────────
Part B: Custom Retrieval System
- Sentence-transformer embedding pipeline (no LangChain/LlamaIndex)
- FAISS vector storage
- Top-K retrieval with similarity scoring
- Hybrid search: vector similarity + keyword overlap (BM25-lite)
- Failure case detection + fix via query expansion
"""

import os
import re
import json
import math
import logging
import pickle
import numpy as np
from collections import Counter

import faiss
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_NAME      = "all-MiniLM-L6-v2"   # 384-dim, fast, free, no API key needed
INDEX_PATH      = "data/faiss.index"
CHUNKS_PATH     = "data/chunks.pkl"
HYBRID_ALPHA    = 0.70   # weight for vector score (1-alpha = keyword weight)
LOW_SCORE_THRESH = 0.35  # below this → flag as potential failure case


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 ── Embedding Pipeline
# ══════════════════════════════════════════════════════════════════════════════

class EmbeddingPipeline:
    """Wraps sentence-transformers for consistent embed calls."""

    def __init__(self, model_name: str = MODEL_NAME):
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dim   = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.dim}")

    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of texts → float32 numpy array of shape (N, dim).
        Normalised to unit length so dot-product == cosine similarity.
        """
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        vectors = self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,   # L2-normalise → cosine via dot product
            convert_to_numpy=True,
        )
        return vectors.astype(np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        """Convenience: embed a single string → shape (1, dim)."""
        return self.embed([text])


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 ── Vector Store (FAISS)
# ══════════════════════════════════════════════════════════════════════════════

class VectorStore:
    """
    FAISS-backed vector store.
    Stores normalised embeddings + parallel list of chunk metadata.
    Uses IndexFlatIP (inner product on normalised vectors = cosine similarity).
    """

    def __init__(self, dim: int):
        self.dim    = dim
        self.index  = faiss.IndexFlatIP(dim)   # inner product on L2-normalised vecs
        self.chunks : list[dict] = []           # parallel list, same order as FAISS

    # ── Build ──
    def add_chunks(self, chunks: list[dict], embedder: EmbeddingPipeline):
        """Embed all chunks and add to the FAISS index."""
        logger.info(f"Embedding {len(chunks)} chunks…")
        texts    = [c["text"] for c in chunks]
        vectors  = embedder.embed(texts)
        self.index.add(vectors)
        self.chunks.extend(chunks)
        logger.info(f"Index size: {self.index.ntotal} vectors")

    # ── Persist ──
    def save(self, index_path: str = INDEX_PATH, chunks_path: str = CHUNKS_PATH):
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(self.index, index_path)
        with open(chunks_path, "wb") as f:
            pickle.dump(self.chunks, f)
        logger.info(f"Saved index → {index_path}, chunks → {chunks_path}")

    def load(self, index_path: str = INDEX_PATH, chunks_path: str = CHUNKS_PATH):
        self.index = faiss.read_index(index_path)
        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)
        logger.info(f"Loaded {self.index.ntotal} vectors, {len(self.chunks)} chunks")

    def exists(self) -> bool:
        return os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH)

    # ── Raw vector search ──
    def search(self, query_vec: np.ndarray, top_k: int) -> tuple[list[float], list[int]]:
        """Returns (scores, indices) — scores are cosine similarities [0,1]."""
        scores_arr, idx_arr = self.index.search(query_vec, top_k)
        return scores_arr[0].tolist(), idx_arr[0].tolist()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 ── Keyword Scoring (BM25-lite)
# ══════════════════════════════════════════════════════════════════════════════

def _tokenize(text: str) -> list[str]:
    """Simple tokeniser: lowercase alphanum tokens."""
    return re.findall(r'[a-z0-9]+', text.lower())


def _idf(term: str, chunks: list[dict]) -> float:
    """Inverse document frequency of a term across all chunks."""
    N  = len(chunks)
    df = sum(1 for c in chunks if term in _tokenize(c["text"]))
    return math.log((N + 1) / (df + 1)) + 1  # smoothed IDF


def keyword_score(query: str, chunk_text: str, chunks: list[dict]) -> float:
    """
    Lightweight TF-IDF overlap score between query and a single chunk.
    Returns a value in [0, 1] (normalised by query term count).
    """
    q_tokens  = set(_tokenize(query))
    c_tokens  = _tokenize(chunk_text)
    c_counter = Counter(c_tokens)

    if not q_tokens:
        return 0.0

    score = 0.0
    for term in q_tokens:
        tf  = c_counter.get(term, 0) / (len(c_tokens) + 1)
        idf = _idf(term, chunks)
        score += tf * idf

    # Normalise to [0, 1] range (approximate)
    return min(score / (len(q_tokens) + 1), 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 ── Query Expansion (fix for failure cases)
# ══════════════════════════════════════════════════════════════════════════════

# Lightweight expansion dictionary — avoids external API calls
EXPANSION_MAP = {
    "inflation":   ["price level", "CPI", "cost of living", "monetary policy"],
    "budget":      ["expenditure", "fiscal", "revenue", "government spending"],
    "election":    ["vote", "ballot", "presidential", "parliamentary", "results"],
    "winner":      ["won", "highest votes", "elected", "victory", "majority"],
    "gdp":         ["gross domestic product", "economic growth", "output"],
    "tax":         ["revenue", "levy", "tariff", "VAT", "income tax"],
    "unemployment":["jobs", "labour market", "employment rate"],
    "region":      ["constituency", "district", "area", "zone"],
    "ndc":         ["National Democratic Congress", "opposition"],
    "npp":         ["New Patriotic Party", "ruling party", "incumbent"],
}

def expand_query(query: str) -> str:
    """
    Expand a short/ambiguous query with domain synonyms.
    This is the FIX for the failure case where vague queries return
    low-similarity chunks.

    Example:
        "Who won?" → "Who won? won highest votes elected victory majority
                       presidential election ballot results"
    """
    tokens   = _tokenize(query)
    additions = []
    for token in tokens:
        if token in EXPANSION_MAP:
            additions.extend(EXPANSION_MAP[token])

    if additions:
        expanded = query + " " + " ".join(additions)
        logger.info(f"Query expanded: '{query}' → '{expanded[:80]}…'")
        return expanded

    return query


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 ── Hybrid Retriever (main interface)
# ══════════════════════════════════════════════════════════════════════════════

class HybridRetriever:
    """
    Full retrieval pipeline:
      1. (Optionally) expand query
      2. Embed query
      3. FAISS vector search → top-K candidates (2× to allow re-ranking)
      4. Compute keyword score for each candidate
      5. Blend: final_score = α × vector + (1-α) × keyword
      6. Re-rank by final score, return top-K
    """

    def __init__(self, vector_store: VectorStore, embedder: EmbeddingPipeline,
                 alpha: float = HYBRID_ALPHA):
        self.store    = vector_store
        self.embedder = embedder
        self.alpha    = alpha

    def retrieve(self, query: str, top_k: int = 5,
                 use_expansion: bool = True) -> list[dict]:
        """
        Main retrieval method.

        Returns list of dicts:
          {text, source, vector_score, keyword_score, final_score,
           is_failure, chunk_id, strategy}
        """
        logs = []
        logs.append(f"[RETRIEVER] Query: {query}")

        if self.store.index.ntotal < 1:
            msg = "Vector index has no documents. Re-initialize the pipeline in the app sidebar."
            logs.append(f"[RETRIEVER] {msg}")
            logger.warning(msg)
            return []

        # ── Step 1: Query expansion ──
        expanded = expand_query(query) if use_expansion else query
        if expanded != query:
            logs.append(f"[RETRIEVER] Expanded: {expanded[:80]}")

        # ── Step 2: Embed ──
        q_vec = self.embedder.embed_one(expanded)

        # ── Step 3: FAISS search (fetch 2× top_k for re-ranking pool) ──
        fetch_k = min(max(1, top_k * 2), self.store.index.ntotal)
        v_scores, indices = self.store.search(q_vec, fetch_k)
        logs.append(f"[RETRIEVER] FAISS returned {len(indices)} candidates")

        # ── Step 4 & 5: Hybrid scoring ──
        results = []
        for rank, (idx, v_score) in enumerate(zip(indices, v_scores)):
            if idx < 0 or idx >= len(self.store.chunks):
                continue
            chunk     = self.store.chunks[idx]
            k_score   = keyword_score(query, chunk["text"], self.store.chunks)
            final     = self.alpha * v_score + (1 - self.alpha) * k_score
            is_fail   = final < LOW_SCORE_THRESH

            results.append({
                "chunk_id":      chunk.get("id", f"chunk_{idx}"),
                "text":          chunk["text"],
                "source":        chunk.get("source", "Unknown"),
                "strategy":      chunk.get("strategy", "unknown"),
                "vector_score":  round(float(v_score),  4),
                "keyword_score": round(float(k_score),  4),
                "final_score":   round(float(final),    4),
                "is_failure":    is_fail,
                "faiss_rank":    rank,
            })
            logs.append(
                f"[RETRIEVER] Chunk {chunk.get('id','?')} | "
                f"vec={v_score:.3f} kw={k_score:.3f} final={final:.3f}"
                f"{' ⚠️ LOW' if is_fail else ''}"
            )

        # ── Step 6: Re-rank and return top_k ──
        results.sort(key=lambda x: x["final_score"], reverse=True)
        top_results = results[:top_k]

        # ── Failure case detection ──
        if top_results and top_results[0]["final_score"] < LOW_SCORE_THRESH:
            logs.append(
                f"[RETRIEVER] ⚠️ FAILURE CASE: Best score {top_results[0]['final_score']:.3f} "
                f"below threshold {LOW_SCORE_THRESH}. Query may be too vague."
            )

        for entry in top_results:
            entry["logs"] = logs

        return top_results


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 ── Build helpers (used by pipeline.py on first run)
# ══════════════════════════════════════════════════════════════════════════════

def build_retriever(chunks: list[dict]) -> tuple[HybridRetriever, list[str]]:
    """
    One-shot builder: embed all chunks → FAISS index → HybridRetriever.
    Saves to disk so subsequent runs skip re-embedding.
    """
    embedder = EmbeddingPipeline()
    store    = VectorStore(dim=embedder.dim)
    store.add_chunks(chunks, embedder)
    store.save()

    retriever = HybridRetriever(store, embedder)
    logs = [
        f"[BUILD] Embedded {len(chunks)} chunks",
        f"[BUILD] FAISS index size: {store.index.ntotal}",
        f"[BUILD] Model: {MODEL_NAME}",
    ]
    return retriever, logs


def load_retriever() -> HybridRetriever:
    """Load a previously saved retriever from disk."""
    embedder = EmbeddingPipeline()
    store    = VectorStore(dim=embedder.dim)
    store.load()
    return HybridRetriever(store, embedder)


if __name__ == "__main__":
    # Quick self-test
    logging.basicConfig(level=logging.INFO)
    dummy_chunks = [
        {"id": "c1", "text": "Ghana's 2025 inflation target is 11.9 percent.", "source": "Budget PDF"},
        {"id": "c2", "text": "NDC won the Volta Region with 76% of votes.",    "source": "Elections CSV"},
        {"id": "c3", "text": "The fiscal deficit target is 3.1% of GDP.",       "source": "Budget PDF"},
    ]
    retriever, _ = build_retriever(dummy_chunks)
    results      = retriever.retrieve("What is the inflation target?", top_k=2)
    for r in results:
        print(f"Score: {r['final_score']:.3f} | {r['text'][:60]}")
