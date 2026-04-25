"""
pipeline.py
───────────
Part D: Full RAG Pipeline
User Query → Retrieval → Context Selection → Prompt → LLM → Response

- Logging at every stage
- Displays retrieved docs, similarity scores, final prompt
- Integrates feedback memory (Part G innovation)
"""

import os
import json
import logging
import requests
from datetime import datetime
from typing import Optional

from data_loader    import prepare_all_chunks
from retriever      import build_retriever, load_retriever, VectorStore
from prompt_builder import build_prompt

logger = logging.getLogger(__name__)
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/pipeline.log", mode="a"),
    ]
)

# ── Config (paths relative to this file, not CWD) ────────────────────────────
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_BASE_DIR, "data")

def _data_csv_path() -> str:
    return os.path.join(_DATA_DIR, "Ghana_Election_Result.csv")

def _data_pdf_path() -> Optional[str]:
    """Use the first .pdf in data/, if any (bundled name may differ)."""
    if not os.path.isdir(_DATA_DIR):
        return None
    pdfs = sorted(
        f for f in os.listdir(_DATA_DIR) if f.lower().endswith(".pdf")
    )
    if not pdfs:
        return None
    return os.path.join(_DATA_DIR, pdfs[0])

FEEDBACK_PATH = "logs/feedback.json"
LOG_PATH      = "logs/query_log.jsonl"

# ── LLM Config ──────────────────────────────────────────────────────────────────
# Uses Groq API (free tier) — OpenAI-compatible endpoint, no SDK needed
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL        = "llama-3.1-8b-instant"   # Groq; replace if deprecations list changes
MAX_TOKENS   = 1024


# ══════════════════════════════════════════════════════════════════════════════
class RAGPipeline:
    """
    Orchestrates the complete RAG flow with stage-level logging.

    Stages:
      INIT  → load data, chunk, embed, build FAISS index
      QUERY → embed query, retrieve, build prompt, call LLM, return result
    """

    def __init__(self):
        self.retriever  = None
        self.ready      = False
        self.query_log  = []
        os.makedirs("logs", exist_ok=True)
        os.makedirs("data", exist_ok=True)

    # ══════════════════════════════════════════════════════════════════════════
    # INIT
    # ══════════════════════════════════════════════════════════════════════════

    def initialize(self, force_rebuild: bool = False):
        """
        Build or load the FAISS index.
        On first run (or if force_rebuild=True), loads data, chunks, embeds.
        Subsequent runs load from disk in seconds.
        """
        logger.info("═" * 60)
        logger.info("[INIT] Starting RAG Pipeline initialization")

        # Check if saved index exists
        from retriever import INDEX_PATH, CHUNKS_PATH
        index_exists = (
            os.path.exists(INDEX_PATH) and
            os.path.exists(CHUNKS_PATH) and
            not force_rebuild
        )

        self.retriever = None
        if index_exists and not force_rebuild:
            logger.info("[INIT] Loading existing FAISS index from disk")
            self.retriever = load_retriever()
            nvec = self.retriever.store.index.ntotal
            if nvec < 1:
                logger.warning(
                    "[INIT] Saved FAISS index is empty (0 vectors). Rebuilding from data folder…"
                )
                self.retriever = None

        if self.retriever is None:
            logger.info("[INIT] Building new index from source documents")

            # Step 1: Load + chunk data
            logger.info("[INIT] Step 1/3 — Loading and chunking documents")
            csv_path = _data_csv_path()
            pdf_path = _data_pdf_path()
            if not os.path.exists(csv_path):
                raise FileNotFoundError(
                    f"Required CSV not found: {csv_path}\n"
                    "Place Ghana_Election_Result.csv in the data/ folder next to the app."
                )
            # PDF is optional: pass a path that may not exist — loader skips if missing
            pdf_for_loader = (
                pdf_path
                if (pdf_path and os.path.exists(pdf_path))
                else os.path.join(_DATA_DIR, "_optional_budget.pdf")
            )
            chunks = prepare_all_chunks(
                csv_path, pdf_for_loader, chunking_strategy="sentence"
            )
            logger.info(f"[INIT] Total chunks: {len(chunks)}")
            if not chunks:
                raise RuntimeError(
                    "No document chunks were loaded. Check that "
                    f"{_DATA_DIR} contains Ghana_Election_Result.csv (and optional PDF), "
                    "then re-initialize."
                )

            # Step 2: Embed + index
            logger.info("[INIT] Step 2/3 — Embedding chunks and building FAISS index")
            self.retriever, build_logs = build_retriever(chunks)
            for log in build_logs:
                logger.info(log)

            # Step 3: Integrate feedback memory
            logger.info("[INIT] Step 3/3 — Loading feedback memory")
            self._load_feedback_memory()

        self.ready = True
        logger.info("[INIT] Pipeline ready ✓")
        logger.info("═" * 60)

    # ══════════════════════════════════════════════════════════════════════════
    # QUERY
    # ══════════════════════════════════════════════════════════════════════════

    def query(self, query: str, top_k: int = 5,
              prompt_style: str = "Hallucination-Controlled") -> dict:
        """
        Full pipeline: query → retrieve → prompt → LLM → response.

        Returns dict with all intermediate data for display.
        """
        if not self.ready:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        all_logs = []
        ts = datetime.now().isoformat()

        logger.info("─" * 60)
        logger.info(f"[QUERY] '{query}' | top_k={top_k} | template={prompt_style}")
        all_logs.append(f"[QUERY] Received: {query}")

        # ── Stage 1: Retrieval ──────────────────────────────────────────────
        logger.info("[STAGE 1] Hybrid retrieval")
        all_logs.append("[STAGE 1] Retrieving relevant chunks")

        retrieved = self.retriever.retrieve(query, top_k=top_k, use_expansion=True)

        for r in retrieved:
            all_logs.extend(r.get("logs", []))
            logger.info(
                f"  → {r['source'][:40]} | "
                f"vec={r['vector_score']:.3f} kw={r['keyword_score']:.3f} "
                f"final={r['final_score']:.3f}"
                f"{' ⚠️' if r['is_failure'] else ''}"
            )

        # ── Stage 2: Context window management ─────────────────────────────
        logger.info("[STAGE 2] Building prompt")
        all_logs.append(f"[STAGE 2] Building prompt with template: {prompt_style}")

        prompt, kept_chunks = build_prompt(query, retrieved, prompt_style)
        all_logs.append(f"[STAGE 2] Context: {len(kept_chunks)} chunks, {len(prompt)} chars")
        logger.info(f"[STAGE 2] Prompt: {len(kept_chunks)} chunks, {len(prompt)} chars")

        # ── Stage 3: LLM Generation ─────────────────────────────────────────
        logger.info("[STAGE 3] Calling LLM")
        all_logs.append(f"[STAGE 3] Sending prompt to Groq API ({MODEL})")

        answer = self._call_llm(prompt)
        all_logs.append(f"[STAGE 3] Response received ({len(answer)} chars)")
        logger.info(f"[STAGE 3] Response: {len(answer)} chars")

        # ── Stage 4: Package result ─────────────────────────────────────────
        result = {
            "query":    query,
            "answer":   answer,
            "chunks":   [c["text"]         for c in kept_chunks],
            "scores":   [c["final_score"]  for c in kept_chunks],
            "sources":  [c["source"]       for c in kept_chunks],
            "prompt":   prompt,
            "logs":     all_logs,
            "retrieved_full": retrieved,   # full metadata for experiment log
            "timestamp": ts,
        }

        # ── Stage 5: Log to disk ────────────────────────────────────────────
        self._log_query(result)

        return result

    # ══════════════════════════════════════════════════════════════════════════
    # LLM CALL
    # ══════════════════════════════════════════════════════════════════════════

    def _call_llm(self, prompt: str) -> str:
        """
        Direct HTTP call to Groq API (OpenAI-compatible format).
        No SDK — raw requests only (satisfies the 'no pre-built pipeline' constraint).
        Groq is free and very fast (tokens served at ~500 tokens/sec).
        """
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            return (
                "⚠️ GROQ_API_KEY not set. "
                "In CMD run: set GROQ_API_KEY=gsk_yourkey"
            )

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
        }
        payload = {
            "model":      MODEL,
            "max_tokens": MAX_TOKENS,
            "messages":   [{"role": "user", "content": prompt}],
            "temperature": 0.3,   # lower = more factual, less creative (good for RAG)
        }

        try:
            resp = requests.post(
                GROQ_API_URL,
                headers=headers,
                json=payload,
                timeout=30
            )
            resp.raise_for_status()
            answer = resp.json()["choices"][0]["message"]["content"].strip()
            return answer

        except requests.exceptions.Timeout:
            return "⚠️ Groq API timed out. Please try again."
        except requests.exceptions.HTTPError as e:
            return f"⚠️ Groq API error: {e.response.status_code} — {e.response.text[:200]}"
        except Exception as e:
            return f"⚠️ Unexpected error: {str(e)}"

    # ══════════════════════════════════════════════════════════════════════════
    # LOGGING & FEEDBACK MEMORY (Part G)
    # ══════════════════════════════════════════════════════════════════════════

    def _log_query(self, result: dict):
        """Append query + result metadata to JSONL log for experiment analysis."""
        entry = {
            "timestamp": result["timestamp"],
            "query":     result["query"],
            "answer_preview": result["answer"][:120],
            "chunks_used":    len(result["chunks"]),
            "top_score":      result["scores"][0] if result["scores"] else 0,
            "sources":        list(set(result["sources"])),
        }
        try:
            with open(LOG_PATH, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass

    def _load_feedback_memory(self):
        """
        Part G — Innovation: Feedback loop memory.
        Reads past 👎 feedback and logs warnings about known bad queries.
        In a production system, this would retrain or re-weight the retriever.
        """
        if not os.path.exists(FEEDBACK_PATH):
            return

        try:
            with open(FEEDBACK_PATH) as f:
                feedback = json.load(f)

            bad = [fb for fb in feedback if fb.get("rating") == "👎"]
            if bad:
                logger.info(
                    f"[MEMORY] Loaded {len(bad)} negative feedback entries. "
                    f"These queries had poor results previously."
                )
                for fb in bad[-5:]:   # log last 5
                    logger.info(f"  ⚠️  Bad query: {fb.get('answer_preview', '')[:60]}")
        except Exception:
            pass

    # ══════════════════════════════════════════════════════════════════════════
    # ADVERSARIAL TEST HELPERS (Part E)
    # ══════════════════════════════════════════════════════════════════════════

    def run_adversarial_tests(self) -> list[dict]:
        """
        Part E: Runs the 2 required adversarial queries and logs results.
        Also runs the same queries against LLM-only (no retrieval) for comparison.
        """
        adversarial_queries = [
            {
                "query":       "Who won?",
                "type":        "Ambiguous",
                "description": "No context — who won what? What year? What region?"
            },
            {
                "query":       "What was Ghana's debt in 2019 according to the 2025 budget?",
                "type":        "Misleading",
                "description": "Asks for 2019 data from a 2025 document — tests hallucination."
            },
        ]

        results = []
        for test in adversarial_queries:
            logger.info(f"[ADVERSARIAL] Running: {test['query']}")

            # RAG response
            rag_result = self.query(test["query"], top_k=5)

            # LLM-only response (no retrieval context)
            llm_only_prompt = f"Question: {test['query']}\nAnswer:"
            llm_only_answer = self._call_llm(llm_only_prompt)

            results.append({
                **test,
                "rag_answer":      rag_result["answer"],
                "rag_top_score":   rag_result["scores"][0] if rag_result["scores"] else 0,
                "llm_only_answer": llm_only_answer,
                "rag_chunks_used": len(rag_result["chunks"]),
            })

            logger.info(f"  RAG:      {rag_result['answer'][:80]}…")
            logger.info(f"  LLM-only: {llm_only_answer[:80]}…")

        # Save to file for evidence-based comparison
        with open("logs/adversarial_results.json", "w") as f:
            json.dump(results, f, indent=2)

        logger.info("[ADVERSARIAL] Results saved to logs/adversarial_results.json")
        return results


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    pipeline = RAGPipeline()
    pipeline.initialize()

    result = pipeline.query(
        "What is Ghana's inflation target for 2025?",
        top_k=5,
        prompt_style="Hallucination-Controlled"
    )
    print("\n" + "="*60)
    print("ANSWER:", result["answer"])
    print("\nTOP CHUNKS:")
    for i, (chunk, score) in enumerate(zip(result["chunks"], result["scores"])):
        print(f"  [{i+1}] Score={score:.3f} | {chunk[:80]}…")
