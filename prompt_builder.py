"""
prompt_builder.py
─────────────────
Part C: Prompt Engineering & Generation
- Three prompt template iterations
- Context window management (rank, filter, truncate)
- Hallucination control mechanisms
"""

import logging

logger = logging.getLogger(__name__)

# ── Token budget constants ─────────────────────────────────────────────────────
MAX_CONTEXT_CHARS  = 3500   # ~875 tokens at 4 chars/token (leaves room for response)
MIN_CHUNK_SCORE    = 0.25   # discard chunks below this score before injecting
MAX_CHUNKS_IN_CTX  = 6      # hard cap regardless of score


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 ── Context Window Management
# ══════════════════════════════════════════════════════════════════════════════

def manage_context_window(retrieved: list[dict],
                          max_chars: int  = MAX_CONTEXT_CHARS,
                          min_score: float = MIN_CHUNK_SCORE,
                          max_chunks: int  = MAX_CHUNKS_IN_CTX) -> list[dict]:
    """
    Filters and trims retrieved chunks to fit within the context window.

    Steps:
      1. Remove chunks below min_score threshold (likely irrelevant)
      2. Sort descending by final_score
      3. Cap at max_chunks
      4. Trim individual chunk text if total chars exceed max_chars
    """
    # Step 1: filter low-quality chunks
    filtered = [c for c in retrieved if c["final_score"] >= min_score]
    if not filtered:
        # Fallback: keep top-1 even if low score so we always have something
        filtered = retrieved[:1] if retrieved else []

    # Step 2 & 3: sort + cap
    filtered.sort(key=lambda x: x["final_score"], reverse=True)
    filtered = filtered[:max_chunks]

    # Step 4: truncate to stay within char budget
    total_chars = 0
    final = []
    for chunk in filtered:
        text    = chunk["text"]
        allowed = max_chars - total_chars
        if allowed <= 50:
            break
        if len(text) > allowed:
            text = text[:allowed] + "…"
        chunk = {**chunk, "text": text}    # don't mutate original
        final.append(chunk)
        total_chars += len(text)

    logger.info(
        f"Context window: {len(retrieved)} chunks → {len(final)} kept "
        f"({total_chars} chars)"
    )
    return final


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 ── Prompt Templates (3 iterations)
# ══════════════════════════════════════════════════════════════════════════════

def _format_chunks(chunks: list[dict]) -> str:
    """Format chunks as a numbered, source-annotated context block."""
    parts = []
    for i, c in enumerate(chunks, 1):
        src   = c.get("source", "Unknown")
        score = c.get("final_score", 0)
        parts.append(
            f"[{i}] Source: {src} (relevance: {score:.2f})\n{c['text']}"
        )
    return "\n\n".join(parts)


# ── Template v1: Basic (baseline for experiment) ──────────────────────────────
TEMPLATE_BASIC = """\
Context:
{context}

Question: {query}
Answer:"""


# ── Template v2: Hallucination-Controlled (recommended default) ───────────────
TEMPLATE_HALLUCINATION_CONTROLLED = """\
You are a knowledgeable assistant for Academic City University, Ghana.
Your role is to answer questions about Ghana's elections and the 2025 Budget.

STRICT RULES:
1. Answer ONLY using the context provided below.
2. If the answer is not present in the context, respond exactly:
   "I don't have enough information in the provided documents to answer this."
3. Do NOT invent figures, names, or statistics.
4. Cite the source number (e.g., [1], [2]) when referencing specific data.
5. Keep your answer concise and factual.

--- CONTEXT START ---
{context}
--- CONTEXT END ---

Question: {query}

Answer (based strictly on the context above):"""


# ── Template v3: Chain-of-Thought (innovation component) ──────────────────────
TEMPLATE_COT = """\
You are an expert research assistant for Academic City University, Ghana, specialising in
Ghanaian politics and economics.

You have been provided with relevant document excerpts below. Use step-by-step
reasoning to construct your answer.

--- RETRIEVED DOCUMENTS ---
{context}
--- END OF DOCUMENTS ---

Question: {query}

Instructions:
- Step 1: Identify which document(s) are most relevant to the question.
- Step 2: Extract the key facts from those documents.
- Step 3: Synthesise those facts into a clear, direct answer.
- Step 4: If any part of the question cannot be answered from the documents,
          explicitly state that.

Your response (structured):
RELEVANT SOURCES: [list the source numbers you will use]
KEY FACTS:
- [fact 1 from source X]
- [fact 2 from source Y]
ANSWER:
[your synthesised answer]"""


TEMPLATES = {
    "Basic":                    TEMPLATE_BASIC,
    "Hallucination-Controlled": TEMPLATE_HALLUCINATION_CONTROLLED,
    "Chain-of-Thought":         TEMPLATE_COT,
}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 ── Prompt Builder
# ══════════════════════════════════════════════════════════════════════════════

def build_prompt(query: str, retrieved: list[dict],
                 template_name: str = "Hallucination-Controlled") -> tuple[str, list[dict]]:
    """
    Full prompt construction pipeline:
      1. Manage context window (filter + trim)
      2. Format chunks into readable context block
      3. Inject into chosen template

    Returns:
        (prompt_string, kept_chunks)
    """
    # Step 1: manage context window
    kept = manage_context_window(retrieved)

    if not kept:
        # No usable context — prompt LLM to admit it doesn't know
        context = "(No relevant documents were retrieved for this query.)"
    else:
        context = _format_chunks(kept)

    # Step 2: pick template
    template = TEMPLATES.get(template_name, TEMPLATE_HALLUCINATION_CONTROLLED)

    # Step 3: fill template
    prompt = template.format(context=context, query=query)

    logger.info(
        f"Prompt built: template='{template_name}', "
        f"chunks={len(kept)}, prompt_chars={len(prompt)}"
    )
    return prompt, kept


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 ── Experiment comparison utility
# ══════════════════════════════════════════════════════════════════════════════

def compare_prompts(query: str, retrieved: list[dict]) -> dict:
    """
    Generate all three prompts for the same query/context.
    Use this output in your experiment log.
    """
    results = {}
    for name in TEMPLATES:
        prompt, kept = build_prompt(query, retrieved, name)
        results[name] = {
            "prompt":      prompt,
            "chunks_used": len(kept),
            "prompt_len":  len(prompt),
        }
    return results


if __name__ == "__main__":
    # Quick demo
    sample_retrieved = [
        {
            "text":        "Ghana's 2025 inflation target is 11.9 percent of GDP.",
            "source":      "2025 Budget PDF (p.14)",
            "final_score": 0.87,
        },
        {
            "text":        "The fiscal deficit target is 3.1 percent of GDP.",
            "source":      "2025 Budget PDF (p.22)",
            "final_score": 0.72,
        },
    ]

    for name in TEMPLATES:
        prompt, kept = build_prompt(
            "What is Ghana's inflation target for 2025?",
            sample_retrieved,
            name
        )
        print(f"\n{'='*60}")
        print(f"Template: {name}  ({len(prompt)} chars, {len(kept)} chunks)")
        print(prompt[:300], "…")
