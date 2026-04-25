"""
data_loader.py
──────────────
Part A: Data Engineering & Preparation
- Loads Ghana Election CSV + 2025 Budget PDF
- Cleans data
- Implements two chunking strategies with justification
"""

import re
import os
import logging
import pandas as pd
import pdfplumber

logger = logging.getLogger(__name__)

# ── Chunking constants (justified below) ──────────────────────────────────────
# PDF chunks: 500 chars, 100 overlap
#   Justification: The budget PDF is dense policy prose. 500 chars (~80-100 words)
#   captures a complete policy idea. 100-char overlap prevents context loss at
#   boundaries (e.g., a sentence that spans two chunks keeps its subject visible).
#
# CSV chunks: one row per chunk, grouped into windows of 5 rows
#   Justification: Each CSV row is an atomic election record. Grouping 5 rows
#   allows the retriever to find regional patterns (e.g., all results in Ashanti)
#   without bloating the context window.

PDF_CHUNK_SIZE    = 500
PDF_CHUNK_OVERLAP = 100
CSV_GROUP_SIZE    = 5   # rows per chunk


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 ── Loaders
# ══════════════════════════════════════════════════════════════════════════════

def load_csv(path: str) -> pd.DataFrame:
    """Load and clean the Ghana Election Results CSV."""
    logger.info(f"Loading CSV: {path}")
    df = pd.read_csv(path)

    original_rows = len(df)

    # ── Cleaning steps ──
    # 1. Strip leading/trailing whitespace from string columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()

    # 2. Replace placeholder strings with NaN
    df.replace(["nan", "None", "N/A", "n/a", "-", ""], pd.NA, inplace=True)

    # 3. Drop fully empty rows
    df.dropna(how="all", inplace=True)

    # 4. Normalize column names: lowercase, replace spaces with underscores
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    # 5. Remove duplicate rows
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    logger.info(
        f"CSV cleaned: {original_rows} → {len(df)} rows "
        f"({original_rows - len(df)} removed)"
    )
    return df


def load_pdf(path: str) -> list[dict]:
    """Load the 2025 Budget PDF and extract raw text per page."""
    logger.info(f"Loading PDF: {path}")
    pages = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text and text.strip():
                # Basic cleaning: collapse multiple newlines, strip headers/footers
                text = re.sub(r'\n{3,}', '\n\n', text)
                text = re.sub(r'[ \t]{2,}', ' ', text)
                text = text.strip()
                pages.append({"page": i + 1, "text": text})

    logger.info(f"PDF loaded: {len(pages)} pages with content")
    return pages


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 ── Chunking Strategies
# ══════════════════════════════════════════════════════════════════════════════

def chunk_text_fixed(text: str, source: str, chunk_size: int = PDF_CHUNK_SIZE,
                     overlap: int = PDF_CHUNK_OVERLAP) -> list[dict]:
    """
    Strategy 1 — Fixed-size character chunking with overlap.
    Best for: dense prose (budget PDF).

    Design decision:
      500 chars ≈ 80-100 words, enough for a complete policy statement.
      100-char overlap ensures sentences split across boundaries are
      still retrievable from either chunk.
    """
    chunks = []
    start  = 0
    idx    = 0
    text   = text.strip()

    while start < len(text):
        end   = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append({
                "id":       f"{source}_fixed_{idx}",
                "text":     chunk,
                "source":   source,
                "strategy": "fixed",
                "chunk_idx": idx,
                "char_start": start,
                "char_end":   end,
            })
            idx += 1

        start += chunk_size - overlap  # slide window with overlap

    return chunks


def chunk_text_sentence(text: str, source: str,
                        words_per_chunk: int = 120,
                        overlap_sentences: int = 2) -> list[dict]:
    """
    Strategy 2 — Sentence-aware chunking.
    Best for: text where sentence boundaries matter (e.g., financial projections).

    Design decision:
      Split on sentence endings first, then group into ~120-word windows.
      Overlap by 2 sentences so questions about transitions between ideas
      are answered correctly.
    """
    # Split into sentences using punctuation heuristic
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences  = [s.strip() for s in sentences if s.strip()]

    chunks  = []
    current = []
    word_ct = 0
    idx     = 0

    for i, sent in enumerate(sentences):
        words    = sent.split()
        word_ct += len(words)
        current.append(sent)

        if word_ct >= words_per_chunk:
            chunk_text = " ".join(current)
            chunks.append({
                "id":        f"{source}_sent_{idx}",
                "text":      chunk_text,
                "source":    source,
                "strategy":  "sentence",
                "chunk_idx": idx,
            })
            idx += 1
            # Keep last N sentences as overlap
            current  = current[-overlap_sentences:] if overlap_sentences > 0 else []
            word_ct  = sum(len(s.split()) for s in current)

    # Flush remaining sentences
    if current:
        chunks.append({
            "id":        f"{source}_sent_{idx}",
            "text":      " ".join(current),
            "source":    source,
            "strategy":  "sentence",
            "chunk_idx": idx,
        })

    return chunks


def csv_to_chunks(df: pd.DataFrame, group_size: int = CSV_GROUP_SIZE) -> list[dict]:
    """
    Strategy 3 — Row-grouping for structured CSV data.
    Each chunk = group_size consecutive rows serialised as readable text.

    Design decision:
      Raw CSV rows are not prose — we convert them to natural language snippets.
      Grouping 5 rows keeps related regional results together, making
      "Who won in [region]?" queries return a useful chunk.
    """
    chunks = []
    for start in range(0, len(df), group_size):
        group = df.iloc[start:start + group_size]
        lines = []
        for _, row in group.iterrows():
            # Convert row to a human-readable sentence
            parts = [f"{col.replace('_', ' ').title()}: {val}"
                     for col, val in row.items()
                     if pd.notna(val) and str(val).strip()]
            lines.append(" | ".join(parts))

        chunk_text = "\n".join(lines)
        if chunk_text.strip():
            chunks.append({
                "id":        f"csv_chunk_{start // group_size}",
                "text":      chunk_text,
                "source":    "Ghana Elections CSV",
                "strategy":  "row_group",
                "chunk_idx": start // group_size,
                "row_start": start,
                "row_end":   start + len(group) - 1,
            })

    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 ── Master preparation function
# ══════════════════════════════════════════════════════════════════════════════

def prepare_all_chunks(csv_path: str, pdf_path: str,
                       chunking_strategy: str = "sentence") -> list[dict]:
    """
    Orchestrates loading + chunking of both data sources.

    Args:
        csv_path:          path to Ghana_Election_Result.csv
        pdf_path:          path to 2025-Budget-Statement.pdf
        chunking_strategy: 'fixed' | 'sentence' (applies to PDF)

    Returns:
        List of chunk dicts ready for embedding.
    """
    all_chunks = []

    # ── 1. CSV ──
    if os.path.exists(csv_path):
        df = load_csv(csv_path)
        csv_chunks = csv_to_chunks(df)
        all_chunks.extend(csv_chunks)
        logger.info(f"CSV → {len(csv_chunks)} chunks")
    else:
        logger.warning(f"CSV not found at {csv_path}")

    # ── 2. PDF ──
    if os.path.exists(pdf_path):
        pages = load_pdf(pdf_path)
        pdf_chunks = []

        for page in pages:
            src  = f"2025 Budget PDF (p.{page['page']})"
            text = page["text"]

            if chunking_strategy == "fixed":
                pdf_chunks.extend(chunk_text_fixed(text, src))
            else:
                pdf_chunks.extend(chunk_text_sentence(text, src))

        all_chunks.extend(pdf_chunks)
        logger.info(f"PDF → {len(pdf_chunks)} chunks ({chunking_strategy} strategy)")
    else:
        logger.warning(f"PDF not found at {pdf_path}")

    logger.info(f"Total chunks prepared: {len(all_chunks)}")
    return all_chunks


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 ── Chunking comparison (for experiment log)
# ══════════════════════════════════════════════════════════════════════════════

def compare_chunking_strategies(text: str, source: str = "test") -> dict:
    """
    Utility: runs both strategies on a sample text and returns stats.
    Use this to populate your manual experiment log.
    """
    fixed    = chunk_text_fixed(text, source)
    sentence = chunk_text_sentence(text, source)

    def stats(chunks):
        lengths = [len(c["text"]) for c in chunks]
        return {
            "count":       len(chunks),
            "avg_chars":   sum(lengths) / len(lengths) if lengths else 0,
            "min_chars":   min(lengths) if lengths else 0,
            "max_chars":   max(lengths) if lengths else 0,
        }

    return {
        "fixed_size":      stats(fixed),
        "sentence_aware":  stats(sentence),
    }


if __name__ == "__main__":
    # Quick smoke test
    logging.basicConfig(level=logging.INFO)
    sample = (
        "The Government of Ghana targets an end-year inflation rate of 11.9 percent. "
        "This will be achieved through prudent fiscal management. The fiscal deficit "
        "is projected to narrow to 3.1 percent of GDP. Revenue measures include "
        "broadening the tax base and improving compliance. The budget also outlines "
        "key expenditure rationalisation efforts across all MDAs."
    )
    result = compare_chunking_strategies(sample)
    print("Chunking comparison:")
    for strat, s in result.items():
        print(f"  {strat}: {s}")
