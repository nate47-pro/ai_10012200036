# ACity RAG Assistant

# Nathaniel Adam
# 10012200036git remote add origin https://github.com/<your-username>/<repo-name>.git


**Retrieval-Augmented Generation Chatbot — Academic City University Ghana**

> Built from scratch — no LangChain, no LlamaIndex, no pre-built RAG pipelines.  
> **LLM:** Groq API (`llama-3.1-8b-instant`) · **Embeddings:** sentence-transformers · **Vector store:** FAISS

---

## Project structure

```
project-root/
├── app.py                  # Streamlit UI (chat, sidebar “Control deck”, theme)
├── pipeline.py             # Full RAG orchestrator (Part D) + Groq HTTP calls
├── data_loader.py          # Data cleaning + chunking (Part A)
├── retriever.py            # Embedding + FAISS + hybrid search (Part B)
├── prompt_builder.py       # Prompt templates + context (Part C)
├── requirements.txt
├── data/
│   ├── Ghana_Election_Result.csv   # elections corpus (expected filename)
│   ├── *.pdf                       # first PDF in folder used as budget source
│   ├── faiss.index                 # auto-generated on init
│   └── chunks.pkl                  # auto-generated on init
├── logs/
│   ├── experiment_log.txt
│   ├── pipeline.log
│   ├── query_log.jsonl
│   ├── feedback.json
│   └── adversarial_results.json
└── README.md
```

Paths to CSV/PDF are resolved **relative to `pipeline.py`** (the `data/` folder next to the code), so the app works regardless of the shell’s current working directory.

---

## Quick start

### 1. Install dependencies

```bash
cd /path/to/project
pip install -r requirements.txt
```

**Streamlit:** `requirements.txt` pins `streamlit>=1.32.0`. A recent version is recommended for **scrollable chat** (`st.container(height=…)`) and **Enter-to-send** in `st.form`.

### 2. Data files

Place in `data/`:

| File | Role |
|------|------|
| `Ghana_Election_Result.csv` | Election results (expected name) |
| Any `*.pdf` | Budget / policy PDF — the pipeline picks the **first** PDF in `data/` alphabetically |

### 3. Groq API key

Create a key at [Groq Console](https://console.groq.com/).

**Windows (CMD)**

```cmd
set GROQ_API_KEY=gsk_your_key_here
```

**Windows (PowerShell)**

```powershell
$env:GROQ_API_KEY=gsk_****************
```

### 4. Run the app

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

### 5. Initialize the pipeline (required before chat)

1. Open the **Control deck** (left sidebar). If it is collapsed, use Streamlit’s **sidebar toggle** at the top-left (`>>` / chevron).
2. Click **Initialize pipeline** (first run embeds chunks and builds FAISS — can take a minute or two; later runs load from disk when the index exists).

Until the pipeline is ready, a **reminder banner** appears above the chat box. Sending a message without init still shows a warning.

### 6. Using the chat UI

- **Try asking…** suggestion buttons **fill** the input field; press **Send** or **Enter** (form submit) to run the query.
- With history, the **message list** scrolls inside a fixed-height region so older turns stay reachable.
- Sidebar: **Top-K**, **prompt template**, toggles for chunks / scores / full prompt, **session stats**, **Clear chat**, and corpus summary.

---

## UI overview (Streamlit)

| Area | Behaviour |
|------|-----------|
| **Theme** | Red / white / cream palette; sidebar “Control deck” styling |
| **Top chrome** | Streamlit header styled for readable toolbar (e.g. Deploy) on a light bar |
| **Chat** | Custom HTML bubbles; optional expanders for chunks and prompt |
| **Input** | `st.form` with text field + **Send**; **Enter** submits the form |
| **Init hint** | Banner when `pipeline_ready` is false |
| **Scrolling** | `st.container(height=…, key=…)` + CSS for tall thread |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                        │
│                    Streamlit (app.py)                        │
└─────────────────────────┬────────────────────────────────────┘
                          │ query
                          ▼
┌──────────────────────────────────────────────────────────────┐
│                     RAG PIPELINE                             │
│                     pipeline.py                              │
│                                                              │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────────┐  │
│  │ data_loader │    │  retriever   │    │ prompt_builder │  │
│  │             │    │              │    │                │  │
│  │ • load CSV  │───▶│ • embed query│───▶│ • select tpl   │  │
│  │ • load PDF  │    │ • FAISS search│   │ • manage ctx   │  │
│  │ • clean     │    │ • keyword score│  │ • fill template│  │
│  │ • chunk     │    │ • hybrid rank│    │                │  │
│  └─────────────┘    └──────────────┘    └────────┬───────┘  │
│         │                  ▲                     │          │
│         │ chunks           │ index               │ prompt   │
│         ▼                  │                     ▼          │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────────┐  │
│  │  Sentence   │    │    FAISS     │    │   Groq API     │  │
│  │ Transformer │───▶│  Vector DB   │    │ (OpenAI-compat)│  │
│  │ (embedder)  │    │  (index)     │    │                │  │
│  └─────────────┘    └──────────────┘    └────────┬───────┘  │
│                                                  │ answer   │
└──────────────────────────────────────────────────┼──────────┘
                                                   │
                          ┌────────────────────────▼───┐
                          │    Response + Metadata     │
                          │  • answer text             │
                          │  • retrieved chunks        │
                          │  • similarity scores       │
                          │  • full prompt             │
                          │  • stage logs              │
                          └────────────────────────────┘
```

---

## Design decisions (summary)

### Part A — Data engineering

- **CSV:** Row-grouped chunks, natural-language lines for retrieval.  
- **PDF:** Sentence-aware windows with overlap for policy text.

### Part B — Retrieval

- **Embeddings:** `all-MiniLM-L6-v2` (sentence-transformers).  
- **FAISS:** `IndexFlatIP` with L2-normalised vectors (cosine via dot product).  
- **Hybrid:** `final_score = 0.7 × vector_score + 0.3 × keyword_score`.

### Part C — Prompting

Templates include **Hallucination-Controlled**, **Chain-of-Thought**, and **Basic** (see `prompt_builder.py`). Prompts reference **Academic City University, Ghana** where institution context is needed.

### Part G — Feedback

👍 / 👎 on answers → `logs/feedback.json`; surfaced in sidebar session stats.

---

## Adversarial tests (Part E)

### Test 1: Ambiguous query

**Query:** “Who won?”

- **RAG:** Returns low-confidence chunks (score ~0.31); answer states it needs more context to determine “who won what.”
- **LLM-only:** May confidently assert a winner (possible hallucination).
- **Finding:** RAG can express uncertainty; a bare LLM risks confabulation.

### Test 2: Misleading / temporal query

**Query:** “What was Ghana’s debt in 2019 according to the 2025 budget?”

- **RAG:** Retrieves 2025 budget chunks and correctly indicates 2019 figures are not in the supplied context.
- **LLM-only:** May invent a 2019 debt number from parametric knowledge.
- **Finding:** Grounding in retrieved text reduces temporal hallucination.

---

## Failure cases and mitigations

**Vague queries** (e.g. “Who won?”) can return weak chunks (similarity scores around 0.31–0.35). The pipeline can flag low-confidence behaviour; **rephrase** or use **query expansion** (domain synonyms) where implemented.

**Empty or stale FAISS index:** Ensure `data/` is correct, then run **Initialize pipeline** again. Retriever logic avoids invalid `k=0` states when the index is empty.

---

## Troubleshooting

| Problem | What to try |
|---------|-------------|
| `GROQ_API_KEY not set` | Set env var (see §3); restart terminal and Streamlit |
| Groq **model deprecated** | In `pipeline.py`, set `MODEL` to a current model from [Groq docs](https://console.groq.com/docs/models) |
| CSV / PDF not found | Ensure `data/Ghana_Election_Result.csv` and at least one `data/*.pdf` exist |
| First init slow | Expected while embedding and building FAISS |
| Sidebar not visible | Use the **top-left** Streamlit sidebar control to expand |
| Suggestions don’t fill input | Use a current Streamlit; fill runs via `session_state` before the text widget |

---


---
*Stack: Python, sentence-transformers, FAISS, Streamlit, Groq (OpenAI-compatible HTTP API via `requests`).*
