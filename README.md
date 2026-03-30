# 🎓 CourseAdvisor AI
### Agentic RAG — Prerequisite & Course Planning Assistant
**Purple Merit Technologies | AI/ML Engineer Intern Assessment 1**

---

## Overview
CourseAdvisor AI is a catalog-grounded academic advising assistant built on a 4-agent RAG pipeline. It answers student questions about course prerequisites, degree requirements, and term planning — strictly using academic catalog documents with verifiable citations. It never guesses or uses outside knowledge.

---

## Features
- ✅ **4-Agent Pipeline** — Intake → Retriever → Planner → Verifier
- ✅ **Grounded answers** with `[Source: filename, Page X]` citations on every claim
- ✅ **Prerequisite reasoning** — ELIGIBLE / NOT ELIGIBLE / NEED MORE INFO
- ✅ **Multi-hop prerequisite chains** — traces A → B → C paths
- ✅ **Safe abstention** — says "I don't have that information" when policy is missing
- ✅ **Hard university filter** — never cites a different university's documents
- ✅ **Clarifying questions** when student profile is incomplete
- ✅ **Streamlit UI** with live agent pipeline progress tracker
- ✅ **Google Colab compatible** with Google Drive integration

---

## Tech Stack
| Component | Choice |
|-----------|--------|
| LLM | Groq `llama-3.1-8b-instant` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store | FAISS (local, saved to Google Drive) |
| Framework | LangChain |
| UI | Streamlit + ngrok |
| Runtime | Google Colab (free tier) |

---

## Project Structure
```
rag_project/
├── CourseAdvisor_FINAL.ipynb   ← Main Colab notebook (run this)
├── eval_set.py                 ← Evaluation script (25 queries)
├── example_transcripts.py      ← 3 sample transcripts
├── writeup.pdf                 ← 1-page submission write-up
├── README.md                   ← This file
└── data/                       ← Put your catalog PDFs here
    ├── fast_cs_catalog.pdf
    ├── nust_cs_handbook.pdf
    └── ...
```

---

## Setup & Run

### Step 1 — Prepare your Google Drive
```
MyDrive/
└── rag_project/
    └── data/        ← upload catalog PDFs here
```
Name your PDFs with the university name so metadata tagging works:
- `fast_cs_catalog.pdf` → tagged as FAST
- `nust_seecs_handbook.pdf` → tagged as NUST
- `pucit_cs_program.pdf` → tagged as PUCIT

### Step 2 — Get free API keys
- **Groq API key**: https://console.groq.com (free)
- **ngrok auth token**: https://dashboard.ngrok.com/get-started/your-authtoken (free)

### Step 3 — Open `CourseAdvisor_FINAL.ipynb` in Google Colab
Run cells in order:
1. **Cell 1** — Mount Google Drive
2. **Cell 2** — Install packages
3. **Cell 3** — Write `app.py` to `/content/`
4. **Cell 4** — Paste ngrok token → get public URL

### Step 4 — Use the app
1. Enter Groq API key in sidebar
2. Click **Load Index** (if index already built) or **Build Index** (first time, takes 5-10 min)
3. Scroll down in sidebar to fill **Student Profile**:
   - University (must match PDF filename keyword e.g. `FAST`)
   - Program (CS / SE / IT / DS)
   - Completed Courses (comma separated)
4. Type your question in the chat box

---

## Example Queries
```
Can I take Data Structures if I completed CS101?
What courses do I need before Database Systems?
Suggest a Fall term plan for a FAST CS student who completed CS101, MATH101
What is the prerequisite chain to reach Machine Learning?
How many total credit hours are required for the CS degree?
Is a Final Year Project mandatory for graduation?
Which professor teaches CS301? [→ correct abstention expected]
Is CS301 offered in Spring 2026? [→ correct abstention expected]
```

---

## Running the Evaluation
```python
# In a new Colab cell, after building the index:
import subprocess
result = subprocess.run(['python', '/content/eval_set.py'], capture_output=True, text=True)
print(result.stdout)
```
Or copy `eval_set.py` to your Drive and run it directly. Results saved to:
```
/content/drive/MyDrive/rag_project/eval_results.json
```

---

## Evaluation Results

| Metric | Score |
|--------|-------|
| Citation coverage rate | 92% (23/25) |
| Eligibility correctness | 80% (8/10 prereq checks) |
| Abstention accuracy | 100% (5/5 not-in-docs) |
| Section completeness | 96% (24/25) |

---

## Agent Responsibilities

### Agent 1 — Intake
- Validates: university, program, completed courses
- Returns clarifying questions if profile is incomplete
- Classifies query type (prereq / plan / policy)

### Agent 2 — Retriever
- Builds context-aware search query
- Retrieves top-7 chunks via FAISS similarity search
- **Hard university filter**: only passes chunks from the student's university
- Returns `NO_SOURCE` abstention chunk if no matching docs found

### Agent 3 — Planner
- Generates structured 5-section response using only retrieved chunks
- Cites every claim with `[Source: filename, Page X]`
- States ELIGIBLE / NOT ELIGIBLE / NEED MORE INFO for prereq questions
- Abstains with exact phrase if information not in catalog

### Agent 4 — Verifier
- Checks citation coverage on every factual claim
- Verifies all 5 sections are present
- Flags hallucinated facts not from retrieved chunks
- Returns VERIFIED or ISSUES

---

## Output Format
Every response follows this exact structure:
```
Answer / Plan:
[answer or plan]

Why (requirements/prereqs satisfied):
[step-by-step reasoning with citations]

Citations:
[Source: filename, Page X — what it says]

Clarifying questions (if needed):
[questions or: None]

Assumptions / Not in catalog:
[what was assumed or not found]
```

---

## Known Limitations & Future Work
1. **Multi-hop chains**: Deep chains (3+ levels) may not always surface all intermediate prereqs — a knowledge graph approach would solve this
2. **Either/OR prerequisites**: Catalog wording "A or B" sometimes interpreted as requiring both
3. **Semester availability**: Course offering schedules are not in catalogs — correctly abstains but cannot advise
4. **Future**: Add BM25 hybrid search for exact course code matching (e.g. CS301)

---

## Data Sources
| University | Source | Date Accessed |
|-----------|--------|---------------|
| FAST-NUCES | nu.edu.pk/academics | March 2026 |
| NUST SEECS | nust.edu.pk/seecs | March 2026 |
| PUCIT | pucit.edu.pk | March 2026 |
| UMT | umt.edu.pk | March 2026 |
| University of Malakand | uom.edu.pk | March 2026 |
| IIT | iit.edu.pk | March 2026 |

---

*Submitted by: Muhammad Ali | L1F22BSCS0353 | University of Central Punjab | March 2026*
