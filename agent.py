# ── FULL 4-AGENT ASSISTANT ──
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

INDEX_PATH   = "/content/drive/MyDrive/rag_project/faiss_index"
EMBED_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_API_KEY = ""  # ← paste your Groq key here

# ── Load components ──
def load_components():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"}
    )
    vectorstore = FAISS.load_local(
        INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 7}
    )
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant",  # ← UPDATED
        temperature=0
    )
    return retriever, llm

# ── AGENT 1: Intake ──
def intake_agent(llm, query, completed, program, university, term, credits):
    query_lower = query.lower()
    is_prereq_query = any(w in query_lower for w in [
        "can i take", "eligible", "prerequisite", "prereq",
        "completed", "enroll", "register", "grade"
    ])
    is_general_query = any(w in query_lower for w in [
        "how many credits", "total credits", "core courses",
        "program require", "capstone", "elective", "fee",
        "tuition", "professor", "available", "online",
        "class size", "minimum cgpa", "policy", "waiver",
        "trace", "path", "chain", "fastest", "need before"
    ])

    if is_general_query:
        if not university or not program:
            return "MISSING: Please tell me your university and program (CS/IT/SE/DS)."
        return "PROFILE_COMPLETE"

    if is_prereq_query:
        missing = []
        if not university: missing.append("Which university are you from?")
        if not program:    missing.append("What is your program? (CS/IT/SE/DS)")
        if not completed:  missing.append("Which courses have you already completed?")
        if missing:
            return "MISSING: " + " | ".join(missing)
        return "PROFILE_COMPLETE"

    if not university or not program:
        return "MISSING: Please provide your university and program."
    return "PROFILE_COMPLETE"

# ── AGENT 2: Retriever ──
def retriever_agent(retriever, query, university):
    query_lower = query.lower()

    if any(w in query_lower for w in ["can i take","eligible","enroll","register"]):
        search_query = f"{university} requires prerequisite {query}"
    elif any(w in query_lower for w in ["chain","path","before","need","fastest"]):
        search_query = f"{university} prerequisite chain {query}"
    elif any(w in query_lower for w in ["policy","rule","regulation","cgpa","credit","fee","waiver"]):
        search_query = f"{university} policy rule regulation {query}"
    else:
        search_query = f"{university} {query}"

    docs = retriever.invoke(search_query)

    university_docs = [
        d for d in docs
        if university.upper() in d.metadata.get("university", "").upper()
    ]
    final_docs = university_docs if len(university_docs) >= 2 else docs

    chunks = []
    for i, doc in enumerate(final_docs):
        chunks.append({
            "id": i + 1,
            "text": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "university": doc.metadata.get("university", "unknown"),
            "program": doc.metadata.get("program", "unknown"),
            "page": doc.metadata.get("page", "?")
        })
    return chunks

# ── Format chunks for prompt ──
def format_chunks(chunks):
    out = ""
    for c in chunks:
        out += f"\n[CHUNK {c['id']}] {c['source']} | Page {c['page']} | {c['university']} | {c['program']}\n"
        out += c['text'] + "\n"
        out += "-" * 50
    return out

# ── AGENT 3: Planner ──
def planner_agent(llm, query, completed, program, university, term, credits, chunks):
    prompt = f"""
You are a strict academic advisor AI.
You ONLY use the catalog excerpts below. NEVER use outside knowledge.
If information is not in the excerpts, say exactly:
"I don't have that information in the provided catalog/policies."

STUDENT PROFILE:
- University: {university}
- Program: {program}
- Completed Courses: {completed or 'None'}
- Target Term: {term}
- Max Credits: {credits}
- Query: {query}

CATALOG EXCERPTS:
{format_chunks(chunks)}

RULES:
1. Cite EVERY claim like this: [Source: filename, Page X]
2. For prerequisite questions state: ELIGIBLE / NOT ELIGIBLE / NEED MORE INFO
3. For eligibility — check if completed courses satisfy the prereq listed in excerpts
4. If student grade is D or below — state NOT ELIGIBLE (minimum C required)
5. Show reasoning step by step
6. Never guess — if not in excerpts, abstain

REQUIRED OUTPUT FORMAT — use exactly these 5 sections:

Answer / Plan:
[your answer]

Why (requirements/prereqs satisfied):
[reasoning with citations]

Citations:
[Source: file, Page X — what it says]

Clarifying questions (if needed):
[questions or write: None]

Assumptions / Not in catalog:
[what you assumed or could not find]
"""
    return llm.invoke(prompt).content

# ── AGENT 4: Verifier ──
def verifier_agent(llm, response):
    prompt = f"""
Check this academic advisor response:

{response}

Verify:
1. Does every factual claim have a [Source: ...] citation?
2. Are all 5 required sections present?
3. Any hallucinated facts not from the excerpts?

Output either:
VERIFIED: Response is properly cited and complete.
OR
ISSUES: [list problems found]
"""
    return llm.invoke(prompt).content

# ── MAIN PIPELINE ──
def ask(
    query,
    completed="",
    program="",
    university="",
    term="Fall",
    credits="18"
):
    print("\n" + "=" * 60)
    retriever, llm = load_components()

    print("[Agent 1] Checking profile...")
    intake = intake_agent(llm, query, completed, program, university, term, credits)
    if "MISSING" in intake:
        print("\nProfile incomplete. Questions for student:")
        print(intake)
        return intake

    print("[Agent 2] Searching catalog...")
    chunks = retriever_agent(retriever, query, university)
    print(f"  Retrieved {len(chunks)} chunks from: "
          f"{set(c['university'] for c in chunks)}")

    print("[Agent 3] Generating answer...")
    answer = planner_agent(llm, query, completed, program,
                           university, term, credits, chunks)

    print("[Agent 4] Verifying citations...")
    verification = verifier_agent(llm, answer)

    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)
    print(answer)
    print("\n--- Verification Result ---")
    print(verification)
    return answer

print("Assistant loaded! Ready to answer questions.")