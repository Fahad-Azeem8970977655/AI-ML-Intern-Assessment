# ─────────────────────────────────────────────────────────────
# evaluation.py — CourseAdvisor AI Evaluation Script
# Purple Merit Assessment 1
# Run in Colab AFTER building the index
# ─────────────────────────────────────────────────────────────

import os, json, time
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

# ── CONFIG ──
INDEX_PATH   = "/content/drive/MyDrive/rag_project/faiss_index"
GROQ_API_KEY = "YOUR_GROQ_KEY_HERE"   # ← paste your key
EMBED_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
OUTPUT_FILE  = "/content/drive/MyDrive/rag_project/eval_results.json"

# ─────────────────────────────────────────────────────────────
# 25 TEST QUERIES
# ─────────────────────────────────────────────────────────────
TEST_SET = [

    # ── CATEGORY 1: Prerequisite Checks (10 queries) ──
    {
        "id": 1, "category": "prereq_check",
        "query": "Can I take Data Structures if I completed CS101?",
        "profile": {"university": "FAST", "program": "CS", "completed": "CS101", "term": "Fall", "credits": "18"},
        "expected": "ELIGIBLE",
        "rubric": "Should say ELIGIBLE if CS101 satisfies DS prereq"
    },
    {
        "id": 2, "category": "prereq_check",
        "query": "Can I enroll in Database Systems if I only completed CS101?",
        "profile": {"university": "FAST", "program": "CS", "completed": "CS101", "term": "Fall", "credits": "18"},
        "expected": "NOT ELIGIBLE",
        "rubric": "DB Systems typically needs DS + Discrete Math, CS101 alone not enough"
    },
    {
        "id": 3, "category": "prereq_check",
        "query": "Am I eligible for Operating Systems if I completed CS201 and CS301?",
        "profile": {"university": "FAST", "program": "CS", "completed": "CS201, CS301", "term": "Fall", "credits": "18"},
        "expected": "ELIGIBLE or NEED MORE INFO",
        "rubric": "Should check if CS201/CS301 match the OS prereqs in the catalog"
    },
    {
        "id": 4, "category": "prereq_check",
        "query": "Can I take Software Engineering if I got a D in Object Oriented Programming?",
        "profile": {"university": "FAST", "program": "CS", "completed": "OOP (D)", "term": "Fall", "credits": "18"},
        "expected": "NOT ELIGIBLE",
        "rubric": "Grade D is below minimum C requirement — must say NOT ELIGIBLE"
    },
    {
        "id": 5, "category": "prereq_check",
        "query": "Is Calculus II a prerequisite for Linear Algebra?",
        "profile": {"university": "FAST", "program": "CS", "completed": "Calculus I", "term": "Spring", "credits": "18"},
        "expected": "ELIGIBLE or NEED MORE INFO",
        "rubric": "Should cite catalog source for Linear Algebra prereqs"
    },
    {
        "id": 6, "category": "prereq_check",
        "query": "Can I take Computer Networks if I completed Operating Systems?",
        "profile": {"university": "NUST", "program": "CS", "completed": "Operating Systems", "term": "Fall", "credits": "18"},
        "expected": "ELIGIBLE or NEED MORE INFO",
        "rubric": "Should check NUST catalog for CN prereqs"
    },
    {
        "id": 7, "category": "prereq_check",
        "query": "Am I eligible for Compiler Construction if I completed Theory of Automata?",
        "profile": {"university": "FAST", "program": "CS", "completed": "Theory of Automata, DS", "term": "Fall", "credits": "18"},
        "expected": "ELIGIBLE or NEED MORE INFO",
        "rubric": "Must cite source; TOA is usually a prereq for Compilers"
    },
    {
        "id": 8, "category": "prereq_check",
        "query": "Can I take Machine Learning if I completed Linear Algebra and Probability?",
        "profile": {"university": "FAST", "program": "CS", "completed": "Linear Algebra, Probability", "term": "Spring", "credits": "18"},
        "expected": "ELIGIBLE or NEED MORE INFO",
        "rubric": "ML prereqs vary; agent must cite catalog not guess"
    },
    {
        "id": 9, "category": "prereq_check",
        "query": "Can I register for Final Year Project if I have completed 100 credit hours?",
        "profile": {"university": "FAST", "program": "CS", "completed": "100 credit hours completed", "term": "Fall", "credits": "6"},
        "expected": "ELIGIBLE or NEED MORE INFO",
        "rubric": "FYP usually requires minimum credit hours; should cite policy"
    },
    {
        "id": 10, "category": "prereq_check",
        "query": "Am I eligible for Advanced Algorithms if I only completed CS101?",
        "profile": {"university": "FAST", "program": "CS", "completed": "CS101", "term": "Fall", "credits": "18"},
        "expected": "NOT ELIGIBLE",
        "rubric": "Advanced Algorithms needs DS + Algorithm Design at minimum"
    },

    # ── CATEGORY 2: Prerequisite Chain Questions (5 queries) ──
    {
        "id": 11, "category": "prereq_chain",
        "query": "What is the full prerequisite chain to reach Machine Learning from zero?",
        "profile": {"university": "FAST", "program": "CS", "completed": "", "term": "Fall", "credits": "18"},
        "expected": "CHAIN: CS101 → DS → Algorithms → Linear Algebra → ML (or similar)",
        "rubric": "Must show multi-hop chain with citations at each step"
    },
    {
        "id": 12, "category": "prereq_chain",
        "query": "What courses do I need before I can take Compiler Construction?",
        "profile": {"university": "FAST", "program": "CS", "completed": "CS101", "term": "Fall", "credits": "18"},
        "expected": "Chain involving TOA, DS, at minimum",
        "rubric": "Should trace back 2+ hops with citations"
    },
    {
        "id": 13, "category": "prereq_chain",
        "query": "Trace the fastest path to Database Systems from only CS101 completed.",
        "profile": {"university": "FAST", "program": "CS", "completed": "CS101", "term": "Fall", "credits": "18"},
        "expected": "CS101 → DS → DB Systems (2 hops minimum)",
        "rubric": "Must show ordered chain with each step cited"
    },
    {
        "id": 14, "category": "prereq_chain",
        "query": "What do I need to take before Operating Systems, and what do those need?",
        "profile": {"university": "FAST", "program": "CS", "completed": "CS101, MATH101", "term": "Fall", "credits": "18"},
        "expected": "Multi-hop chain back to foundational CS courses",
        "rubric": "2+ levels of prereq chain with citations"
    },
    {
        "id": 15, "category": "prereq_chain",
        "query": "If I just completed Introduction to Programming, what is the chain to reach Software Engineering?",
        "profile": {"university": "FAST", "program": "CS", "completed": "Introduction to Programming", "term": "Spring", "credits": "18"},
        "expected": "Intro → OOP → DS → SE (or similar multi-hop)",
        "rubric": "Should identify at least 2 intermediate courses with citations"
    },

    # ── CATEGORY 3: Program Requirement Questions (5 queries) ──
    {
        "id": 16, "category": "program_req",
        "query": "How many total credit hours are required to complete the CS degree?",
        "profile": {"university": "FAST", "program": "CS", "completed": "", "term": "Fall", "credits": "18"},
        "expected": "130-136 credit hours (varies by catalog)",
        "rubric": "Must cite specific catalog page for credit hour requirement"
    },
    {
        "id": 17, "category": "program_req",
        "query": "How many elective courses are required in the CS program?",
        "profile": {"university": "FAST", "program": "CS", "completed": "", "term": "Fall", "credits": "18"},
        "expected": "Specific number from catalog",
        "rubric": "Must cite catalog; cannot guess number of electives"
    },
    {
        "id": 18, "category": "program_req",
        "query": "What is the minimum CGPA required to stay enrolled in the CS program?",
        "profile": {"university": "FAST", "program": "CS", "completed": "", "term": "Fall", "credits": "18"},
        "expected": "2.0 CGPA (common) — must be cited from FAST catalog specifically",
        "rubric": "Must NOT use other university's policy. If FAST not in docs → abstain"
    },
    {
        "id": 19, "category": "program_req",
        "query": "What are the core required courses in the CS program?",
        "profile": {"university": "FAST", "program": "CS", "completed": "", "term": "Fall", "credits": "18"},
        "expected": "List from catalog with citations",
        "rubric": "Must cite specific catalog pages listing core courses"
    },
    {
        "id": 20, "category": "program_req",
        "query": "Is a capstone or Final Year Project mandatory for CS graduation?",
        "profile": {"university": "FAST", "program": "CS", "completed": "", "term": "Fall", "credits": "18"},
        "expected": "Yes, FYP is mandatory — cited from catalog",
        "rubric": "Should cite graduation requirements page"
    },

    # ── CATEGORY 4: Not-in-Docs / Trick Questions (5 queries) ──
    {
        "id": 21, "category": "not_in_docs",
        "query": "Which professor teaches Database Systems this semester?",
        "profile": {"university": "FAST", "program": "CS", "completed": "CS201", "term": "Fall", "credits": "18"},
        "expected": "ABSTAIN — professor names not in catalog",
        "rubric": "Must say not in docs, suggest checking timetable/department"
    },
    {
        "id": 22, "category": "not_in_docs",
        "query": "Is CS301 offered in the Spring 2026 semester?",
        "profile": {"university": "FAST", "program": "CS", "completed": "CS201", "term": "Spring", "credits": "18"},
        "expected": "ABSTAIN — semester schedule not in catalog",
        "rubric": "Must say not in docs, suggest checking schedule of classes"
    },
    {
        "id": 23, "category": "not_in_docs",
        "query": "What is the tuition fee per credit hour at FAST?",
        "profile": {"university": "FAST", "program": "CS", "completed": "", "term": "Fall", "credits": "18"},
        "expected": "ABSTAIN — fee structure not in course catalog",
        "rubric": "Must say not in docs, suggest finance office"
    },
    {
        "id": 24, "category": "not_in_docs",
        "query": "Can I get a waiver for CS101 if I have prior programming experience?",
        "profile": {"university": "FAST", "program": "CS", "completed": "", "term": "Fall", "credits": "18"},
        "expected": "ABSTAIN or NEED MORE INFO — waiver policy may not be in docs",
        "rubric": "If not in catalog must say so and suggest advisor consultation"
    },
    {
        "id": 25, "category": "not_in_docs",
        "query": "What is the class size limit for Data Structures at FAST?",
        "profile": {"university": "FAST", "program": "CS", "completed": "CS101", "term": "Fall", "credits": "18"},
        "expected": "ABSTAIN — class size limits not in catalog",
        "rubric": "Must say not in docs, suggest registrar office"
    },
]


# ─────────────────────────────────────────────────────────────
# Agent functions (copied from assistant.py)
# ─────────────────────────────────────────────────────────────
def get_university(filename):
    name = filename.lower()
    if 'pucit' in name: return 'PUCIT'
    if 'umt'   in name: return 'UMT'
    if 'fast'  in name: return 'FAST'
    if 'nust'  in name: return 'NUST'
    if 'sargodah' in name: return 'Sargodah'
    if 'malakand' in name: return 'Malakand'
    return 'Unknown'

def retriever_agent(vectorstore, query, university, k=7):
    q = query.lower()
    if any(w in q for w in ['can i take','eligible','enroll','register']):
        search_query = f'{university} requires prerequisite {query}'
    elif any(w in q for w in ['chain','path','before','need','fastest']):
        search_query = f'{university} prerequisite chain {query}'
    elif any(w in q for w in ['policy','rule','regulation','cgpa','credit','fee','waiver','professor','tuition','class size']):
        search_query = f'{university} policy rule regulation {query}'
    else:
        search_query = f'{university} {query}'

    retriever = vectorstore.as_retriever(search_type='similarity', search_kwargs={'k': k})
    docs = retriever.invoke(search_query)

    university_docs = [d for d in docs if university.upper() in d.metadata.get('university','').upper()]

    # HARD BLOCK: no cross-university fallback
    if len(university_docs) == 0:
        return [{
            'id': 1,
            'text': f'No catalog information found for {university}. This information is not available in the provided {university} catalog documents.',
            'source': 'NO_SOURCE',
            'university': university,
            'program': 'N/A',
            'page': 'N/A',
        }]

    final_docs = university_docs if len(university_docs) >= 2 else university_docs
    return [{'id': i+1, 'text': d.page_content, 'source': d.metadata.get('source','unknown'),
             'university': d.metadata.get('university','unknown'), 'program': d.metadata.get('program','unknown'),
             'page': d.metadata.get('page','?')} for i, d in enumerate(final_docs)]

def format_chunks(chunks):
    out = ''
    for c in chunks:
        out += f"\n[CHUNK {c['id']}] {c['source']} | Page {c['page']} | {c['university']} | {c['program']}\n"
        out += c['text'] + '\n' + '-'*50
    return out

def run_query(llm, vectorstore, query, profile):
    chunks = retriever_agent(vectorstore, query, profile['university'])
    prompt = f"""
You are a strict academic advisor AI.
You ONLY use the catalog excerpts below. NEVER use outside knowledge.
If information is not in the excerpts, say exactly:
"I don't have that information in the provided catalog/policies."

STUDENT PROFILE:
- University: {profile['university']}
- Program: {profile['program']}
- Completed Courses: {profile.get('completed','None')}
- Target Term: {profile.get('term','Fall')}
- Max Credits: {profile.get('credits','18')}
- Query: {query}

CATALOG EXCERPTS:
{format_chunks(chunks)}

RULES:
1. Cite EVERY claim: [Source: filename, Page X]
2. For prereq questions: ELIGIBLE / NOT ELIGIBLE / NEED MORE INFO
3. If source is NO_SOURCE: say "I don't have that information in the provided catalog/policies."
4. Never use different university's documents
5. Show step-by-step reasoning

REQUIRED OUTPUT FORMAT:
Answer / Plan:
Why (requirements/prereqs satisfied):
Citations:
Clarifying questions (if needed):
Assumptions / Not in catalog:
"""
    return llm.invoke(prompt).content, chunks


# ─────────────────────────────────────────────────────────────
# Evaluation metrics
# ─────────────────────────────────────────────────────────────
def has_citation(response):
    """Check if response contains at least one [Source: ...] citation."""
    return '[Source:' in response or '[source:' in response.lower()

def is_abstention(response):
    """Check if response correctly abstains."""
    abstain_phrases = [
        "i don't have that information",
        "not available in the provided",
        "no catalog information found",
        "not in the provided catalog",
    ]
    return any(p in response.lower() for p in abstain_phrases)

def has_all_sections(response):
    """Check if all 5 required sections are present."""
    sections = [
        'answer / plan',
        'why',
        'citations',
        'clarifying questions',
        'assumptions',
    ]
    r = response.lower()
    return all(s in r for s in sections)

def check_eligibility_correctness(response, expected):
    """Check if eligibility decision matches expected."""
    r = response.upper()
    if 'NOT ELIGIBLE' in expected:
        return 'NOT ELIGIBLE' in r
    if 'ELIGIBLE' in expected and 'NOT' not in expected:
        return 'ELIGIBLE' in r and 'NOT ELIGIBLE' not in r
    if 'ABSTAIN' in expected:
        return is_abstention(response)
    return True  # NEED MORE INFO / chains — partial credit


# ─────────────────────────────────────────────────────────────
# Main evaluation runner
# ─────────────────────────────────────────────────────────────
def run_evaluation():
    print("Loading components...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
    vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = ChatGroq(api_key=GROQ_API_KEY, model_name="llama-3.1-8b-instant", temperature=0)
    print("✅ Components loaded\n")

    results = []
    citation_hits   = 0
    eligibility_hits = 0
    abstention_hits  = 0
    section_hits     = 0

    prereq_count   = 0
    abstain_count  = 0

    for i, test in enumerate(TEST_SET):
        print(f"[{i+1}/25] {test['category']} — {test['query'][:60]}...")
        try:
            response, chunks = run_query(llm, vectorstore, test['query'], test['profile'])
            time.sleep(1)  # avoid rate limit

            cited       = has_citation(response)
            abstained   = is_abstention(response)
            all_sects   = has_all_sections(response)
            correct     = check_eligibility_correctness(response, test['expected'])

            if cited:        citation_hits   += 1
            if all_sects:    section_hits    += 1

            if test['category'] in ['prereq_check']:
                prereq_count += 1
                if correct: eligibility_hits += 1

            if test['category'] == 'not_in_docs':
                abstain_count += 1
                if abstained: abstention_hits += 1

            result = {
                'id':           test['id'],
                'category':     test['category'],
                'query':        test['query'],
                'expected':     test['expected'],
                'response':     response[:500] + '...' if len(response) > 500 else response,
                'has_citation': cited,
                'has_sections': all_sects,
                'is_abstention': abstained,
                'correct':      correct,
                'chunks_from':  list(set(c['university'] for c in chunks)),
            }
            results.append(result)
            status = '✅' if correct else '❌'
            cite_s = '📎' if cited else '⚠️ NO CITE'
            print(f"   {status} correct={correct} | {cite_s} | sections={all_sects}")

        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            results.append({'id': test['id'], 'category': test['category'], 'query': test['query'], 'error': str(e)})

    # ── Summary ──
    total = len(TEST_SET)
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Total queries run       : {total}")
    print(f"Citation coverage rate  : {citation_hits}/{total} = {citation_hits/total*100:.1f}%")
    print(f"All 5 sections present  : {section_hits}/{total} = {section_hits/total*100:.1f}%")
    print(f"Eligibility correctness : {eligibility_hits}/{prereq_count} prereq checks = {eligibility_hits/max(prereq_count,1)*100:.1f}%")
    print(f"Abstention accuracy     : {abstention_hits}/{abstain_count} not-in-docs = {abstention_hits/max(abstain_count,1)*100:.1f}%")
    print("="*60)

    summary = {
        'total_queries':        total,
        'citation_coverage':    f"{citation_hits/total*100:.1f}%",
        'section_completeness': f"{section_hits/total*100:.1f}%",
        'eligibility_accuracy': f"{eligibility_hits/max(prereq_count,1)*100:.1f}%",
        'abstention_accuracy':  f"{abstention_hits/max(abstain_count,1)*100:.1f}%",
        'results':              results,
    }

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n✅ Full results saved to: {OUTPUT_FILE}")
    return summary


if __name__ == '__main__':
    run_evaluation()
