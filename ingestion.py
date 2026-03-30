import os
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_PATH  = "/content/drive/MyDrive/rag_project/data"
INDEX_PATH = "/content/drive/MyDrive/rag_project/faiss_index"

# ── extract university name from filename ──
def get_university(filename):
    name = filename.lower()
    if "pucit"    in name: return "PUCIT"
    if "umt"      in name: return "UMT"
    if "fast"     in name: return "FAST"
    if "nust"     in name: return "NUST"
    if "sargodah" in name: return "Sargodah"
    if "malakand" in name: return "Malakand"
    if "iit"      in name: return "IIT"
    if "usa"      in name: return "USA"
    return "Unknown"

# ── extract program name from filename ──
def get_program(filename):
    name = filename.lower()
    if "data-science" in name or "data science" in name or "ds" in name: return "DS"
    if "software-engineering" in name or "se" in name or "bese" in name or "bsse" in name: return "SE"
    if "information-technology" in name or "it" in name: return "IT"
    if "cs" in name or "cse" in name: return "CS"
    if "rule" in name or "regulation" in name or "handbook" in name: return "POLICY"
    return "GENERAL"

# ── CHANGE 1: clean text ──
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\.{3,}', ' ', text)
    text = re.sub(r'\x0c', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return text.strip()

# ── CHANGE 2: prereq extractor ──
# Finds table rows like "CS2001  Data Structures  CS1004"
# Converts them into clear sentences the retriever can find
def extract_prereq_text(text):
    lines = text.split('\n')
    enriched = []
    pattern = re.compile(
        r'([A-Z]{2,4}[-\s]?\d{3,4})\s+'
        r'([A-Za-z\s&/]{5,50}?)\s+'
        r'([A-Z]{2,4}[-\s]?\d{3,4})',
        re.IGNORECASE
    )
    for line in lines:
        match = pattern.search(line)
        if match:
            code   = match.group(1).strip()
            title  = match.group(2).strip()
            prereq = match.group(3).strip()
            clear  = (
                f"Course {code} ({title}) requires {prereq} as a prerequisite. "
                f"To enroll in {title} you must have completed {prereq}."
            )
            enriched.append(clear)
        else:
            enriched.append(line)
    return '\n'.join(enriched)

# ── STEP 1: Load all PDFs ──
print("Loading PDFs...")
all_docs = []
for filename in os.listdir(DATA_PATH):
    if not filename.endswith(".pdf"):
        continue
    path = os.path.join(DATA_PATH, filename)
    try:
        loader = PyPDFLoader(path)
        pages  = loader.load()
        univ   = get_university(filename)
        prog   = get_program(filename)
        for page in pages:
            # CHANGE 3: apply both clean + prereq extractor
            page.page_content = clean_text(page.page_content)
            page.page_content = extract_prereq_text(page.page_content)
            page.metadata["source"]     = filename
            page.metadata["university"] = univ
            page.metadata["program"]    = prog
        all_docs.extend(pages)
        print(f"  OK  {filename}  ({len(pages)} pages) [{univ} | {prog}]")
    except Exception as e:
        print(f"  FAIL {filename}: {e}")

print(f"\nTotal pages loaded: {len(all_docs)}")

# ── STEP 2: Chunk ──
print("\nChunking...")
# CHANGE 4: bigger chunks + more overlap to keep prereq context together
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunks = splitter.split_documents(all_docs)
print(f"Total chunks: {len(chunks)}")

# ── STEP 3: Embed + Save ──
print("\nBuilding FAISS index (takes 3-5 mins)...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local(INDEX_PATH)
print(f"\nDone! Index saved to: {INDEX_PATH}")