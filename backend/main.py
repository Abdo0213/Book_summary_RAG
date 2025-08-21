from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import sqlite3, os
from passlib.context import CryptContext
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import re
import hashlib
from datetime import datetime
from typing import List, Dict, Optional
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import chromadb
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredEPubLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_cohere import ChatCohere
from langchain.schema import Document, StrOutputParser
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import json
import threading


load_dotenv()

app = FastAPI()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

CHROMA_DIR = "/content/chroma_books"
COLLECTION_NAME = "books_rag"

file_path="/content/attention.pdf"
document_type=""
model_name = "google/flan-t5-large"

UPLOAD_DIR = "../uploads"
SUMMARY_DIR = "../summaries"

######## LLM MOdELS, EMBEDDING, Environment Variable ########

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(SUMMARY_DIR, exist_ok=True)

cohere_key = os.getenv("COHERE_API_KEY")
if cohere_key:
    os.environ["CO_API_KEY"] = cohere_key
else:
    print("⚠️ Warning: COHERE_API_KEY not set, skipping Cohere integration.")


model_name = "google/flan-t5-base"  # or your model
llm_save_path = "./models/flan-t5-base"  # local folder
# Download and save
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer.save_pretrained(llm_save_path)
model.save_pretrained(llm_save_path)
print(f"✅ Model saved to {llm_save_path}")


embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
save_path_embedding = "./models/all-MiniLM-L6-v2"
# Download and save
emb = SentenceTransformer(embedding_model)
emb.save(save_path_embedding)
print(f"✅ Embedding model saved at {save_path_embedding}")


########### Usage ###########
local_path = "./models/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(local_path)
model = AutoModelForSeq2SeqLM.from_pretrained(local_path)
embeddings = HuggingFaceEmbeddings(model_name="./models/all-MiniLM-L6-v2")
cohere_llm = ChatCohere(model="command-a-03-2025")


def get_conn():
    conn = sqlite3.connect("db.sqlite3")
    conn.execute("""CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        email TEXT UNIQUE,
        password TEXT
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS books (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        title TEXT,
        filepath TEXT,
        favorite INTEGER DEFAULT 0,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )""")
    return conn


###### Data Models ######
class User(BaseModel):
    username: str
    email: str
    password: str

class File(BaseModel):
    path: str
    flag: bool

class QuestionRequest(BaseModel):
    question: str
    n_results: int = 5

class AnswerResponse(BaseModel):
    answer: str

class SummaryResponse(BaseModel):
    summary: str

class TranslateRequest(BaseModel):
    content: str
    #target_language: str = "Arabic"   # default to Arabic

class TranslateResponse(BaseModel):
    translated_text: str


#Initialize Vector DB

vs = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR
)

#Initialize Splitter
# Initialize components (should be done once, outside the function)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=900,
    chunk_overlap=100,
    length_function=len,
)

#Initialize Load function
def load_file(path: str) -> List[Document]:
    ext = path.lower().split(".")[-1]
    if ext == "pdf":
        loader = PyPDFLoader(path)
        docs = loader.load()  # returns per-page docs
    elif ext in ("txt", "md"):
        loader = TextLoader(path, encoding="utf-8")
        docs = loader.load()
    elif ext in ("epub",):
        loader = UnstructuredEPubLoader(path)
        docs = loader.load()
    else:
        raise ValueError(f"Unsupported file type: .{ext}")
    return docs

#######################################
############### Utilis ################
#######################################

def clean_text(text: str) -> str:
    text = re.sub(r"[\x00-\x1F\x7F]", " ", text)   # Remove control chars
    text = re.sub(r"\n+", "\n", text)              # Collapse newlines
    text = re.sub(r"\s+", " ", text)               # Collapse spaces
    text = re.sub(r"<.*?>", "", text)              # Remove HTML tags
    return text.strip()

# ========== STEP 2: Chunk-level ==========
def make_chunks(page_docs: List[Document], file_name: str, ts: str) -> List[Document]:
    chunk_docs = text_splitter.split_documents(page_docs)
    for i, d in enumerate(chunk_docs):
        page_num = d.metadata.get('page', d.metadata.get('approx_page_group', 1))
        d.metadata.update({
            #"file_id": file_id,
            "file_name": file_name,
            "level": "chunk",
            "chunk_id": i + 1,
            "page_start": page_num,
            "page_end": page_num,
            "ts": ts
        })
    return chunk_docs

def chunk_text(text, max_tokens=400):
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""
    current_len = 0
    for sentence in sentences:
        current_chunk += sentence + '. '
        current_len += len(sentence.split())
        if current_len >= max_tokens:
            chunks.append(current_chunk.strip())
            current_chunk = ""
            current_len = 0
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# ========== STEP 6: Index into Chroma (overwrite mode) ==========
def index_documents(all_docs, vs):
    #vs.delete_collection()  # clear old
    vs.add_documents(all_docs)
    vs.persist()
    # return vs._collection.count()

def name_file(path: str):
    match = re.search(r"[^/\\]+$", path)
    if match:
        filename = match.group(0)
    name_only = re.sub(r"\.[^.]+$", "", filename)

def get_relevant_chunks_with_scores(question, level="chunk", n_results=5):
    if level == "chunk":
        k = n_results
    elif level == "page":
        k = n_results * 2
    elif level == "section":
        k = n_results * 3
    elif level == "book":
        k = n_results * 5
    else:
        k = n_results

    results = vs.similarity_search_with_score(question, k=k)
    sorted_results = sorted(results, key=lambda x: x[1]) 

    return [
        f"📄 **File:** {doc.metadata.get('file_name', 'Unknown')}  \n"
        f"📑 **Page:** {doc.metadata.get('page_start', 'N/A')}  \n"
        f"🔹 **Chunk ID:** {doc.metadata.get('chunk_id', 'N/A')}  \n\n"
        f"{doc.page_content}"
        for doc, score in sorted_results
    ]

#######################################
############## EndUtilis ##############
#######################################

######### PROMPTS #########

# summary_prompt = ChatPromptTemplate.from_template(
#     """You are an expert summarizer.
#     Summarize the following research paper chunks into a structured summary.
#     Include: Objective, Methods, Results, and Conclusion.

#     Paper Content:
#     {context}

#     Provide the summary in bullet points.
#     """
# )

summary_prompt = ChatPromptTemplate.from_template(
    """You are an expert summarizer. 
    Summarize the following text (which may be a research paper OR a book). 

    - If it is a **research paper**, provide a structured summary with:
        - Objective
        - Methods
        - Results
        - Conclusion

    - If it is a **book** (non-research content), summarize it with:
        - Main Themes
        - Key Ideas/Arguments
        - Supporting Evidence or Examples
        - Conclusion / Takeaways

    Text to summarize:
    {context}

    Provide the summary in **clear bullet points**.
    """
)

qa_prompt = ChatPromptTemplate.from_template(
    """You are an expert research assistant.
    Use the provided paper chunks to answer the user’s question.
    Be clear, concise, and reference the paper name or source URL when useful.

    User Question:
    {question}

    Relevant Paper Chunks:
    {context}

    Answer in an academic but easy-to-understand style.
    """
)

rewrite_prompt = PromptTemplate.from_template("""
You are a helpful assistant that rewrites raw answers into a beautiful, well-structured response.
Rules:
- Use Markdown formatting (## headings, **bold**, bullet points ✅)
- Highlight important terms with **bold** or 🔥 emoji
- Keep it clear, concise, and engaging
- Remove unnecessary filler

Raw Answer:
{raw_answer}

Rewritten (beautiful, Markdown-styled) Answer:
""")

rewrite_query_prompt = PromptTemplate.from_template("""
You are a smart academic assistant.  
Your job is to analyze the user query and decide the proper retrieval level.  

### 🎯 Tasks:
1. **Rewrite the user query** into a clearer academic question (if needed).  
2. **Decide retrieval level** (choose exactly one):  
    - 'chunk' → fine-grained page-level detail  
    - 'page' → page-level summary  
    - 'section' → chapter-level summary  
    - 'book' → global understanding  

### 📌 Rules:
- Output must be **strict JSON**.  
- JSON Keys: `"rewritten_query"` and `"level"`.  

### User Query:
{raw_query}

### ✅ Expected Output:
{{
    "rewritten_query": "Your improved academic question here",
    "level": "chunk | page | section | book"
}}
""")

answer_prompt = PromptTemplate.from_template("""
You are a helpful academic assistant.  
Answer the user query using the provided context.  

Rules:  
- Use **Markdown formatting** (## headings, **bold**, bullet points ✅).  
- Be **concise, clear, and structured**.  
- Highlight key insights with 🔥 or **bold**.  
- If the answer is uncertain, state so clearly.  

User Query: {query}  

Context (from retrieved documents):  
{context}  

Final Answer (well-structured, Markdown-styled):  
""")

# Translation prompt
translate_prompt = PromptTemplate.from_template("""
    You are a professional translator.
    Translate the following text from English to **Arabic**, keeping the same meaning, tone, and style.
    If the text contains emojis, keep them.

    English Text:
    {raw_answer}

    Arabic Translation:
""")

# ======== Chat Prompt Templates ========
page_prompt_template = ChatPromptTemplate.from_template(
    """You are an expert summarizer.
Summarize the following text into clear, concise bullet points.

Guidelines:
- Capture all important information, key ideas, and concepts.
- Organize the summary in a logical order.
- Make each bullet point informative and self-contained.
- Avoid unnecessary repetition or filler text.
- Make at least 5 bullets.

Text:
{content}

Summary (Bullet Points):
- """
)

section_prompt_template = ChatPromptTemplate.from_template(
    """You are an expert summarizer.
Summarize the following content into 3-5 clear, concise bullet points capturing all important ideas.

Content:
{content}

Chunk Summary:
- """
)

book_prompt_template = ChatPromptTemplate.from_template(
    """You are an expert book summarizer and scholar.
Summarize the following book or multi-chapter content into a structured summary. Include:

- **Central Thesis/Main Plot:** What is the primary argument or story arc?
- **Key Themes & Topics:** What are the recurring ideas, motifs, or arguments?
- **Overall Conclusion/Resolution:** What is the final takeaway or conclusion?

Content:
{content}

Provide the summary in clear bullet points.
- """
)


def summarize(n_results=8):
    results = vs.similarity_search("summary", k=n_results)
    context = "\n\n".join([doc.page_content for doc in results])
    chain = summary_prompt | cohere_llm | StrOutputParser()
    return chain.invoke({"context": context})

def generate_answer(question, n_results=5):
    relevant_chunks = get_relevant_chunks_with_scores(question, n_results)
    context = "\n\n".join(relevant_chunks)
    chain = qa_prompt | cohere_llm | StrOutputParser()
    return chain.invoke({"context": context, "question": question})

def beautify_answer(raw_answer, llm):
    chain = rewrite_prompt | llm | StrOutputParser()
    return chain.invoke({"raw_answer": raw_answer})

def route_query(query, llm):
    chain = rewrite_query_prompt | llm | StrOutputParser()
    return chain.invoke({"raw_query": query})

def ask_with_level(query, llm, vs):
    route_result = route_query(query, llm)
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", route_result.strip(), flags=re.DOTALL)
    print("🔎 Router Output:", cleaned)

    # If it's a string, parse it. If it's already a dict, keep it.
    if isinstance(cleaned, str):
        route_result = json.loads(cleaned)

    rewritten_query = route_result["rewritten_query"]
    level = route_result["level"]

    context = "\n\n---\n\n".join(get_relevant_chunks_with_scores(rewritten_query, level))

    chain = answer_prompt | llm | StrOutputParser()
    return chain.invoke({"query": rewritten_query, "context": context})

def translate_to_arabic(raw_answer, llm):
    chain = translate_prompt | llm | StrOutputParser()
    return chain.invoke({"raw_answer": raw_answer})

#################################### Summary with levels ####################################
# ========== STEP 1: Normalize Pages with Cleaning ==========
def normalize_to_pages(raw_docs: List[Document]) -> List[Document]:
    """Convert raw docs into page-like docs and clean the text"""

    def clean_doc_content(doc: Document) -> Document:
        cleaned_content = clean_text(doc.page_content)
        return Document(page_content=cleaned_content, metadata=doc.metadata)

    if len(raw_docs) > 1:  # PDF-like
        page_docs = [clean_doc_content(doc) for doc in raw_docs]
        for idx, doc in enumerate(page_docs, start=1):
            if 'page' not in doc.metadata:
                doc.metadata['page'] = idx
    else:  # TXT-like → simulate pages
        chunks_tmp = text_splitter.split_documents(raw_docs)
        PAGE_CHUNKS = 3
        page_docs = []
        for i in range(0, len(chunks_tmp), PAGE_CHUNKS):
            merged_content = "\n\n".join([clean_text(c.page_content) for c in chunks_tmp[i:i+PAGE_CHUNKS]])
            page_docs.append(Document(
                page_content=merged_content,
                metadata={"approx_page_group": i // PAGE_CHUNKS + 1,
                          "page": i // PAGE_CHUNKS + 1}
            ))

    return page_docs

# ======== Hugging Face Summarizer ========
def get_fast_summarizer(model_path=llm_save_path):
    """Initialize a Hugging Face summarization pipeline from local path"""
    return pipeline(
        "summarization",
        model=model_path,        # local folder path
        tokenizer=model_path,    # use tokenizer from same folder
        device=0 if torch.cuda.is_available() else -1,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
# ======== Simplified Summarization Functions ========
#Without chunk splitting
# def summarize_page(text, summarizer):
#     try:
#         prompt = page_prompt_template.format(content=text)
#         summary = summarizer(prompt, max_length=512, do_sample=False, truncation=False)
#         bullet_text = summary[0]['summary_text'].strip()
#         if not bullet_text.startswith('-'):
#             bullet_text = '- ' + bullet_text.replace('\n', '\n- ')
#         return bullet_text

#     except Exception as e:
#         print(f"Page summarization error: {e}")
#         sentences = text.split('. ')
#         return '. '.join(sentences[:10]) + '.'

def summarize_section(text, summarizer):
    try:
        prompt = section_prompt_template.format(content=text)
        summary = summarizer(prompt, max_length=512, do_sample=False, truncation=False)
        bullet_text = summary[0]['summary_text'].strip()
        if not bullet_text.startswith('-'):
            bullet_text = '- ' + bullet_text.replace('\n', '\n- ')
        return bullet_text

    except Exception as e:
        print(f"Section summarization error: {e}")
        sentences = text.split('. ')
        return '. '.join(sentences[:10]) + '.'

def summarize_book(text, summarizer):
    try:
        final_prompt = book_prompt_template.format(content=text)
        final_summary = summarizer(final_prompt, max_length=600, do_sample=False, truncation=False)
        bullet_text = final_summary[0]['summary_text'].strip()
        if not bullet_text.startswith('-'):
            bullet_text = '- ' + bullet_text.replace('\n', '\n- ')
        return bullet_text

    except Exception as e:
        print(f"Book summarization error: {e}")
        sentences = text.split('. ')
        return '. '.join(sentences[:10]) + '.'

# ========== STEP 2: Updated Pipeline Functions ==========
def make_page_summaries(page_docs: List[Document], summarizer, file_name: str, ts: str):
    page_summary_docs = []
    for idx, pdoc in enumerate(page_docs, start=1):
        page_num = pdoc.metadata.get('page', idx)
        summary = summarize_page(pdoc.page_content, summarizer) # Use new function
        page_summary_docs.append(Document(
            page_content=summary,
            metadata={
                "file_name": file_name,
                "level": "page",
                "page_start": page_num,
                "page_end": page_num,
                "ts": ts
            }
        ))
    return page_summary_docs

def make_section_summaries1(page_summary_docs, summarizer, file_name, ts):
    section_docs = []
    for i in range(0, len(page_summary_docs), 5):
        block_summaries = page_summary_docs[i:i+5]
        page_start = block_summaries[0].metadata['page_start']
        page_end = block_summaries[-1].metadata['page_end']
        combined_content = "\n\n".join([doc.page_content for doc in block_summaries])
        summary = summarize_section(combined_content, summarizer) # Use new function
        section_docs.append(Document(
            page_content=summary,
            metadata={
                "file_name": file_name,
                "level": "page",
                "page_start": page_start,
                "page_end": page_end,
                "ts": ts
            }
        ))
    return section_docs

def make_section_summaries(page_summary_docs, summarizer, file_name, ts):
    section_docs = []
    for i in range(0, len(page_summary_docs), 10):
        block_summaries = page_summary_docs[i:i+10]
        page_start = block_summaries[0].metadata['page_start']
        page_end = block_summaries[-1].metadata['page_end']
        combined_content = "\n\n".join([doc.page_content for doc in block_summaries])
        summary = summarize_section(combined_content, summarizer) # Use new function
        section_docs.append(Document(
            page_content=summary,
            metadata={
                "file_name": file_name,
                "level": "section",
                "page_start": page_start,
                "page_end": page_end,
                "ts": ts
            }
        ))
    return section_docs

def make_book_summary(section_docs, page_summary_docs, summarizer, file_name, page_count, ts):
    if section_docs:
        all_section_content = "\n\n".join([doc.page_content for doc in section_docs])
        book_summary_text = summarize_book(all_section_content, summarizer) # Use new function
    else:  # fallback
        all_page_content = "\n\n".join([doc.page_content for doc in page_summary_docs[:50]])
        book_summary_text = summarize_book(all_page_content, summarizer) # Use new function

    return Document(
        page_content=book_summary_text,
        metadata={
            "file_name": file_name,
            "level": "book",
            "page_start": 1,
            "page_end": page_count,
            "ts": ts
        }
    )

# ========== MAIN PIPELINE (Remains the same) ==========
def build_hierarchy_and_index(page_docs: List[Document], file_name: str, language_hint="en"):
    summarizer = get_fast_summarizer() # This now uses a better model
    ts = datetime.utcnow().isoformat()

    chunk_docs = make_chunks(page_docs, file_name, ts)
    # print("Generating page summaries...")
    # page_summary_docs = make_page_summaries(page_docs, summarizer, file_name, ts)

    print("Generating page summaries...")
    page_summary_docs = make_section_summaries1(chunk_docs, summarizer, file_name, ts)

    print("Generating section summaries...")
    section_docs = make_section_summaries(page_summary_docs, summarizer, file_name, ts)

    print("Generating book summary...")
    book_doc = make_book_summary(section_docs, page_summary_docs, summarizer, file_name, len(page_docs), ts)

    all_docs = page_summary_docs + section_docs + [book_doc]

    print("Indexing documents...")
    collection_size = index_documents(all_docs, vs)

    print(f"Successfully indexed {len(all_docs)} documents for {file_name}")
    print(f"Total documents in collection: {collection_size}")

    return {
        "chunks": len(chunk_docs),
        "pages": len(page_summary_docs),
        "sections": len(section_docs),
        "book": 1,
        "total": len(all_docs),
        "collection_size": collection_size
    }
################################## End Summary with levels ##################################

@app.post("/build")
async def ask(file: File):
    docs = load_file(file.path)
    filename= name_file(file.path)
    ts = datetime.utcnow().isoformat()
    chunked_documents = make_chunks(docs, filename, ts)
    vs.add_documents(chunked_documents)

    # Run the summarization/indexing in a separate thread
    if file.flag:
        threading.Thread(
            target=build_hierarchy_and_index,
            args=(docs, filename),
            daemon=True  # so it won’t block app shutdown
        ).start()

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    answer = generate_answer(request.question, request.n_results)
    return AnswerResponse(answer=answer)

@app.post("/ask_level", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    answer = ask_with_level(request.question, cohere_llm, vs)
    return AnswerResponse(answer=answer)

@app.get("/summarize", response_model=SummaryResponse)
async def summarize_file():
    summary = summarize()
    enhanced_summary = beautify_answer(summary, cohere_llm)
    return SummaryResponse(summary=enhanced_summary)

@app.post("/translate", response_model=TranslateResponse)
async def summarize_file(summary : TranslateRequest):
    arabic_summary = translate_to_arabic(summary.content, cohere_llm)
    return TranslateResponse(translated_text=arabic_summary)