from langchain.prompts import PromptTemplate, ChatPromptTemplate
from dotenv import load_dotenv
import os
from langchain.schema import StrOutputParser
from langchain_cohere import ChatCohere
from utils.vectorstore import get_vs
import re
import json

load_dotenv()

cohere_key = os.getenv("COHERE_API_KEY")
if cohere_key:
    os.environ["CO_API_KEY"] = cohere_key
else:
    print("⚠️ Warning: COHERE_API_KEY not set, skipping Cohere integration.")

cohere_llm = ChatCohere(model="command-a-03-2025")

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

translate_prompt = PromptTemplate.from_template("""
    You are a professional translator.
    Translate the following text from English to **Arabic**, keeping the same meaning, tone, and style.
    If the text contains emojis, keep them.

    English Text:
    {raw_answer}

    Arabic Translation:
""")

def get_relevant_chunks_with_scores(question, level="chunk", n_results=5):
    vs = get_vs()
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
        # f"📄 **File:** {doc.metadata.get('file_name', 'Unknown')}  \n"
        # f"📑 **Page:** {doc.metadata.get('page_start', 'N/A')}  \n"
        # f"🔹 **Chunk ID:** {doc.metadata.get('chunk_id', 'N/A')}  \n\n"
        f"{doc.page_content}"
        for doc, score in sorted_results
    ]

def summarize(n_results=8):
    vs = get_vs()
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

def ask_with_level(query, llm):
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
