from typing import List
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langchain.docstore.document import Document
from langchain.schema import Document
from utils.load_file import text_splitter
from langchain.prompts import ChatPromptTemplate

model_name = "google/flan-t5-base"  # or your model
llm_save_path = "./models/flan-t5-base"  # local folder
# Download and save
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer.save_pretrained(llm_save_path)
print(f"✅ Model saved to {llm_save_path}")

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
