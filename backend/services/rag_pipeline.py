from utils.vectorstore import index_documents
from utils.summarizer import (
    get_fast_summarizer, make_chunks, make_section_summaries,
    make_book_summary, make_section_summaries1
)
from datetime import datetime

def build_hierarchy_and_index(page_docs, file_name, language_hint="en"):
    summarizer = get_fast_summarizer()
    ts = datetime.utcnow().isoformat()

    chunk_docs = make_chunks(page_docs, file_name, ts)
    page_summary_docs = make_section_summaries1(chunk_docs, summarizer, file_name, ts)
    section_docs = make_section_summaries(page_summary_docs, summarizer, file_name, ts)
    book_doc = make_book_summary(section_docs, page_summary_docs, summarizer, file_name, len(page_docs), ts)

    all_docs = page_summary_docs + section_docs + [book_doc]
    collection_size = index_documents(all_docs)

    return {
        "chunks": len(chunk_docs),
        "pages": len(page_summary_docs),
        "sections": len(section_docs),
        "book": 1,
        "total": len(all_docs),
        "collection_size": collection_size
    }
