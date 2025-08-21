from fastapi import APIRouter, Query
from data_models.file import File, SummaryResponse, TranslateRequest, TranslateResponse
from services.rag_pipeline import build_hierarchy_and_index
# from utils.vectorstore import vs
from utils.text_utils import clean_text
from datetime import datetime
import threading
from utils.summarizer import make_chunks
from utils.load_file import load_file
from utils.retreival import (summarize, beautify_answer, translate_to_arabic, cohere_llm)
from utils.pdf_utils import get_pdf_response
from utils.pdf_utils import create_pdf_wrapped, create_pdf_wrapped_ar
from utils.vectorstore import index_documents

router = APIRouter()

@router.post("/build")
async def build_index(file: File):
    docs = load_file(file.path)
    filename = file.path.split("/")[-1]
    ts = datetime.utcnow().isoformat()
    chunked_documents = make_chunks(docs, filename, ts)
    global vs
    vs = index_documents(chunked_documents)  # update reference

    if file.flag:
        threading.Thread(
            target=build_hierarchy_and_index,
            args=(docs, filename),
            daemon=True
        ).start()

    return {"status": "Building index started"}

@router.get("/summarize", response_model=SummaryResponse)
def summarize_file():
    summary_text = summarize()
    enhanced_summary = beautify_answer(summary_text, cohere_llm)
    pdf_filename = "summary_en.pdf"
    create_pdf_wrapped(enhanced_summary, pdf_filename)
    return SummaryResponse(summary=enhanced_summary)

@router.post("/translate")
def translate_summary(req: TranslateRequest):
    translated_text = translate_to_arabic(req.content, cohere_llm)
    pdf_filename = "summary_ar.pdf"
    create_pdf_wrapped_ar(translated_text, pdf_filename)
    return TranslateResponse(translated_text=translated_text)

@router.get("/download")
def download_summary(filename: str = Query(..., description="Name of the summary PDF to download")):
    """
    Download a summary PDF.
    Example: /download?filename=summary_en.pdf
    """
    return get_pdf_response(filename)
