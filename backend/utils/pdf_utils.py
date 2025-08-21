import os
from reportlab.lib.pagesizes import letter
from fastapi import HTTPException
from fastapi.responses import FileResponse
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from bidi.algorithm import get_display 
import arabic_reshaper


SUMMARY_DIR = "../summaries"
os.makedirs(SUMMARY_DIR, exist_ok=True)


FONT_PATH = os.path.join(os.path.dirname(__file__), "../../fonts/Amiri-Regular.ttf")
pdfmetrics.registerFont(TTFont('Amiri', FONT_PATH))

def create_pdf_wrapped_ar(text: str, filename: str) -> str:
    """
    Create a PDF from Arabic text with proper line wrapping and page breaks.
    Overwrites existing file if present.
    """
    filepath = os.path.join(SUMMARY_DIR, filename)
    
    if os.path.exists(filepath):
        os.remove(filepath)
    
    doc = SimpleDocTemplate(filepath, pagesize=letter,
                            rightMargin=50, leftMargin=50,
                            topMargin=50, bottomMargin=50)
    
    styles = getSampleStyleSheet()
    style = ParagraphStyle(
        name='Normal',
        parent=styles['Normal'],
        fontName='Amiri',  # استخدم الخط المسجل
        fontSize=12,
        leading=15,
        alignment=2,
    )
    
    # Reshape and reorder Arabic text for proper display
    def prepare_arabic_paragraph(p):
        reshaped_text = arabic_reshaper.reshape(p)
        bidi_text = get_display(reshaped_text)
        return Paragraph(bidi_text.replace('\n', '<br/>'), style)
    
    # Split text into paragraphs by double newlines
    paragraphs = [prepare_arabic_paragraph(p) for p in text.split('\n\n')]
    
    flowables = []
    for p in paragraphs:
        flowables.append(p)
        flowables.append(Spacer(1, 0.2*inch))
    
    doc.build(flowables)
    return filepath

def create_pdf_wrapped(text: str, filename: str) -> str:
    """
    Create a PDF from text with proper line wrapping and page breaks.
    Overwrites existing file.
    """
    filepath = os.path.join(SUMMARY_DIR, filename)
    
    if os.path.exists(filepath):
        os.remove(filepath)
    
    doc = SimpleDocTemplate(filepath, pagesize=letter,
                            rightMargin=50, leftMargin=50,
                            topMargin=50, bottomMargin=50)
    
    styles = getSampleStyleSheet()
    style = ParagraphStyle(
        name='Normal',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=15
    )
    
    # Split text into paragraphs based on double newlines
    paragraphs = [Paragraph(p.replace('\n', '<br/>'), style) for p in text.split('\n\n')]
    
    flowables = []
    for p in paragraphs:
        flowables.append(p)
        flowables.append(Spacer(1, 0.2*inch))
    
    doc.build(flowables)
    return filepath

def get_pdf_response(filename: str) -> FileResponse:
    """Return a FileResponse for an existing PDF in SUMMARY_DIR"""
    filepath = os.path.join(SUMMARY_DIR, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"{filename} not found")
    return FileResponse(filepath, media_type="application/pdf", filename=filename)
