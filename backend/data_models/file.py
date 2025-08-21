from pydantic import BaseModel

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

class TranslateResponse(BaseModel):
    translated_text: str
