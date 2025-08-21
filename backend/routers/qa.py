from fastapi import APIRouter
from data_models.file import AnswerResponse, QuestionRequest
from utils.vectorstore import get_vs
from utils.retreival import (ask_with_level , generate_answer, cohere_llm)

router = APIRouter()

@router.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    answer = generate_answer(request.question, request.n_results)
    return AnswerResponse(answer=answer)

@router.post("/ask_level", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    vs = get_vs()
    answer = ask_with_level(request.question, cohere_llm, vs)
    return AnswerResponse(answer=answer)
