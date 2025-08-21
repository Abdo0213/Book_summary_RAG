from fastapi import FastAPI
from routers import auth, books, qa
from db.conn import get_conn

app = FastAPI(title="Book RAG API")

# Routers
app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
app.include_router(books.router, prefix="/books", tags=["Books"])
app.include_router(qa.router, prefix="/qa", tags=["Q&A"])
