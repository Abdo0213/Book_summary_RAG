# routers/auth.py
from fastapi import APIRouter, HTTPException
from data_models.user import User, LoginRequest
from db.conn import get_conn
from passlib.context import CryptContext
import sqlite3

router = APIRouter()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

@router.post("/register")
def register(user: User):
    conn = get_conn()
    cur = conn.cursor()
    hashed_pw = pwd_context.hash(user.password)
    try:
        cur.execute(
            "INSERT INTO users (username, email, password, categories) VALUES (?, ?, ?, ?)",
            (user.username, user.email, hashed_pw, user.categories)
        )
        conn.commit()
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Username or email already exists")
    finally:
        conn.close()
    return {"msg": "User registered"}

@router.post("/login")
def login(user: LoginRequest):
    conn = get_conn()
    cur = conn.cursor()
    # fetch id, username, password, and categories
    cur.execute("SELECT id, username, password, categories FROM users WHERE username=?", (user.username,))
    row = cur.fetchone()
    conn.close()
    if not row or not pwd_context.verify(user.password, row[2]):  # row[2] = password
        raise HTTPException(status_code=400, detail="Invalid credentials")
    return {
        "msg": "Login successful",
        "user_id": row[0],        # id
        "username": row[1],       # username
        "categories": row[3]      # categories (string or JSON depending on your schema)
    }
