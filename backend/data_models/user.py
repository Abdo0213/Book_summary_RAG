from pydantic import BaseModel

class User(BaseModel):
    username: str
    email: str
    password: str
    categories: str

class LoginRequest(BaseModel):
    username: str
    password: str
