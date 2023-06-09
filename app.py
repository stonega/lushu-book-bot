from fastapi import FastAPI
from pydantic import BaseModel
from ask import ask_question
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "https://lushu-book.vercel.app",
    "https://lushu-book.stonegate.me",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Item(BaseModel):
    question: str

@app.get("/")
def root():
    return {"message": "Hello World"}

@app.post("/ask")
def root(ask: Item):
    question = ask.question
    result = ask_question(question)
    return result
