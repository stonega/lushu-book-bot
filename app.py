from fastapi import FastAPI
from pydantic import BaseModel
from qa import ask_question
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://lushu-book.vercel.app",
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
