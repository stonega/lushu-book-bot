from fastapi import FastAPI
from dotenv import load_dotenv
from pydantic import BaseModel
from qa import ask_question

load_dotenv()

app = FastAPI()

class Item(BaseModel):
    question: str

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/ask")
def root(ask: Item):
    question = ask.question
    result = ask_question(question)
    return result
