from fastapi import FastAPI 
from pydantic import BaseModel
from app.rag import prepare_data, get_qa_chain
from fastapi.middleware.cors import CORSMiddleware 

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str

vectorstore = prepare_data()
qa_chain = get_qa_chain(vectorstore)

@app.post("/ask")
def ask_question(query: Query):
    answer = qa_chain.run(query.question)
    return {"answer": answer}
