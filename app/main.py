import os
from fastapi import FastAPI, UploadFile, File
from app.services.pdf_service import process_pdf
from app.services.vector_service import create_vector_store, load_vector_store
from app.services.chat_service import get_qa_chain

app = FastAPI()

vectorstore = None

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global vectorstore

    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    docs = process_pdf(file_path)
    vectorstore = create_vector_store(docs)

    return {
        "message": "PDF processed successfully",
        "chunks": len(docs)
    }


@app.post("/chat")
async def chat(query: str):
    global vectorstore

    if not vectorstore:
        vectorstore = load_vector_store()

    qa_chain = get_qa_chain(vectorstore)
    response = qa_chain.run(query)

    return {"answer": response}
