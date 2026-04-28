import os
import re
from fastapi import FastAPI, UploadFile, File, HTTPException
from app.services.pdf_service import process_pdf
from app.services.vector_service import create_vector_store, load_vector_store
from app.services.chat_service import get_qa_chain

app = FastAPI()

vectorstore = None


def _fallback_answer_from_docs(query: str):
    """
    If local Llama GGUF model is missing, answer from retrieved document chunks
    directly so PDF Q&A can still work.
    """
    global vectorstore
    if vectorstore is None:
        return "The answer is not found in the document."

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(query)
    if not docs:
        return "The answer is not found in the document."

    q_terms = set(re.findall(r"\w+", query.lower()))
    best_text = ""
    best_score = -1
    for d in docs:
        text = (d.page_content or "").strip()
        if not text:
            continue
        lower = text.lower()
        score = sum(1 for t in q_terms if t in lower)
        if score > best_score:
            best_score = score
            best_text = text

    if not best_text:
        return "The answer is not found in the document."

    # Return first 2 sentences max to keep concise.
    parts = re.split(r"(?<=[.?!])\s+", best_text)
    snippet = " ".join(parts[:2]).strip()
    if not snippet:
        snippet = best_text[:500].strip()
    return snippet

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global vectorstore

    filename = file.filename or "document.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    try:
        docs = process_pdf(file_path)
        vectorstore = create_vector_store(docs)
    except Exception as e:
        # Most common issue: invalid/corrupt PDF content.
        raise HTTPException(status_code=400, detail=f"Failed to process PDF: {e}")
    finally:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            pass

    return {
        "message": "PDF processed successfully",
        "chunks": len(docs)
    }


@app.post("/chat")
async def chat(query: str):
    global vectorstore

    if not vectorstore:
        vectorstore = load_vector_store()

    try:
        qa_chain = get_qa_chain(vectorstore)
    except FileNotFoundError as e:
        response = _fallback_answer_from_docs(query)
        return {
            "answer": response,
            "mode": "retrieval_fallback",
            "warning": str(e),
        }
    response = qa_chain.run(query)

    return {"answer": response}
