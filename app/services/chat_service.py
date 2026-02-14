from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from app.config import MODEL_PATH, RETRIEVAL_K

llm = LlamaCpp(
    model_path=MODEL_PATH,
    temperature=0.1,
    max_tokens=256,
    n_ctx=2048,
    n_gpu_layers=20,
    n_batch=128,
    streaming=True,
    verbose=False
)


prompt_template = """
You are an academic assistant.

Use ONLY the provided context to answer.
If the answer is not in the document, say:
"The answer is not found in the document."

Context:
{context}

Question:
{question}

Answer clearly:
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

def get_qa_chain(vectorstore):
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K}),
        chain_type_kwargs={"prompt": PROMPT}
    )
