from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from app.config import CHROMA_PATH

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5"
)

def create_vector_store(docs):
    vectorstore = Chroma.from_documents(
        docs,
        embedding_model,
        persist_directory=CHROMA_PATH
    )
    vectorstore.persist()
    return vectorstore

def load_vector_store():
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_model
    )
