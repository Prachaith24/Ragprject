from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from chunking import chunks
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)
print(f"stored{len(chunks)}chunks in chorma_DB")