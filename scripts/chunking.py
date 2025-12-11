from langchain_text_splitters import RecursiveCharacterTextSplitter
from document_loader import docs

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(docs)
print(f"Orignal Documents: {(len(docs))}")
print(f"Number of Chunks: {(len(chunks))}")
print(f"\n First Chunk preview:\n {(chunks[0].page_content[:200])}...")