from langchain_community.document_loaders import PyPDFLoader

file_path = "document/preview-9781449363901_A24457138.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()
docs[0]