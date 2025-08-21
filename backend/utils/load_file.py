from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredEPubLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from typing import List

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=900,
    chunk_overlap=100,
    length_function=len,
)

#Initialize Load function
def load_file(path: str) -> List[Document]:
    ext = path.lower().split(".")[-1]
    if ext == "pdf":
        loader = PyPDFLoader(path)
        docs = loader.load()  # returns per-page docs
    elif ext in ("txt", "md"):
        loader = TextLoader(path, encoding="utf-8")
        docs = loader.load()
    elif ext in ("epub",):
        loader = UnstructuredEPubLoader(path)
        docs = loader.load()
    else:
        raise ValueError(f"Unsupported file type: .{ext}")
    return docs
