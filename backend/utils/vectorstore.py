from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

CHROMA_DIR = "./chroma_books"
COLLECTION_NAME = "books_rag"

embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
save_path_embedding = "./models/all-MiniLM-L6-v2"
# Download and save
emb = SentenceTransformer(embedding_model)
emb.save(save_path_embedding)
print(f"✅ Embedding model saved at {save_path_embedding}")

embeddings = HuggingFaceEmbeddings(model_name="./models/all-MiniLM-L6-v2")

# vs = Chroma(
#     collection_name=COLLECTION_NAME,
#     embedding_function=embeddings,
#     persist_directory=CHROMA_DIR
# )
# vectorstore.py
_vs = None

def get_vs():
    global _vs
    if _vs is None:
        _vs = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=CHROMA_DIR,
        )
    return _vs

def index_documents(all_docs):
    global _vs
    vs = get_vs()
    vs.delete_collection()
    _vs = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    _vs.add_documents(all_docs)
    _vs.persist()
    return _vs


# def index_documents(all_docs):
#     # Drop + recreate collection
#     vs = Chroma(
#         collection_name=COLLECTION_NAME,
#         embedding_function=embeddings,
#         persist_directory=CHROMA_DIR,
#     )
#     vs.delete_collection()

#     vs = Chroma(
#         collection_name=COLLECTION_NAME,
#         embedding_function=embeddings,
#         persist_directory=CHROMA_DIR,
#     )
#     vs.add_documents(all_docs)
#     vs.persist()
#     return vs   # return the new instance
