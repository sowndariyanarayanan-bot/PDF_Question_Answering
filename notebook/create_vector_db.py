
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

print("Loading chunks...")

df = pd.read_csv("dataset/chunks.csv")
texts = df["Text"].dropna().astype(str).tolist()

if not texts:
    raise ValueError("No chunks found!")

print("Loading Embedding Model...")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Creating ChromaDB...")

vector_db = Chroma.from_texts(
    texts=texts,
    embedding=embedding_model,
    collection_name="pdf_documents",
    persist_directory="chroma_db"
)

print("Embeddings Generated Successfully!")
print("Total Documents Stored:", len(texts))
print("Stored in ChromaDB!")

