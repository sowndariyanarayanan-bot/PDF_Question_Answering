
import os
import pandas as pd
from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 1. PROJECT PATHS
# ==========================================

DATASET = Path("dataset")
PDF_PATH = DATASET / "sample.pdf"
TEXT_PATH = DATASET / "extracted_text.txt"
CHUNKS_PATH = DATASET / "chunks.csv"
QUERIES_PATH = DATASET / "queries.csv"
RESULTS_PATH = DATASET / "search_results.csv"
ANSWERS_PATH = DATASET / "answers.csv"

CHROMA_PATH = "chroma_db"

DATASET.mkdir(exist_ok=True)

# ==========================================
# 2. PDF TEXT EXTRACTION
# ==========================================

print("\n===== PDF TEXT EXTRACTION =====")

if not PDF_PATH.exists():
    raise FileNotFoundError(
        "sample.pdf not found inside dataset folder!"
    )

reader = PdfReader(str(PDF_PATH))

extracted_text = ""

for page_number, page in enumerate(reader.pages, start=1):
    page_text = page.extract_text()

    if page_text:
        extracted_text += (
            f"\nPage {page_number}\n{page_text}\n"
        )

if not extracted_text.strip():
    raise ValueError("No readable text found in PDF!")

TEXT_PATH.write_text(extracted_text, encoding="utf-8")

print("Total Pages:", len(reader.pages))
print("PDF text extracted successfully!")

# ==========================================
# 3. TEXT CHUNKING
# ==========================================

print("\n===== TEXT CHUNKING =====")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_text(extracted_text)

if not chunks:
    raise ValueError("No text chunks generated!")

chunks_df = pd.DataFrame({
    "Chunk_ID": range(1, len(chunks) + 1),
    "Text": chunks
})

chunks_df.to_csv(CHUNKS_PATH, index=False)

print("Total Chunks:", len(chunks))
print("chunks.csv created!")

# ==========================================
# 4. LOAD EMBEDDING MODEL
# ==========================================

print("\n===== LOADING EMBEDDING MODEL =====")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded!")

# ==========================================
# 5. STORE VECTORS IN CHROMADB
# ==========================================

print("\n===== CREATING CHROMADB =====")

# Rebuild collection to avoid duplicate old data
vector_db = Chroma(
    collection_name="pdf_documents",
    embedding_function=embeddings,
    persist_directory=CHROMA_PATH
)

vector_db.delete_collection()

vector_db = Chroma(
    collection_name="pdf_documents",
    embedding_function=embeddings,
    persist_directory=CHROMA_PATH
)

ids = [f"chunk-{i}" for i in range(1, len(chunks) + 1)]

metadatas = [
    {"chunk_id": i}
    for i in range(1, len(chunks) + 1)
]

vector_db.add_texts(
    texts=chunks,
    metadatas=metadatas,
    ids=ids
)

print("Vectors stored successfully!")
print("Documents in ChromaDB:", len(chunks))

# ==========================================
# 6. CREATE SAMPLE QUERIES
# ==========================================

print("\n===== PREPARING QUERIES =====")

if not QUERIES_PATH.exists():
    queries = [
        "What is artificial intelligence?",
        "What are the applications of artificial intelligence?",
        "Explain machine learning.",
        "What is deep learning?",
        "How is artificial intelligence used in healthcare?"
    ]

    pd.DataFrame({
        "Query_ID": range(1, len(queries) + 1),
        "Query": queries
    }).to_csv(QUERIES_PATH, index=False)

query_df = pd.read_csv(QUERIES_PATH)

# ==========================================
# 7. KEYWORD AND SEMANTIC SEARCH
# ==========================================

print("\n===== SEARCH COMPARISON =====")

# Keyword search using TF-IDF
tfidf = TfidfVectorizer(
    stop_words="english"
)

chunk_vectors = tfidf.fit_transform(chunks)

search_results = []

for _, row in query_df.iterrows():

    query_id = row["Query_ID"]
    query = str(row["Query"])

    print("\nQuery:", query)

    # ----- KEYWORD SEARCH -----

    query_vector = tfidf.transform([query])

    keyword_scores = cosine_similarity(
        query_vector,
        chunk_vectors
    )[0]

    keyword_top5 = keyword_scores.argsort()[-5:][::-1]

    for rank, index in enumerate(keyword_top5, start=1):
        search_results.append({
            "Query_ID": query_id,
            "Query": query,
            "Search_Type": "Keyword",
            "Rank": rank,
            "Chunk_ID": index + 1,
            "Document": chunks[index],
            "Similarity_Score": round(
                float(keyword_scores[index]), 4
            )
        })

    # ----- SEMANTIC SEARCH -----

    retrieved = vector_db.similarity_search(
        query,
        k=min(5, len(chunks))
    )

    query_embedding = embeddings.embed_query(query)

    retrieved_texts = [doc.page_content for doc in retrieved]
    retrieved_embeddings = embeddings.embed_documents(
        retrieved_texts
    )

    semantic_scores = cosine_similarity(
        [query_embedding],
        retrieved_embeddings
    )[0]

    for rank, (doc, score) in enumerate(
        zip(retrieved, semantic_scores), start=1
    ):
        search_results.append({
            "Query_ID": query_id,
            "Query": query,
            "Search_Type": "Semantic",
            "Rank": rank,
            "Chunk_ID": doc.metadata.get("chunk_id"),
            "Document": doc.page_content,
            "Similarity_Score": round(float(score), 4)
        })

    print("Keyword Top-5:", len(keyword_top5))
    print("Semantic Top-5:", len(retrieved))

# Save search results
results_df = pd.DataFrame(search_results)

results_df.to_csv(
    RESULTS_PATH,
    index=False,
    encoding="utf-8"
)

print("\nSearch results saved successfully!")

# ==========================================
# 8. PDF QUESTION ANSWERING USING OLLAMA
# ==========================================

print("\n===== PDF QUESTION ANSWERING =====")

answers = []

try:
    llm = ChatOllama(
        model="gemma3:1b",
        temperature=0
    )

    for _, row in query_df.iterrows():

        query = str(row["Query"])

        relevant_docs = vector_db.similarity_search(
            query,
            k=5
        )

        context = "\n\n".join(
            [doc.page_content for doc in relevant_docs]
        )

        prompt = f"""
You are a PDF Question Answering Assistant.

Answer the question using ONLY the given context.
If the answer is not available in the context,
say "Information not found in the PDF."

Context:
{context}

Question:
{query}

Answer:
"""

        response = llm.invoke(prompt)

        answer = response.content

        answers.append({
            "Query": query,
            "Answer": answer
        })

        print("\nQuestion:", query)
        print("Answer:", answer)

except Exception as error:

    print("\nOllama error:", error)
    print("Check Ollama installation and model availability.")

    for _, row in query_df.iterrows():
        answers.append({
            "Query": str(row["Query"]),
            "Answer": "Not generated - Check Ollama setup."
        })

# Save answers
pd.DataFrame(answers).to_csv(
    ANSWERS_PATH,
    index=False,
    encoding="utf-8"
)

# ==========================================
# 9. FINAL SUMMARY
# ==========================================

print("\n====================================")
print(" PDF QUESTION ANSWERING PROJECT")
print(" COMPLETED")
print("====================================")

print("Extracted Text :", TEXT_PATH)
print("Chunks         :", CHUNKS_PATH)
print("Search Results :", RESULTS_PATH)
print("Answers        :", ANSWERS_PATH)
print("Vector Database:", CHROMA_PATH)

