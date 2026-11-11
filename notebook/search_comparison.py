
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_huggingface import HuggingFaceEmbeddings

# =====================================
# 1. LOAD DOCUMENT CHUNKS
# =====================================

print("Loading document chunks...")

df = pd.read_csv("dataset/chunks.csv")
chunks = df["Text"].dropna().astype(str).tolist()

if not chunks:
    raise ValueError("No document chunks found!")

# =====================================
# 2. PREPARE 15 USER QUESTIONS
# =====================================

questions = [
    "What is artificial intelligence?",
    "What are the applications of artificial intelligence?",
    "Explain machine learning.",
    "What is deep learning?",
    "How does AI work?",
    "What are the benefits of artificial intelligence?",
    "What are the challenges of AI?",
    "Explain neural networks.",
    "What is natural language processing?",
    "How is AI used in healthcare?",
    "What is computer vision?",
    "What is supervised learning?",
    "What is unsupervised learning?",
    "How does AI support decision making?",
    "What is the future of artificial intelligence?"
]

query_df = pd.DataFrame({
    "Query_ID": range(1, 16),
    "Query": questions
})

query_df.to_csv("dataset/queries.csv", index=False)

print("15 Questions Created!")

# =====================================
# 3. KEYWORD SEARCH USING TF-IDF
# =====================================

print("\nPerforming Keyword Search...")

tfidf = TfidfVectorizer(stop_words="english")

chunk_vectors = tfidf.fit_transform(chunks)

# =====================================
# 4. LOAD SEMANTIC EMBEDDING MODEL
# =====================================

print("\nLoading Semantic Embedding Model...")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Generate document embeddings
document_vectors = embedding_model.embed_documents(chunks)

# =====================================
# 5. PERFORM BOTH SEARCHES
# =====================================

all_results = []

for _, row in query_df.iterrows():

    query_id = row["Query_ID"]
    query = row["Query"]

    print("\nProcessing:", query)

    # ---------------------------------
    # KEYWORD SEARCH
    # ---------------------------------

    query_vector = tfidf.transform([query])

    keyword_scores = cosine_similarity(
        query_vector, chunk_vectors
    )[0]

    keyword_top5 = keyword_scores.argsort()[-5:][::-1]

    for rank, index in enumerate(keyword_top5, start=1):

        all_results.append({
            "Query_ID": query_id,
            "Query": query,
            "Search_Type": "Keyword",
            "Rank": rank,
            "Chunk_ID": int(df.iloc[index]["Chunk_ID"]),
            "Document": chunks[index],
            "Similarity_Score": round(
                float(keyword_scores[index]), 4
            )
        })

    # ---------------------------------
    # SEMANTIC SEARCH
    # ---------------------------------

    query_embedding = embedding_model.embed_query(query)

    semantic_scores = cosine_similarity(
        [query_embedding],
        document_vectors
    )[0]

    semantic_top5 = semantic_scores.argsort()[-5:][::-1]

    for rank, index in enumerate(semantic_top5, start=1):

        all_results.append({
            "Query_ID": query_id,
            "Query": query,
            "Search_Type": "Semantic",
            "Rank": rank,
            "Chunk_ID": int(df.iloc[index]["Chunk_ID"]),
            "Document": chunks[index],
            "Similarity_Score": round(
                float(semantic_scores[index]), 4
            )
        })

    print("Keyword Top-5:", len(keyword_top5))
    print("Semantic Top-5:", len(semantic_top5))

# =====================================
# 6. SAVE SEARCH RESULTS
# =====================================

results_df = pd.DataFrame(all_results)

results_df.to_csv(
    "dataset/search_results.csv",
    index=False,
    encoding="utf-8"
)

print("\n================================")
print("SEARCH COMPARISON COMPLETED!")
print("================================")

print("Total Questions:", len(query_df))
print("Total Search Results:", len(results_df))
print("Results saved in search_results.csv")




