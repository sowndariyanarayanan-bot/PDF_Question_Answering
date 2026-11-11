
import pandas as pd
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama

# =====================================
# 1. LOAD EMBEDDING MODEL
# =====================================

print("Loading Embedding Model...")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# =====================================
# 2. LOAD CHROMADB
# =====================================

print("Connecting to ChromaDB...")

vector_db = Chroma(
    collection_name="pdf_documents",
    embedding_function=embedding_model,
    persist_directory="chroma_db"
)

print("ChromaDB Connected Successfully!")

# =====================================
# 3. LOAD OLLAMA MODEL
# =====================================

print("Loading Ollama LLM...")

llm = ChatOllama(
    model="gemma3:1b",
    temperature=0
)

print("Ollama Model Ready!")

# =====================================
# 4. LOAD USER QUESTIONS
# =====================================

query_path = Path("dataset/queries.csv")

queries_df = pd.read_csv(query_path)

answers = []

# =====================================
# 5. RETRIEVE AND GENERATE ANSWERS
# =====================================

print("\nPDF QUESTION ANSWERING STARTED!")

for _, row in queries_df.iterrows():

    query = str(row["Query"])

    print("\nQuestion:", query)

    # Retrieve Top-5 relevant chunks
    documents = vector_db.similarity_search(
        query,
        k=5
    )

    context = "\n\n".join(
        [doc.page_content for doc in documents]
    )

    # Construct RAG prompt
    prompt = f"""
You are a PDF Question Answering Assistant.

Answer the question using only the information
provided in the context.

If the answer is not available, say:
Information not found in the PDF.

Context:
{context}

Question:
{query}

Answer:
"""

    # Generate answer using Ollama
    response = llm.invoke(prompt)

    answer = response.content

    print("Answer:", answer)

    # Save result
    answers.append({
        "Query": query,
        "Retrieved_Chunks": len(documents),
        "Answer": answer
    })

# =====================================
# 6. SAVE ANSWERS INTO CSV
# =====================================

answers_df = pd.DataFrame(answers)

answers_df.to_csv(
    "dataset/answers.csv",
    index=False,
    encoding="utf-8"
)

print("\n================================")
print("PDF QUESTION ANSWERING COMPLETED!")
print("================================")

print("Total Questions:", len(answers))
print("Answers saved in dataset/answers.csv")
