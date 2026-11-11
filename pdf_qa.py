
from pathlib import Path
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Step 1: Read extracted text
text_path = Path("dataset/extracted_text.txt")

if not text_path.exists():
    raise FileNotFoundError("extracted_text.txt not found. Run PDF extraction first.")

with open(text_path, "r", encoding="utf-8") as file:
    text = file.read()

# Step 2: Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_text(text)

# Step 3: Store chunks in CSV
df = pd.DataFrame({
    "Chunk_ID": range(1, len(chunks) + 1),
    "Text": chunks
})

df.to_csv("dataset/chunks.csv", index=False, encoding="utf-8")

# Step 4: Display results
print("Text Chunking Completed!")
print("Total Chunks:", len(chunks))
print("\nFirst Chunk:")
print(chunks[0] if chunks else "No text found")
print("\nChunks saved to dataset/chunks.csv")
