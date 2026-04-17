import os
import faiss
import json
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

INDEX_FILE = "manual_index.faiss"
METADATA_FILE = "manual_metadata.json"

def test_retrieval(query):
    print(f"Testing RAG for query: {query}")
    
    # Load index and metadata
    index = faiss.read_index(INDEX_FILE)
    with open(METADATA_FILE, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    # Embedding query
    res = client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    )
    query_vector = np.array([res.data[0].embedding]).astype('float32')
    
    # Search
    D, I = index.search(query_vector, 3)
    
    print("\n--- Retrieval Results ---")
    for i, idx in enumerate(I[0]):
        if idx < len(chunks):
            print(f"Result {i+1}:")
            print(chunks[idx][:300] + "...")
            print("-" * 30)

if __name__ == "__main__":
    test_retrieval("행동 활성화(BA)의 핵심 원리")
    print("\n\n")
    test_retrieval("인지행동치료(CBT)의 자동적 사고")
