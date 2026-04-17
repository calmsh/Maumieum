
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

def search_manual(query, top_k=5):
    if not os.path.exists(INDEX_FILE):
        return "Index not found"
    
    index = faiss.read_index(INDEX_FILE)
    with open(METADATA_FILE, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
        
    emb_res = client.embeddings.create(input=query, model="text-embedding-3-small")
    query_vector = np.array([emb_res.data[0].embedding]).astype('float32')
    
    D, I = index.search(query_vector, top_k)
    results = [chunks[idx] for idx in I[0] if idx < len(chunks)]
    return results

if __name__ == "__main__":
    query = "인지재구조화 자동적 사고 인지적 오류 유형 종류 대응 방법"
    results = search_manual(query)
    print("--- RAG Search Results ---")
    for i, res in enumerate(results):
        print(f"\n[Result {i+1}]: {res}")
