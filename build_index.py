import os
import json
import numpy as np
import faiss
from pypdf import PdfReader
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

INDEX_FILE = "manual_index.faiss"
METADATA_FILE = "manual_metadata.json"
PDF_PATH = "우울증을_위한_행동활성화_프로그램_매뉴얼_치료자용.PDF"

def extract_text_from_pdf(pdf_path):
    print(f"📄 '{pdf_path}'에서 텍스트 추출 중...")
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content + "\n"
    return text

def chunk_text(text, chunk_size=600, overlap=100):
    print(f"✂️ 텍스트 청킹 중 (사이즈: {chunk_size}, 오버랩: {overlap})...")
    # 특수문자 및 불필요한 공백 제거
    text = " ".join(text.split())
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= len(text):
            break
        start += (chunk_size - overlap)
        
    return [c for c in chunks if len(c.strip()) > 30] 

def get_embeddings(texts):
    print(f"🧠 {len(texts)}개 청크 임베딩 생성 중...")
    embeddings = []
    # 20개씩 배치 처리하여 안정성 확보
    for i in range(0, len(texts), 20):
        batch = texts[i:i+20]
        try:
            res = client.embeddings.create(
                input=batch,
                model="text-embedding-3-small"
            )
            embeddings.extend([e.embedding for e in res.data])
        except Exception as e:
            print(f"⚠️ 배치 {i} 임베딩 생성 오류: {e}")
            # 오류 발생 시 해당 배치는 기본 벡터로 채우거나 스킵 (여기서는 전체 중단을 위해 다시 raise)
            raise e
    return np.array(embeddings).astype('float32')

def build_faiss_index(embeddings, chunks):
    print(f"📈 FAISS 인덱스 구축 중...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    faiss.write_index(index, INDEX_FILE)
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"✅ 구축 완료: {INDEX_FILE}, {METADATA_FILE} 저장됨.")

if __name__ == "__main__":
    if not os.path.exists(PDF_PATH):
        print(f"❌ PDF 파일을 찾을 수 없습니다: {PDF_PATH}")
    else:
        full_text = extract_text_from_pdf(PDF_PATH)
        chunks = chunk_text(full_text)
        if chunks:
            embeddings = get_embeddings(chunks)
            build_faiss_index(embeddings, chunks)
        else:
            print("❌ 추출된 텍스트 청크가 없습니다.")
