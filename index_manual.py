import os
import csv
import fitz  # PyMuPDF
import json
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import faiss

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

INDEX_FILE = "manual_index.faiss"
METADATA_FILE = "manual_metadata.json"

# ── 인덱싱할 PDF 목록 ─────────────────────────────────────────
PDF_FILES = [
    {
        "path": "우울증을_위한_행동활성화_프로그램_매뉴얼_치료자용.PDF",
        "label": "BA매뉴얼(치료자용)",
    },
    {
        "path": "우울증을_위한_행동활성화_프로그램_워크북_참가자용 (1).PDF",
        "label": "BA워크북(참가자용)",
    },
    # ── CBT 관련 PDF ──
    {
        "path": "CBT/CBT-D_Manual_Depression.pdf",
        "label": "CBT-D매뉴얼(우울증)",
    },
    {
        "path": "CBT/Coping-with-Depression.pdf",
        "label": "CBT우울극복가이드",
    },
    {
        "path": "CBT/IJPsy-62-223.pdf",
        "label": "CBT학술논문",
    },
]

# ── JSONL 파일 목록 (학술 논문 코퍼스) ───────────────────────
JSONL_FILES = [
    {
        "path": "CBT/cbt_ba_corpus_final.jsonl",
        "label": "CBT/BA논문코퍼스",
    },
]

# ── CSV 파일 목록 (동기면담 대화 데이터) ─────────────────────
# AnnoMI: 실제 MI 상담 대화 전사본 (치료사 발화만 고품질로 추출)
CSV_FILES = [
    {
        "path": "CBT/AnnoMI-full.csv",
        "label": "동기면담(AnnoMI)",
        "text_col": "utterance_text",
        "filter_col": "mi_quality",      # 'high' 품질만 사용
        "filter_val": "high",
        "role_col": "interlocutor",      # 'therapist' 발화만 사용
        "role_val": "therapist",
        "group_col": "transcript_id",    # 같은 세션 발화를 그룹핑
    },
]

CHUNK_SIZE = 400   # 청크 하나의 글자 수
OVERLAP    = 100   # 청크 간 중첩 글자 수


def chunk_text(text: str, label: str, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    """텍스트를 청크로 분할하고 출처 레이블을 붙입니다."""
    text = " ".join(text.split())
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if len(chunk) > 30:
            chunks.append(f"[출처: {label}] {chunk}")
        if end >= len(text):
            break
        start += (chunk_size - overlap)
    return chunks


def extract_text_from_pdf(pdf_path: str, label: str):
    """PDF에서 텍스트를 추출하고 청크로 분할합니다."""
    print(f"  📖 텍스트 추출 중: {pdf_path}")
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text() + "\n"
    doc.close()
    chunks = chunk_text(full_text, label)
    print(f"  ✅ {len(chunks)}개 청크 생성")
    return chunks


def extract_from_jsonl(jsonl_path: str, label: str):
    """JSONL 학술 논문 코퍼스에서 텍스트를 추출합니다.
    각 논문의 title + clean_content를 합쳐 청크화합니다."""
    print(f"  📖 JSONL 추출 중: {jsonl_path}")
    chunks = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
                title = doc.get("title", "")
                pages = doc.get("pages", [])
                for page in pages:
                    content = page.get("clean_content", "")
                    if content:
                        full = f"{title}. {content}" if title else content
                        chunks.extend(chunk_text(full, label))
            except json.JSONDecodeError:
                continue
    print(f"  ✅ {len(chunks)}개 청크 생성")
    return chunks


def extract_from_csv(csv_info: dict):
    """CSV 동기면담 대화에서 치료사 발화를 세션별로 그룹핑해 추출합니다."""
    path      = csv_info["path"]
    label     = csv_info["label"]
    text_col  = csv_info["text_col"]
    filter_col = csv_info.get("filter_col")
    filter_val = csv_info.get("filter_val")
    role_col  = csv_info.get("role_col")
    role_val  = csv_info.get("role_val")
    group_col = csv_info.get("group_col")

    print(f"  📖 CSV 추출 중: {path}")
    session_texts = {}

    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 품질 필터
            if filter_col and row.get(filter_col) != filter_val:
                continue
            # 역할 필터 (치료사 발화만)
            if role_col and row.get(role_col) != role_val:
                continue
            text = row.get(text_col, "").strip()
            if not text or len(text) < 20:
                continue

            group_key = row.get(group_col, "default") if group_col else "default"
            if group_key not in session_texts:
                session_texts[group_key] = []
            session_texts[group_key].append(text)

    # 세션별 발화를 합쳐 청크화
    chunks = []
    for _, utterances in session_texts.items():
        combined = " ".join(utterances)
        chunks.extend(chunk_text(combined, label))

    print(f"  ✅ {len(chunks)}개 청크 생성 ({len(session_texts)}개 세션)")
    return chunks


def get_embeddings(texts: list) -> np.ndarray:
    """청크 리스트를 배치로 임베딩합니다."""
    print(f"  🧬 임베딩 생성 중 (총 {len(texts)}개)...")
    embeddings = []
    batch_size = 50
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(
            input=batch,
            model="text-embedding-3-small"
        )
        embeddings.extend([d.embedding for d in response.data])
        print(f"    배치 {i // batch_size + 1}/{(len(texts) - 1) // batch_size + 1} 완료")
    return np.array(embeddings).astype('float32')


def build_index():
    all_chunks = []
    included_labels = []

    # PDF 처리
    for pdf_info in PDF_FILES:
        path  = pdf_info["path"]
        label = pdf_info["label"]
        if not os.path.exists(path):
            print(f"⚠️  파일 없음 (건너뜀): {path}")
            continue
        print(f"\n[{label}] 처리 시작")
        chunks = extract_text_from_pdf(path, label)
        all_chunks.extend(chunks)
        included_labels.append(label)

    # JSONL 처리
    for jsonl_info in JSONL_FILES:
        path  = jsonl_info["path"]
        label = jsonl_info["label"]
        if not os.path.exists(path):
            print(f"⚠️  파일 없음 (건너뜀): {path}")
            continue
        print(f"\n[{label}] 처리 시작")
        chunks = extract_from_jsonl(path, label)
        all_chunks.extend(chunks)
        included_labels.append(label)

    # CSV 처리
    for csv_info in CSV_FILES:
        path  = csv_info["path"]
        label = csv_info["label"]
        if not os.path.exists(path):
            print(f"⚠️  파일 없음 (건너뜀): {path}")
            continue
        print(f"\n[{label}] 처리 시작")
        chunks = extract_from_csv(csv_info)
        all_chunks.extend(chunks)
        included_labels.append(label)

    if not all_chunks:
        print("❌ 인덱싱할 청크가 없습니다. 파일 경로를 확인하세요.")
        return

    print(f"\n총 {len(all_chunks)}개 청크 → 임베딩 생성 시작")
    embeddings = get_embeddings(all_chunks)

    print("\n📈 FAISS 인덱스 구축 중...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    faiss.write_index(index, INDEX_FILE)
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"\n🚀 인덱싱 완료!")
    print(f"   저장 파일: {INDEX_FILE}, {METADATA_FILE}")
    print(f"   총 청크 수: {len(all_chunks)}")
    print(f"   포함된 문서: {included_labels}")


if __name__ == "__main__":
    build_index()
