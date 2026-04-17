# 마음이음 (Maumium)

마음이음은 **rPPG 기반 HRV(심박 변이도) 측정**과 **AI 상담**을 결합하여 사용자의 우울증 및 스트레스를 관리하는 디지털 헬스케어 솔루션입니다.

## 🌟 주요 기능

- **심박 변이도(HRV) 측정**: 스마트폰 카메라를 이용한 rPPG 기술로 SDNN, RMSSD 등 주요 자율신경 지표를 분석합니다.
- **우울함 자가진단 (PHQ-9)**: 글로벌 표준 우울증 선별 검사를 통해 주관적 상태를 정량화합니다.
- **AI 마음지기**: GPT-4o와 전문 심리치료 매뉴얼(RAG)을 결합하여 개인 맞춤형 심리 상담을 제공합니다.
- **활동 및 기분 기록**: 사용자의 일과와 기분 변화를 시각화하여 자기관리 과정을 돕습니다.

## 🏗️ 기술 스택

- **Backend**: FastAPI (Python)
- **Frontend**: Vanilla JS, HTML, CSS, Chart.js
- **AI/ML**: 
  - OpenAI GPT-4o (상담 모델)
  - FAISS (지식 베이스 검색/RAG)
  - XGBoost (HRV 우울증 예측 모델)
- **Bio-Signal**: NeuroKit2 (HRV 분석 엔진)

## 🚀 시작하기

### 1. 필수 조건
- Python 3.9 이상
- OpenAI API Key

### 2. 설치 방법
```bash
# 저장소 복제
git clone https://github.com/calmsh/Maumieum.git
cd Maumieum

# 필수 라이브러리 설치
pip install -r requirements.txt
```

### 3. 환경 설정
루트 폴더 내에 `.env` 파일을 생성하고 다음과 같이 API 키를 설정합니다.
```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 4. 실행 방법
```bash
# 모든 파일이 루트 폴더에 준비되어 있습니다.
python api.py
```
서버 실행 후 브라우저에서 `http://localhost:8000/index.html`으로 접속하세요.

## 📁 주요 파일 구조

- `api.py`: 백엔드 API 서버 (메인 실행 파일)
- `index.html`: 프론트엔드 웹 페이지
- `manual_index.faiss`: 상담 지식 베이스 검색 인덱스
- `xgboost_depression_model.pkl`: HRV 분석용 학습 모델

---
본 프로젝트는 사용자의 신체적, 심리적 데이터를 종합하여 더 나은 마음 건강을 돕기 위해 개발되었습니다.
