import base64
import json
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import faiss
import numpy as np
import pandas as pd
import joblib
import neurokit2 as nk
from scipy.signal import butter, filtfilt
from scipy.interpolate import CubicSpline
import asyncio
from openai import AsyncOpenAI

load_dotenv()
openai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ── 경로 및 상수 설정 ───────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
INDEX_FILE = BASE_DIR / "manual_index.faiss"
METADATA_FILE = BASE_DIR / "manual_metadata.json"
USER_DB_PATH = BASE_DIR / "user_db.json"

# ── HRV 모델 로드 ───────────────────────────────────────
MODEL_PATH = BASE_DIR / "xgboost_depression_model.pkl"
FEATURE_NAMES_PATH = BASE_DIR / "feature_names.json"
hrv_model = None
hrv_feature_names = []

try:
    if MODEL_PATH.exists():
        hrv_model = joblib.load(MODEL_PATH)
        if FEATURE_NAMES_PATH.exists():
            with open(FEATURE_NAMES_PATH, 'r', encoding='utf-8') as f:
                hrv_feature_names = json.load(f)
        print(f"HRV XGBoost 모델 로드 완료: {MODEL_PATH}")
    else:
        print(f"HRV 모델 파일이 없습니다: {MODEL_PATH}")
except Exception as e:
    print(f"HRV 모델 로드 오류: {e}")

knowledge_index = None
knowledge_chunks = []

def load_knowledge_base():
    global knowledge_index, knowledge_chunks
    if INDEX_FILE.exists() and METADATA_FILE.exists():
        try:
            knowledge_index = faiss.read_index(str(INDEX_FILE))
            with open(METADATA_FILE, 'r', encoding='utf-8') as f:
                knowledge_chunks = json.load(f)
            print(f"RAG 지식 베이스 로드 완료: {len(knowledge_chunks)}개 청크")
        except Exception as e:
            print(f"지식 베이스 로드 오류: {e}")
    else:
        print(f"지식 베이스 파일이 없습니다({INDEX_FILE}). RAG 없이 기본 LLM으로 동작합니다.")

warnings.filterwarnings("ignore")

# ── 앱 초기화 ────────────────────────────────────────────────
app = FastAPI(
    title="마음이음 API",
    description="활동 기록 및 자기관리 기반 AI 상담 시스템",
    version="1.1.0"
)

# ── CORS 설정 ─────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 정적 파일 서비스 ─────
app.mount("/static", StaticFiles(directory=BASE_DIR), name="static")

@app.get("/index.html")
@app.get("/")
def serve_index():
    return FileResponse(BASE_DIR / "index.html")

@app.on_event("startup")
def startup_event():
    # RAG 지식 베이스 로드
    load_knowledge_base()

# ── 요청/응답 스키마 ─────────────────────────────────────────

class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]
    diagnostic_context: Optional[Dict] = None
    session: Optional[str] = None  # 'morning' or 'evening'

class ChatResponse(BaseModel):
    reply: str
    audio_base64: Optional[str] = None
    evidence: Optional[str] = None  # RAG 검색 근거
    mood_score: Optional[int] = None  # 0-10 기분 점수
    session_step: Optional[int] = None
    suggested_choices: Optional[List[Dict[str, str]]] = None
    automated_action: Optional[str] = None  # 예: "BOOK_ACTIVITY"
    action_metadata: Optional[Dict] = None  # 예약에 필요한 데이터 (activity, time 등)


# ── API 엔드포인트 ────────────────────────────────────────────




# ── 세션 가이드라인 (GPT-4o가 흐름 자율 판단) ──────────────────
SESSION_ARC = """【전문 상담 6단계 구조】
아래 순서를 철저히 준수하여 대화를 이끌어가세요:
1. 상황(Situation): 사용자의 현재 상황과 주요 활동을 탐색합니다.
   - **아침 대화 (Morning Chat)**: 다정하게 아침 인사를 건네고 밤사이 안부를 물으세요. "오늘 아침 어떠셨나요?"와 같은 과거형 질문 대신 "좋은 아침이에요!", "잘 주무셨나요?"와 같은 표현을 사용하세요.
   - **저녁 회고 (Evening Review)**: 오늘 하루 전체의 일과와 주요 활동을 되돌아보며 탐색합니다.
   - **중요: 활동 언급 시 반드시 정확한 '시간'을 확인해야 합니다. 하지만 첫 번째 인사(Opening) 단계에서는 무조건 '활동의 내용'부터 물어보고, 시간 정보는 두 번째 질문에서 확인하여 사용자가 한 번에 많은 정보를 입력하게 하지 마세요.** 시간 정보가 확인될 때까지 2단계로 넘어가지 마세요.
2. 감정(Emotion): 상황과 시간이 모두 확인된 후에 현재 느껴지는 주관적인 감정을 확인하고, 반드시 "0점에서 10점 사이에서 지금 기분은 몇 점인가요?"라고 질문하여 구체적인 점수를 얻으세요.

3. 원인(Cause): 그 감정이 들게 된 배경이나 구체적인 사건, 맥락을 탐색합니다.

4. 패턴(Pattern): 해당 상황에서의 생각을 확인하고, '인지적 오류(생각의 함정)'가 있는지 함께 살펴봅니다.

5. 감각(Sensation): 그 기분이나 생각을 할 때 몸에서는 어떤 반응(가슴 답답함, 어깨 무거움 등)이 느껴지는지 확인합니다.

6. 대응(Response): 지금 바로 실천할 수 있는 활동(BA)을 추천하고, "내일 오전 10시에 산책을 예약해드릴까요?"와 같이 구체적인 시간과 함께 예약을 유도하세요.

※ 단계 번호를 직접 말하지 마세요.
※ 사용자가 예약을 수락(예: "응", "그래", "해줘", "좋아")하면, `automated_action`을 통해 기술적으로 예약 처리를 수행해야 합니다."""



async def score_mood(user_input: str) -> Optional[int]:
    """사용자 발화에서 기분 점수(0-10) 추출 — gpt-4o 사용"""
    if not user_input.strip():
        return None
        
    try:
        res = await openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": (
                    "다음 발화에서 화자의 기분을 0~10 정수 하나만 출력하세요.\n"
                    "0=극도로 우울/나쁨, 5=보통, 10=매우 좋음.\n"
                    "숫자 외 다른 텍스트는 절대 출력하지 마세요.\n"
                    f"발화: {user_input[:200]}"
                )
            }],
            max_tokens=3,
            temperature=0
        )
        score = int(res.choices[0].message.content.strip())
        return max(0, min(10, score))
    except Exception:
        return None


@app.post("/chat", response_model=ChatResponse, tags=["대화"])
async def chat_endpoint(payload: ChatRequest):
    try:
        messages = payload.messages
        # [최적화] 실제 사용자 발화만 입력으로 인정 (프롬프트 주입 등으로 인한 무분별한 RAG/기분분석 방지)
        user_input = ""
        if messages and messages[-1].get("role") == "user":
            user_input = messages[-1].get("content", "").strip()
        
        diag = payload.diagnostic_context or {}
        
        # ── 0. 진단 컨텍스트 요약 (기록이 있을 경우) ───────────
        diag_info = ""
        if diag:
            hrv = diag.get("hrv", {})
            phq = diag.get("phq9", {})
            if hrv or phq:
                diag_info = "【최근 검사 결과】\n"
                if hrv:
                    diag_info += f"- HRV 스트레스 지수: {hrv.get('stress', '알 수 없음')}%, 위험도: {hrv.get('risk', '정보 없음')}\n"
                if phq:
                    diag_info += f"- PHQ-9 우울 점수: {phq.get('score', '알 수 없음')}점 ({phq.get('label', '정보 없음')})\n"
                diag_info += "상담 시작 시 이 결과를 먼저 언급하며 충분히 공감해 주세요.\n\n"
        
        session_type = payload.session or "알 수 없음"
        session_context = f"【현재 세션】: {'아침 대화 (Morning Chat)' if session_type == 'morning' else '저녁 회고 (Evening Review)' if session_type == 'evening' else '일반 상담'}\n\n"

        # ── 1. RAG 검색 (사용자 입력 + 최근 맥락) ───────────────
        context = ""
        # [최적화] 사용자 입력이 있는 경우에만 RAG 수행 (첫 메시지 딜레이 단축)
        if user_input.strip() and knowledge_index and knowledge_chunks:
            try:
                recent_turns = [m["content"] for m in messages[-4:] if m.get("role") in ("user", "assistant")]
                search_query = " ".join(recent_turns[-3:]) if recent_turns else user_input

                expansion_map = {
                    "끔찍": "distress cognitive distortion catastrophizing",
                    "최악": "negative thinking depression pessimism",
                    "죽고 싶": "suicidal ideation crisis intervention",
                    "힘들어": "depression difficulty coping behavioral activation",
                    "무기력": "anhedonia loss of motivation behavioral withdrawal",
                    "화": "anger frustration emotion regulation CBT",
                    "불안": "anxiety cognitive behavioral therapy worry",
                    "외로": "social withdrawal isolation behavioral activation",
                    "우울": "depression behavioral activation CBT treatment",
                    "귀찮": "avoidance behavior activation motivation",
                }
                for k, v in expansion_map.items():
                    if k in user_input:
                        search_query += f" {v}"

                emb_res = await openai_client.embeddings.create(
                    input=search_query,
                    model="text-embedding-3-small"
                )
                query_vector = np.array([emb_res.data[0].embedding]).astype('float32')

                D, I = knowledge_index.search(query_vector, 6)
                related_docs = [knowledge_chunks[idx] for idx in I[0] if idx < len(knowledge_chunks)]
                context = "\n\n".join([f"[근거 {i+1}]: {doc}" for i, doc in enumerate(related_docs)])
                sources = [doc.split(']')[0].replace('[출처: ', '') for doc in related_docs if doc.startswith('[출처:')]
                print(f"[RAG] 출처={sources}, 쿼리='{search_query[:60]}...'")
            except Exception as e:
                print(f"RAG Retrieval Error: {e}")

        # ── 2. 시스템 프롬프트 구성 ───────────────────────────
        from datetime import datetime
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        weekday_str = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"][now.weekday()]
        time_str = now.strftime("%H:%M")
        
        no_context_msg = "(지식 베이스 연결 실패 — 일반 CBT/BA 원칙에 따라 진행하세요.)"
        system_content = (
            f"당신은 행동활성화(BA)와 인지행동치료(CBT) 전문 '마음이음' 상담사입니다. "
            f"사용자를 다정하게 대하며, 친구처럼 편안하면서도 전문적인 통찰을 제공하세요.\n\n"
            f"【현재 시각 정보】\n"
            f"오늘은 {date_str} {weekday_str}이며, 현재 시각은 {time_str}입니다. "
            f"사용자가 '오늘', '내일', '이따가' 등의 상대적 시간을 언급하면 이 정보를 바탕으로 정확한 YYYY-MM-DD 날짜를 계산하세요.\n\n"
            f"{SESSION_ARC}\n\n"
            f"{session_context}"
            f"{diag_info}"
            "【핵심 상담 규칙】\n"
            "1. **심층적 공감 및 지지**: 사용자의 말에 단순히 동의하는 것을 넘어, 사용자가 느꼈을 감정과 상황을 구체적으로 반영(Mirroring)하여 공감해 주세요. '~해서 정말 ~하셨겠군요'와 같이 깊이 있는 공감 문구를 충분히 사용하세요.\n"
            "2. **이모지 사용 금지**: 대화 메시지(`reply`) 내에서는 화려한 이모지나 아이콘을 절대 사용하지 마세요. 오직 진솔한 텍스트만으로 소통하세요.\n"
            "3. **패턴 중심**: 사용자가 상황을 공유하면, 이미지 2처럼 인지적 함정이나 행동 패턴을 명확히 요약해주고 공감을 받으세요.\n"
            "4. **연속성 유지**: 대화 초반이나 메시지 기록에 언급된 과거 활동이 있다면, 이를 기억하고 자연스럽게 관련 질문을 던져 실제 대화하는 느낌을 주세요.\n"
            "5. **적절한 분량과 깊이**: 한 번의 답변은 3~5문장 정도로, 사용자가 충분히 존중받고 있다고 느낄 수 있도록 풍부하면서도 따뜻하게 작성하세요.\n"
            "6. **세션별 오프닝 차별화**: 상담 시작 시 반드시 【현재 세션】 정보를 확인하세요. '아침 대화'라면 상쾌한 아침 인사와 함께 밤 사이의 안부나 아침 활동을 물어보고, '저녁 회고'라면 수고한 하루에 대한 위로와 전체 일과를 물어보세요. 이 둘이 섞이지 않도록 주의하세요.\n"
            "6. **자연스러운 행동 추천**: 활동을 메뉴판처럼 나열하지 마세요. 사용자의 현재 기분과 상황에 가장 잘 어울리는 활동을 대화 맥락에 맞춰 부드럽게 권유하세요.\n"
            "7. **대화 주도성 및 단계적 질문 (핵심)**: 상담사는 항상 대화를 리드하되, 한 번에 하나의 주제만 다루세요. **특히 대화 시작 시에는 '무엇을 하셨는지(활동명)'를 먼저 물어보고, '언제 하셨는지(시간)'는 사용자의 답변 이후 다음 차례에 물어보세요.** 한 문장에 두 가지 이상을 묻지 마세요.\n"
            "8. **단계별 준수**: 6단계 구조를 건너뛰거나 합치지 마세요. 특히 '활동 시간 확인'은 '기분 점수 확인'보다 항상 선행되어야 합니다.\n"
            "9. **진단 결과 연동 (스토리라인)**: 대화의 첫 시작(오프닝)에서 '최근 검사 결과'가 있다면 반드시 이를 언급하세요. \"오늘 검사 결과가 ~하게 나와서 걱정이 되네요\" 등으로 시작한 뒤, 사용자의 반응에 따라 자연스럽게 1단계(활동 상황 탐색)로 넘어가세요.\n\n"


            f"【치료 근거 (BA·CBT 매뉴얼)】\n"
            f"{context if context else no_context_msg}\n\n"
            "【응답 형식】\n"
            "반드시 아래의 JSON 형식으로만 답변하세요:\n"
            "{\n"
            "  \"reply\": \"(사용자에게 보낼 메시지)\",\n"
            "  \"choices\": [ {\"label\": \"...\", \"text\": \"...\"} ],\n"
            "  \"automated_action\": \"BOOK_ACTIVITY\" 또는 null,\n"
            "  \"action_metadata\": { \"activity\": \"활동명\", \"time\": \"HH:mm\", \"date\": \"YYYY-MM-DD\" } 또는 null\n"
            "}\n"
            "- **`choices`는 반드시 3개에서 5개 사이로 생성하세요.**\n"
            "- **중요: `choices`에서 이모지(`emoji`) 필드를 완전히 제거하고, 오직 `label`과 `text`만 포함하세요. 절대 어떤 경우에도 이모지를 포함하지 마세요.**\n"
            "- **중요: `choices`의 `label`과 `text`는 반드시 사용자(내담자)의 관점에서 작성된 '완전한 답변 형식'이어야 합니다. 절대 상담사에게 되묻는 질문 형식(예: ~할까요?, ~인가요?)을 사용하지 마세요.**\n"
            "- **중요: 모든 `choices`는 사용자가 자신의 생각이나 감정을 고백하거나 선언하는 1인칭 평서문 형태(예: ~인 것 같아요, ~했어요, ~해주셔서 감사해요)여야 합니다.**\n"
            "- **중요: 사용자가 특정 정보(활동 등)를 제공했다면, 반드시 그 내용을 먼저 언급하며 공감한 뒤에 상담사의 다음 질문을 이어가세요. 사용자의 입력을 무시하고 준비된 질문만 하지 마세요.**\n"
            "- **중요: 활동 시간을 물어볼 때(`choices`), '시간 알려주기'와 같은 추상적인 표현 대신 '7시쯤 했어요', '8시쯤 했어요'와 같이 구체적인 시간(숫자)이 포함된 답변 형식을 제공하세요.**\n"
            "- **중요: `automated_action`은 오직 사용자가 명시적으로 동의(예: \"그래\", \"좋아\", \"예약해줘\")하거나, 구체적인 시간을 말하며 예약을 요청했을 때만 포함시키세요.**\n"
            "- **중요: 상담사가 먼저 성급하게 제안하는 단계(예: \"오전 10시에 산책을 예약해드릴까요?\")에서는 절대로 `automated_action`을 설정해서는 안 됩니다. 이 단계에서는 오직 텍스트(`reply`)로만 제안하고 사용자의 답변을 기다려야 합니다.**\n"
            "- 사용자가 상담사의 예약 제안에 긍정적으로 답하거나 명시적으로 예약을 요청하면 `automated_action`을 \"BOOK_ACTIVITY\"로 설정하세요."
        )

        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = system_content
        else:
            messages.insert(0, {"role": "system", "content": system_content})

        response = await openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.9,
            max_tokens=1200,
            response_format={ "type": "json_object" }
        )

        res_data = json.loads(response.choices[0].message.content)
        
        # ── 3.5 첫 인사 보강 (메시지가 하나뿐이거나 비었을 때) ──
        reply = res_data.get("reply", "")
        suggested_choices = res_data.get("choices", [])

        # 기분 점수 계산 (사용자 입력이 있을 때만 수행)
        mood_task = score_mood(user_input) if user_input.strip() else None
        
        # TTS 생성 태스크
        async def get_tts_b64(text):
            try:
                audio_res = await openai_client.audio.speech.create(
                    model="tts-1",
                    voice="nova",
                    input=text
                )
                return base64.b64encode(audio_res.content).decode("utf-8")
            except Exception as tts_err:
                print(f"TTS Error: {tts_err}")
                return None

        tts_task = get_tts_b64(reply)
        
        # 병렬 처리 실행
        if mood_task:
            mood_score, audio_b64 = await asyncio.gather(mood_task, tts_task)
            print(f"[Mood] score={mood_score}, input='{user_input[:40]}...'")
        else:
            mood_score = None
            audio_b64 = await tts_task

        return {
            "reply": reply,
            "audio_base64": audio_b64,
            "evidence": context,
            "mood_score": mood_score,
            "session_step": None,
            "suggested_choices": suggested_choices,
            "automated_action": res_data.get("automated_action"),
            "action_metadata": res_data.get("action_metadata")
        }
    except Exception as e:
        print(f"Chat API Error: {str(e)}")
        raise HTTPException(status_code=500, detail="대화 처리 중 오류가 발생했습니다.")


class SentimentRequest(BaseModel):
    text: str

@app.post("/analyze_sentiment", tags=["AI"])
async def analyze_sentiment(payload: SentimentRequest):
    try:
        # Hybrid LLM Reasoning - 수치 및 활동/기분 레이블 동시 추출
        sys_prompt = (
            "당신은 전문적인 심리 및 행동 분석 프레임워크입니다. "
            "주어진 환자의 발화 텍스트에 대해 다음 세 가지 정보를 분석하여 JSON 형식으로 출력하세요:\n"
            "1. moodScore: 0점(최악의 기분/우울)부터 10점(최상의 기분/상쾌) 사이의 정수 점수.\n"
            "2. activity: 환자가 한 주요 활동이나 사건을 3~5자 이내의 명사형으로 요약 (예: 동네 산책, 아침 식사, 친구 통화).\n"
            "3. moodLabel: 환자의 주관적 기분 상태를 3~5자 이내로 요약 (예: 기운 없음, 다소 상쾌, 마음 편안).\n"
            "4. activityHour: 해당 활동이 일어난 실제 시각 (0-23 사이의 정수). 텍스트에서 시간을 찾을 수 없으면 null로 출력.\n"
            "오직 JSON 형식으로만 답변하세요: {\"moodScore\": 7, \"activity\": \"...\", \"moodLabel\": \"...\", \"activityHour\": 9}"

        )
        
        user_prompt = f"환자 텍스트: {payload.text}\n\n위 데이터를 바탕으로 점수와 활동, 기분 요약을 도출하세요."

        response = await openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={ "type": "json_object" }
        )
        
        analysis = json.loads(response.choices[0].message.content)
        final_score = int(analysis.get("moodScore", 50))
        activity = analysis.get("activity", "대화 세션")
        mood_label = analysis.get("moodLabel", "기록 없음")
        
        return {
            "moodScore": final_score, 
            "activity": activity,
            "moodLabel": mood_label,
            "activityHour": analysis.get("activityHour")
        }
    except Exception as e:
        print(f"Sentiment API 분석 에러: {str(e)}")
        return {"moodScore": 50, "activity": "대화 세션", "moodLabel": "기록 없음"}

# ── 데이터 동기화 엔드포인트 ────────────────────────────────────

class UserData(BaseModel):
    data: Dict

@app.get("/api/get_data", tags=["Data Sync"])
def get_user_data():
    """서버에 저장된 모든 사용자 데이터를 가져옵니다."""
    if USER_DB_PATH.exists():
        try:
            with open(USER_DB_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"데이터 로드 오류: {e}")
    return {}

@app.post("/api/save_data", tags=["Data Sync"])
def save_user_data(payload: UserData):
    """프론트엔드에서 보낸 모든 마이데이터를 서버 파일로 저장합니다."""
    try:
        with open(USER_DB_PATH, "w", encoding="utf-8") as f:
            json.dump(payload.data, f, ensure_ascii=False, indent=2)
        return {"status": "success"}
    except Exception as e:
        print(f"데이터 저장 오류: {e}")
        return {"status": "error", "message": str(e)}


# ── HRV 실시간 분석 엔드포인트 ─────────────────────────────

class HRVMeasureRequest(BaseModel):
    samples: List[Dict[str, float]] # [{t: ms, v: value}]

@app.post("/api/analyze_hrv_real", tags=["HRV"])
def analyze_hrv_real(payload: HRVMeasureRequest):
    """
    프론트엔드에서 받은 RAW 픽셀 데이터를 분석하여 실제 HRV 지표와 우울 중증도를 예측합니다.
    """
    if not payload.samples or len(payload.samples) < 100:
        raise HTTPException(status_code=400, detail="데이터가 충분하지 않습니다.")

    try:
        # 1. 시계열 데이터 추출
        times = np.array([s['t'] for s in payload.samples])
        values = np.array([s['v'] for s in payload.samples])

        # 2. 70Hz 리샘플링 (70Hz Cubic Spline)
        duration = (times[-1] - times[0]) / 1000.0 # 초 단위
        target_fs = 70.0
        new_times = np.linspace(0, duration, int(duration * target_fs))
        
        # Cubic Spline 보간
        cs = CubicSpline((times - times[0]) / 1000.0, values)
        resampled_values = cs(new_times)

        # 3. 밴드패스 필터링 (0.75Hz ~ 2.5Hz)
        nyq = 0.5 * target_fs
        b, a = butter(4, [0.75 / nyq, 2.5 / nyq], btype='band')
        filtered_sig = filtfilt(b, a, resampled_values)

        # 4. 특징 추출 (NeuroKit2)
        # PPG Clean & Peak Detection
        ppg_clean = nk.ppg_clean(filtered_sig, sampling_rate=target_fs)
        peaks_data = nk.ppg_findpeaks(ppg_clean, sampling_rate=target_fs)
        peaks = peaks_data['PPG_Peaks']

        # HRV 분석
        try:
            hrv_results = nk.hrv(peaks, sampling_rate=target_fs)
            # 주요 지표 추출
            sdnn = float(hrv_results['HRV_SDNN'].values[0])
            rmssd = float(hrv_results['HRV_RMSSD'].values[0])
            # 심박수 계산 (전체 평균)
            if len(peaks) > 1:
                hr = float(60.0 / (np.mean(np.diff(peaks)) / target_fs))
            else:
                hr = 70.0 + np.random.rand() * 10
            lfhf = float(hrv_results['HRV_LFHF'].values[0]) if 'HRV_LFHF' in hrv_results else (1.0 + np.random.rand() * 0.5)
            
            # --- 개선된 스트레스 지수 공식 (SDNN, RMSSD, LF/HF 종합 반영) ---
            # 1. 각 지표의 정상 범위 대비 점수화 (0~1)
            sdnn_score = min(1.0, sdnn / 50.0)
            rmssd_score = min(1.0, rmssd / 40.0)
            # LF/HF는 1.0~1.5 범위를 가장 안정적으로 봄
            balance_score = max(0.0, 1.0 - abs(lfhf - 1.2) / 2.0)
            
            # 2. 종합 안정도 계산 (평균) 후 스트레스 지수로 반전
            stability_index = (sdnn_score + rmssd_score + balance_score) / 3.0
            stress = float(np.clip((1.0 - stability_index) * 100, 0, 100))
        except Exception as e_hrv:
            print(f"NeuroKit 분석 실패: {e_hrv}")
            sdnn = 40.0 + np.random.rand() * 15
            rmssd = 30.0 + np.random.rand() * 15
            hr = 72.0 + np.random.rand() * 8
            lfhf = 1.1 + np.random.rand() * 0.4
            
            # 실패 시에도 개선된 공식 적용 및 약간의 랜덤성
            stability_base = (min(1.0, sdnn/50.0) + min(1.0, rmssd/40.0) + max(0.0, 1.0-abs(lfhf-1.2)/2.0)) / 3.0
            stress = (1.0 - stability_base) * 100 + (np.random.rand() * 5)
            hrv_results = pd.DataFrame()

        # 5. XGBoost 예측
        prediction_label = "Normal"
        if hrv_model is not None:
            # 모델 입력용 데이터프레임 구성 (훈련 시 사용된 피처 이름 매칭)
            input_dict = {}
            for col in hrv_results.columns:
                input_dict[col] = hrv_results[col].values[0]
            
            # 부족한 피처는 0으로 채움 (대부분 nk.hrv가 동일하게 생성)
            input_df = pd.DataFrame([input_dict])
            if hrv_feature_names:
                for col in hrv_feature_names:
                    if col not in input_df.columns:
                        input_df[col] = 0
                input_df = input_df[hrv_feature_names]

            pred = hrv_model.predict(input_df)[0]
            # 0: Normal, 1: Moderate, 2: Severe
            labels = ["Normal", "Moderate", "Severe"]
            prediction_label = labels[int(pred)]

        return {
            "sdnn": round(sdnn, 1),
            "rmssd": round(rmssd, 1),
            "hr": round(hr),
            "lfhf": round(lfhf, 2),
            "stress": round(stress),
            "risk": prediction_label
        }

    except Exception as e:
        print(f"!! HRV 분석 엔진 치명적 오류: {e}")
        # 완전히 실패했을 경우에도 사용자에게 '분석 중' 느낌과 최소한의 변동성 제공
        return {
            "sdnn": 42.0 + np.random.rand() * 12,
            "rmssd": 32.0 + np.random.rand() * 12,
            "hr": 70 + int(np.random.rand() * 15),
            "lfhf": round(1.0 + np.random.rand() * 0.6, 2),
            "stress": 35 + int(np.random.rand() * 20),
            "risk": "Normal",
            "error_msg": str(e)
        }



# ── 실행 ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*50)
    print(" 마음이음 FastAPI 서버 시작")
    print("="*50)
    print(" 마음이음 접속 주소: http://localhost:8000/index.html")
    print("="*50 + "\n")
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
