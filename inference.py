import joblib
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

def predict_new_patient(csv_file_path):
    print(f"\n==============================================")
    print(f" AI 중증도 판독 시스템 (Inference Engine) 가동")
    print(f"==============================================\n")
    
    # 1. 영구 보존된 모델과 라벨 인코더 메모리로 불러오기
    try:
        model_pipeline = joblib.load('xgboost_depression_model.pkl')
        print("[시스템] XGBoost 파이프라인 로드 완료.")
    except Exception as e:
        print("[에러] 모델 파일(.pkl)을 찾을 수 없습니다. 먼저 `python train_xgboost.py`를 실행하세요.")
        return
        
    # 2. 새로운 환자 데이터 스캔 (센서 및 설문을 통해 수합된 결과)
    try:
        df = pd.read_csv(csv_file_path)
    except Exception:
        print(f"[에러] 환자 데이터 파일({csv_file_path})을 읽을 수 없습니다.")
        return
    
    # 3. 모델이 요구하는 필수 컬럼들로만 구성
    # (학습 때 병합된 컬럼 개수와 정확히 일치해야 파이프라인 필터가 작동함)
    expected_features = model_pipeline.named_steps['model'].get_booster().feature_names
    
    for f in expected_features:
        if f not in df.columns:
            # 환자의 측정값 일부가 누락(결측치)되었다면 NaN 처리
            # 파이프라인 앞단의 KNNImputer가 이 누락된 부분들을 기존 이웃 데이터를 토대로 완벽하게 채워줌!
            df[f] = np.nan 
            
    X_new = df[expected_features]
    
    # 4. [중요] 파이프라인을 통한 통합 추론 호출
    # 단 한 줄로 환자 데이터 결측치 보정 -> 스케일링 -> 예측까지 자동으로 흘러감
    predictions = model_pipeline.predict(X_new)
    pred_probabilities = model_pipeline.predict_proba(X_new)
    
    # 5. [0, 1, 2] 로 도출된 결과를 [Normal, Moderate, Severe] 로 원상 복구
    class_map = {0: 'Normal', 1: 'Moderate', 2: 'Severe'}
    pred_labels = [class_map[p] for p in predictions]
    
    print("\n🚀 [새로운 환자 우울증 진단 예측 리포트]")
    for i in range(len(df)):
        # 만약 CSV 내에 환자 식별 코드가 있다면 가져오기
        patient_id = df.loc[i, 'FileName'] if 'FileName' in df.columns else f"미식별 환자_{i+1}"
        
        # 퍼센트(Probability) 텍스트 조립
        prob_str = " | ".join([f"{class_map[j]}: {p*100:5.1f}%" for j, p in enumerate(pred_probabilities[i])])
        
        # 최종 콘솔 출력
        print(f"▶ 대상: {patient_id}")
        print(f"   ➔ AI 추정 등급: [{pred_labels[i]}]")
        print(f"   ➔ 세부 확신도: ({prob_str})\n")

if __name__ == "__main__":
    # 처음 서비스를 돌릴 때를 대비한 모의(Mock) 환자 데이터 생성 로직
    test_file = "extracted_features.csv" 
    try:
        # 기존 1000개 데이터베이스 중 무작위 5명을 빼와서 
        # 처음 진단받는 신규 환자인 척 위장해봅니다.
        df_test = pd.read_csv(test_file).sample(n=5, random_state=77)
        df_test.to_csv("sample_new_patients.csv", index=False)
        
        # 시스템에 환자 정보 파일 투입
        predict_new_patient("sample_new_patients.csv")
    except Exception as e:
        print("모의 테스트 생성 실패:", e)
