import os
import json
import joblib
import numpy as np
import pandas as pd
import neurokit2 as nk
from scipy.signal import butter, filtfilt
from scipy.interpolate import CubicSpline
from xgboost import XGBClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
from imblearn.combine import SMOTEENN
import warnings
from tqdm import tqdm

# 경고 무시
warnings.filterwarnings('ignore')

# --- 설정 (Configuration) ---
DATA_DIR = 'd:/hrvdata/data'
ORIGINAL_FS = 30.0  # 카메라 기본 프레임레이트
TARGET_FS = 70.0    # 최적 파이프라인 (보고서 섹션 2.1)
LOWCUT = 0.75
HIGHCUT = 2.5

"""
================================================================================
1. 데이터 로딩 및 타겟 생성 (Task Mapping)
- PHQ-9 점수를 기반으로 3단계 중증도 분류 (0-9, 10-19, 20-27)
================================================================================
"""
def get_3class_label(phq9_score):
    if phq9_score <= 9: return 0  # 정상/경증
    if phq9_score <= 19: return 1 # 중등도
    return 2                      # 중증

def load_data_from_json(data_dir):
    records = []
    files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    
    print(f"데이터 로드 시작: {len(files)}개 파일 탐색 중...")
    
    for fname in tqdm(files):
        try:
            with open(os.path.join(data_dir, fname), 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # PHQ-9 점수 추출
            phq9_dialog = data.get('assessment', {}).get('PHQ-9', {}).get('dialog', [])
            if not phq9_dialog: continue
            score = sum(i.get('point', 0) for i in phq9_dialog)
            
            # 인구통계 정보 (나이, 성별)
            sf = data.get('sf', {})
            birth_year = sf.get('birthyear')
            gender = sf.get('gender')
            if birth_year is None or gender is None: continue
            
            age = 2026 - int(birth_year)
            gender_val = 1 if str(gender).upper() == 'M' else 0
            
            # HRV RAW 데이터 (meta)
            meta = data.get('hrv_deep', {}).get('data', {}).get('meta', [])
            if len(meta) < 300: continue # 최소 데이터 길이 보장
            
            time_us = np.array([m['time'] for m in meta])
            r = np.array([m['sigR'] for m in meta])
            g = np.array([m['sigG'] for m in meta])
            b = np.array([m['sigB'] for m in meta])
            
            records.append({
                'file': fname,
                'phq9_score': score,
                'target': get_3class_label(score),
                'age': age,
                'gender': gender_val,
                'time_us': time_us,
                'r': r, 'g': g, 'b': b
            })
        except Exception:
            continue
            
    return records

"""
================================================================================
2. 신호 전처리 (Preprocessing: 70Hz Cubic Spline + CHROM)
- 보고서 섹션 2.1: 카메라 지연 보정 및 미세 변이(HRV) 해상도 극대화
================================================================================
"""
def process_hrv_signal(record):
    # 1. Cubic Spline 리샘플링 (70Hz)
    time_sec = (record['time_us'] - record['time_us'][0]) / 1_000_000.0
    duration = time_sec[-1]
    new_t = np.linspace(0, duration, int(duration * TARGET_FS))
    
    # R, G, B 각각 보간
    r_res = CubicSpline(time_sec, record['r'])(new_t)
    g_res = CubicSpline(time_sec, record['g'])(new_t)
    b_res = CubicSpline(time_sec, record['b'])(new_t)
    
    # 2. CHROM 알고리즘 적용 (PPG 추출)
    rm, gm, bm = np.mean(r_res), np.mean(g_res), np.mean(b_res)
    if rm == 0 or gm == 0 or bm == 0: return None
    
    rn, gn, bn = r_res / rm, g_res / gm, b_res / bm
    
    # Bandpass Filter (0.75-2.5Hz)
    nyq = 0.5 * TARGET_FS
    b, a = butter(4, [LOWCUT / nyq, HIGHCUT / nyq], btype='band')
    
    rf = filtfilt(b, a, rn)
    gf = filtfilt(b, a, gn)
    bf = filtfilt(b, a, bn)
    
    x = 3 * rf - 2 * gf
    y = 1.5 * rf + gf - 1.5 * bf
    alpha = np.std(x) / np.std(y) if np.std(y) > 0 else 1.0
    ppg = x - alpha * y
    
    return ppg

"""
================================================================================
3. 특징 추출 (Feature Extraction: Standard + Complexity)
- 보고서 섹션 2.2: Entropy 및 비선형 지표 중심의 특징 구성
================================================================================
"""
def extract_optimal_features(ppg_signal):
    try:
        # PPG Clean & Peak Detection
        ppg_clean = nk.ppg_clean(ppg_signal, sampling_rate=TARGET_FS)
        peaks = nk.ppg_findpeaks(ppg_clean, sampling_rate=TARGET_FS)['PPG_Peaks']
        
        # 1. 표준 HRV 특징 (Time, Frequency)
        hrv_std = nk.hrv(peaks, sampling_rate=TARGET_FS)
        
        # 2. 비선형 및 복잡도 특징 (MSEn, FuzzyEn 등)
        # NeuroKit2 hrv() 함수는 기본적으로 여러 비선형 지표를 포함함
        # 필요한 경우 추가 계산 가능
        
        features = {}
        for col in hrv_std.columns:
            features[col] = hrv_std[col].values[0]  # HRV_ 접두사 유지 (api.py 호환)

        return features
    except:
        return None

"""
================================================================================
4. 모델 학습 및 평가 (Model Training: Cost-Sensitive XGBoost)
- 보고서 섹션 6.3: 변별력 측면에서 최상의 성능을 보인 전략 적용
================================================================================
"""
MODEL_PATH       = "xgboost_depression_model.pkl"
FEATURE_NAMES_PATH = "feature_names.json"

def make_base_xgb():
    return XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        eval_metric='mlogloss'
    )


def train_optimal_model(X, y):
    print("\n[모델 학습 및 평가 시작: 5-Fold Stratified CV]")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    feature_cols = list(X.columns)

    metrics_list = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx].values, X.iloc[val_idx].values
        y_train, y_val = y.iloc[train_idx].values, y.iloc[val_idx].values

        # 결측치 보완 + 스케일링
        imputer = KNNImputer(n_neighbors=5)
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(imputer.fit_transform(X_train))
        X_val   = scaler.transform(imputer.transform(X_val))

        # SMOTE-ENN 오버샘플링
        smote_enn = SMOTEENN(random_state=42)
        X_train_res, y_train_res = smote_enn.fit_resample(X_train, y_train)

        # OvO XGBoost
        model = OneVsOneClassifier(make_base_xgb())
        model.fit(X_train_res, y_train_res)

        y_pred  = model.predict(X_val)
        y_proba = model.predict_proba(X_val)

        acc = accuracy_score(y_val, y_pred)
        f1  = f1_score(y_val, y_pred, average='macro')
        mcc = matthews_corrcoef(y_val, y_pred)
        auc = roc_auc_score(y_val, y_proba, multi_class='ovr', average='macro')

        metrics_list.append([acc, f1, mcc, auc])
        print(f"Fold {fold+1}: Acc={acc:.4f}, F1={f1:.4f}, MCC={mcc:.4f}, AUC={auc:.4f}")

    avg_metrics = np.mean(metrics_list, axis=0)
    print("\n" + "="*50)
    print("최종 평균 성능 (Final Average Performance)")
    print(f"Accuracy:  {avg_metrics[0]:.4f}")
    print(f"F1-Score:  {avg_metrics[1]:.4f}")
    print(f"MCC:       {avg_metrics[2]:.4f}")
    print(f"AUC:       {avg_metrics[3]:.4f}")
    print("="*50)

    # ── 전체 데이터로 최종 모델 학습 및 저장 ────────────────────
    print("\n[최종 모델 전체 데이터 학습 및 저장 중...]")
    X_all = X.values
    y_all = y.values

    final_imputer = KNNImputer(n_neighbors=5)
    final_scaler  = StandardScaler()
    X_all_imp = final_imputer.fit_transform(X_all)
    X_all_sc  = final_scaler.fit_transform(X_all_imp)

    smote_enn_final = SMOTEENN(random_state=42)
    X_res, y_res = smote_enn_final.fit_resample(X_all_sc, y_all)

    final_ovo = OneVsOneClassifier(make_base_xgb())
    final_ovo.fit(X_res, y_res)

    # sklearn Pipeline으로 묶어 저장 (api.py 호환)
    final_pipeline = Pipeline([
        ('imputer', final_imputer),
        ('scaler',  final_scaler),
        ('model',   final_ovo),
    ])

    joblib.dump(final_pipeline, MODEL_PATH)
    with open(FEATURE_NAMES_PATH, 'w', encoding='utf-8') as f:
        json.dump(feature_cols, f, ensure_ascii=False)

    print(f"✅ 모델 저장 완료: {MODEL_PATH}")
    print(f"✅ 피처 목록 저장 완료: {FEATURE_NAMES_PATH} ({len(feature_cols)}개)")
    return final_pipeline

def main():
    # 1. 데이터 로드
    raw_records = load_data_from_json(DATA_DIR)
    
    final_data = []
    print("\n전처리 및 특징 추출 중...")
    
    # 2 & 3. 전처리 및 특징 추출
    for rec in tqdm(raw_records):
        ppg = process_hrv_signal(rec)
        if ppg is None: continue
        
        hrv_features = extract_optimal_features(ppg)
        if hrv_features:
            # Context 정보 추가
            hrv_features['Demographic_Age']    = rec['age']
            hrv_features['Demographic_Gender'] = rec['gender']
            hrv_features['target'] = rec['target']
            final_data.append(hrv_features)
            
    if not final_data:
        print("유효한 데이터가 추출되지 않았습니다.")
        return
        
    df = pd.DataFrame(final_data)
    
    # 데이터 정제 (Inf, NaN 처리)
    X = df.drop(columns=['target'])
    y = df['target']
    
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median()).fillna(0)
    
    # 4. 모델 학습, 평가 및 저장
    train_optimal_model(X, y)
    print("\n모든 분석 및 저장이 성공적으로 완료되었습니다. 🎉")

if __name__ == "__main__":
    main()
