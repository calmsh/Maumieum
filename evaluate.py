import os
import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings

warnings.filterwarnings("ignore")
# 줄어든 Optuna 로그를 위해 INFO 레벨 로깅 비활성화
optuna.logging.set_verbosity(optuna.logging.WARNING)

def optimize_and_evaluate_models(X, y, title):
    print(f"\n{'='*50}\nEvaluating: {title}\n{'='*50}")
    
    # 1. Preprocessing
    X = X.replace([np.inf, -np.inf], np.nan)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # 평가 대상은 논문에서 사용된 탑티어 모델 4종
    models_to_test = ['RandomForest', 'XGBoost', 'SVM RBF', 'Ridge']
    
    with open("eval_results.txt", "a", encoding="utf-8") as f:
        f.write(f"\n{'='*50}\nEvaluating: {title}\n{'='*50}\n")
        f.write(f"Number of samples: {len(y)}\n\n")
        
        for model_name in models_to_test:
            print(f"[{model_name}] Optuna Hyperparameter Optimization Running...")
            
            # Optuna 최적화를 위한 목적 함수 정의
            def objective(trial):
                # 앙상블 파이프라인: 결측치 대체 -> 스케일링 -> 데이터 증강(SMOTE)
                pipeline = ImbPipeline([
                    ('imputer', KNNImputer(n_neighbors=5)),
                    ('scaler', StandardScaler()),
                    ('sampler', SMOTE(random_state=42))
                ])
                
                # 머신러닝 알고리즘별 하이퍼파라미터 서치 스페이스 구성
                if model_name == 'RandomForest':
                    rf_params = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                        'max_depth': trial.suggest_int('max_depth', 5, 25),
                        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10)
                    }
                    pipeline.steps.append(['model', RandomForestClassifier(**rf_params, random_state=42, n_jobs=1)])
                    
                elif model_name == 'XGBoost':
                    xgb_params = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                        'max_depth': trial.suggest_int('max_depth', 3, 10),
                        'subsample': trial.suggest_float('subsample', 0.5, 1.0)
                    }
                    pipeline.steps.append(['model', XGBClassifier(**xgb_params, random_state=42, n_jobs=1)])
                    
                elif model_name == 'SVM RBF':
                    svm_params = {
                        'C': trial.suggest_float('C', 0.1, 10.0, log=True),
                        'gamma': trial.suggest_float('gamma', 0.001, 1.0, log=True)
                    }
                    pipeline.steps.append(['model', SVC(**svm_params, kernel='rbf', probability=False, random_state=42)])
                    
                elif model_name == 'Ridge':
                    # Ridge는 분류기로 쓰일 때 강력한 정규화 효과를 발휘함
                    ridge_params = {
                        'alpha': trial.suggest_float('alpha', 0.1, 100.0, log=True)
                    }
                    pipeline.steps.append(['model', RidgeClassifier(**ridge_params, random_state=42)])
                    
                # 5-Fold 점수 산출
                scores = cross_validate(pipeline, X, y, cv=cv, scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'], n_jobs=-1, return_train_score=False)
                
                # Optuna 기록용 User Attributes (수동 로깅)
                trial.set_user_attr('accuracy', scores['test_accuracy'].mean())
                trial.set_user_attr('precision', scores['test_precision_macro'].mean())
                trial.set_user_attr('recall', scores['test_recall_macro'].mean())
                
                # 최종 최대화 목표: Multi-class 밸런스를 측정하는 최적의 지표인 Macro F1
                return scores['test_f1_macro'].mean()

            # TPE(Tree-structured Parzen Estimator) 활용 지능형 서치
            study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(objective, n_trials=30)
            
            # 최상위 모델 산출
            best_trial = study.best_trial
            f1 = best_trial.value
            acc = best_trial.user_attrs['accuracy']
            prec = best_trial.user_attrs['precision']
            rec = best_trial.user_attrs['recall']
            
            print(f"--- {model_name} (Tuned) ---")
            print(f"Accuracy:  {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1 Score: {f1:.4f}\n")
            
            f.write(f"--- {model_name} (Tuned) ---\n")
            f.write(f"Best Params: {best_trial.params}\n")
            f.write(f"Accuracy:  {acc:.4f}\n")
            f.write(f"Precision: {prec:.4f}\n")
            f.write(f"Recall:    {rec:.4f}\n")
            f.write(f"F1 Score:  {f1:.4f}\n\n")

if __name__ == "__main__":
    df = pd.read_csv("extracted_features.csv")
    print(f"Loaded {len(df)} samples from dataset.")
    
    if os.path.exists("eval_results.txt"):
        os.remove("eval_results.txt")
        
    df = df.dropna(subset=['Label'])
    
    # 카테고리성 텍스트 라벨을 숫자 배열로 최적화 인코딩 (XGBoost 필수 조건)
    le = LabelEncoder()
    df['Encoded_Label'] = le.fit_transform(df['Label'])
    print(f"Classes Mapped: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    # 💡 데이터 유출 (Data Leakage) 차단:
    # 절대 포함되어서는 안되는 정답 의존 변수(PHQ-9 점수, 파일명) 철저히 제외
    exclude_cols = ['Label', 'Encoded_Label', 'FileName']
    if 'PHQ9_Score' in df.columns:
        exclude_cols.append('PHQ9_Score')
        
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X = df[feature_cols]
    
    # 통합된 임상 설문 마커 검증
    clinical_markers = [c for c in feature_cols if c.startswith("Clinical")]
    print(f"\n[Check] Added Clinical Multimodal Features: {clinical_markers}")
    
    # 3-Class Model (Normal, Moderate, Severe) 성능 튜닝 검증 시작
    if len(df['Label'].unique()) > 1:
        optimize_and_evaluate_models(X, df['Encoded_Label'], "3-Class Optuna Tuned Classifier (HRV + Clinical Markers)")
