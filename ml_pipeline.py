import os
import glob
import json
import numpy as np
import pandas as pd
import neurokit2 as nk
import warnings
from scipy.stats import skew, kurtosis

warnings.filterwarnings("ignore")

def extract_features_and_label(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Parse PHQ-9
        phq9_dialog = data.get('assessment', {}).get('PHQ-9', {}).get('dialog', [])
        if not phq9_dialog:
            return None
            
        phq9_score = 0
        for item in phq9_dialog:
            if item.get('id', '').startswith('phq9_sv_'):
                phq9_score += item.get('point', 0)
                
        # Parse CSS / GAD-7 / KOSS for Multi-modal Clinical Features
        gad7_dialog = data.get('assessment', {}).get('GAD-7', {}).get('dialog', [])
        gad7_score = sum(item.get('point', 0) for item in gad7_dialog)
        
        css_dialog = data.get('assessment', {}).get('CSS', {}).get('dialog', [])
        css_score = sum(item.get('point', 0) for item in css_dialog)
        
        koss_dialog = data.get('assessment', {}).get('KOSS', {}).get('dialog', [])
        koss_score = sum(item.get('point', 0) for item in koss_dialog)
                
        # Label mapping (3-Class: Normal, Moderate, Severe)
        if phq9_score <= 4:
            label = "Normal"
        elif 5 <= phq9_score <= 14:
            label = "Moderate"
        else: # 15 ~ 27
            label = "Severe"
            
        # PPG Signal (sigR, sigG, sigB)
        meta = data.get('hrv_deep', {}).get('data', {}).get('meta', [])
        if len(meta) < 100:
            return None
            
        times = [m.get('time', 0) for m in meta]
        sigR = [m.get('sigR', 0) for m in meta]
        sigG = [m.get('sigG', 0) for m in meta]
        sigB = [m.get('sigB', 0) for m in meta]
        
        times_sec = np.array(times) / 1e6
        dt = np.diff(times_sec)
        
        if len(dt) == 0 or np.mean(dt) <= 0:
            return None
            
        sampling_rate = 1.0 / np.mean(dt)
        
        # CHROM Algorithm
        # 1. Bandpass filter 0.75 - 2.5Hz
        R_f = nk.signal_filter(sigR, sampling_rate=sampling_rate, lowcut=0.75, highcut=2.5, method='butterworth', order=3)
        G_f = nk.signal_filter(sigG, sampling_rate=sampling_rate, lowcut=0.75, highcut=2.5, method='butterworth', order=3)
        B_f = nk.signal_filter(sigB, sampling_rate=sampling_rate, lowcut=0.75, highcut=2.5, method='butterworth', order=3)
        
        # 2. Compute X and Y
        X = 3 * R_f - 2 * G_f
        Y = 1.5 * R_f + G_f - 1.5 * B_f
        
        # 3. Alpha
        Y_std = np.std(Y)
        alpha = np.std(X) / Y_std if Y_std != 0 else 1.0
        
        # 4. CHROM BVP Signal
        chrom_sig = X - alpha * Y
        
        # 5. Clean BVP signal for robust peak detection
        clean_sig = nk.ppg_clean(chrom_sig, sampling_rate=sampling_rate, method='elgendi')
        
        # 6. Extract Morphological Features from the clean BVP signal
        morph_features = {
            'bvp_mean': float(np.mean(clean_sig)),
            'bvp_std': float(np.std(clean_sig)),
            'bvp_var': float(np.var(clean_sig)),
            'bvp_skewness': float(skew(clean_sig)),
            'bvp_kurtosis': float(kurtosis(clean_sig)),
            'bvp_energy': float(np.sum(np.square(clean_sig))),
            'bvp_ptp': float(np.ptp(clean_sig))
        }
        
        # Find peaks
        info = nk.ppg_findpeaks(clean_sig, sampling_rate=sampling_rate)
        
        if len(info['PPG_Peaks']) < 5: 
            return None
            
        # Extract HRV
        hrv_df = nk.hrv(info, sampling_rate=sampling_rate, show=False)
        
        if hrv_df.empty:
            return None
            
        features = hrv_df.iloc[0].to_dict()
        features.update(morph_features)
        
        # 성별 및 나이를 재귀적으로 탐색하여 추출 (깊이 숨겨진 JSON 대응)
        def find_val(obj, key):
            if isinstance(obj, dict):
                if key in obj: return obj[key]
                for v in obj.values():
                    res = find_val(v, key)
                    if res is not None: return res
            elif isinstance(obj, list):
                for item in obj:
                    res = find_val(item, key)
                    if res is not None: return res
            return None
            
        birthyear = find_val(data, 'birthyear')
        gender = find_val(data, 'gender')
        
        # 수집 시점을 감안하여 통상적인 기준으로 나이 계산 (결측치일 경우 None 처리)
        age = 2024 - birthyear if birthyear is not None else None
        
        # Clinical Features (Multi-modal)
        features['Clinical_GAD7'] = gad7_score
        features['Clinical_CSS'] = css_score
        features['Clinical_KOSS'] = koss_score
        features['Demographic_Age'] = age
        features['Demographic_Gender'] = gender
        
        features['Label'] = label
        features['FileName'] = os.path.basename(filepath)
        return features
        
    except Exception as e:
        return None

if __name__ == "__main__":
    json_files = glob.glob('data/*.json')
    print(f"Found {len(json_files)} json files. Starting extraction with CHROM...")
    
    dataset = []
    count = 0
    for f in json_files:
        res = extract_features_and_label(f)
        if res is not None:
            dataset.append(res)
        count += 1
        if count % 100 == 0:
            print(f"Processed {count}/{len(json_files)} files.")
            
    df = pd.DataFrame(dataset)
    df.to_csv("extracted_features.csv", index=False)
    print(f"\nExtraction complete. Successfully processed files: {len(df)}")
