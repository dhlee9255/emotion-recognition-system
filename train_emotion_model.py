import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib 
import os
import csv 
import numpy as np 

# --- 데이터 파일 경로 설정 ---
csv_filename = os.path.join("emotion_data", "face_emotion_landmarks.csv")
model_output_dir = "emotion_models" 
if not os.path.exists(model_output_dir):
    os.makedirs(model_output_dir)
model_filename = os.path.join(model_output_dir, "emotion_classifier_rf.joblib")
label_encoder_filename = os.path.join(model_output_dir, "label_encoder.joblib")


print(f"데이터 로딩 중: {csv_filename}")
labels = []
features = []
expected_labels = ['Happy', 'Neutral', 'Surprise', 'Sad']
valid_data_count = 0

try:
    with open(csv_filename, 'r', newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        header = next(csv_reader) # 첫 번째 줄은 헤더이므로 건너뜁니다.
        
        for row in csv_reader:
            if not row: # 빈 줄 건너뛰기
                continue
            
            label_str = row[0].strip() # 첫 번째 요소를 레이블로, 공백 제거
            
            # 예상되는 레이블 목록에 포함되는지 확인하여 필터링
            if label_str in expected_labels:
                try:
                    # 나머지 요소들을 float으로 변환하여 특징으로 사용
                    feature_row = [float(val) for val in row[1:]]
                    
                    # --- 여기서 정확한 특징 개수를 다시 한번 확인합니다 (478개 기준으로 변경) ---
                    if len(feature_row) == 478 * 3: # 478개 랜드마크 * 3좌표 (x, y, z) = 1434
                        labels.append(label_str)
                        features.append(feature_row)
                        valid_data_count += 1
                    else:
                        print(f"DEBUG: 로드 중 비정상적인 특징 개수 발견 (무시): 줄 {csv_reader.line_num} (개수: {len(feature_row)})")
                except ValueError as ve:
                    print(f"DEBUG: 로드 중 숫자 변환 오류 발생 (무시): 줄 {csv_reader.line_num}, 에러: {ve}")
            # else:
            #     print(f"DEBUG: 로드 중 예상치 못한 레이블 발견 (무시): 줄 {csv_reader.line_num}, 레이블: '{label_str}'")

    print(f"총 {valid_data_count}개의 유효한 데이터 포인트 로드됨.")

    # 유효한 데이터가 없으면 즉시 종료
    if valid_data_count == 0:
        print("\n오류: CSV 파일에서 유효한 데이터 포인트가 전혀 로드되지 않았습니다.")
        print("CSV 파일에 'Happy', 'Neutral', 'Surprise', 'Sad' 레이블과 478*3개의 특징이 올바르게 기록되어 있는지 확인하세요.")
        exit("유효한 데이터 부족으로 인한 학습 불가능.")

    # 리스트를 NumPy 배열로 변환
    X = np.array(features)
    y = np.array(labels)

except FileNotFoundError:
    print(f"오류: '{csv_filename}' 파일을 찾을 수 없습니다. 먼저 데이터 수집 코드를 실행해주세요.")
    exit()
except Exception as e:
    print(f"오류: CSV 파일 읽기 또는 파싱 중 예기치 않은 문제가 발생했습니다: {e}")
    print("CSV 파일의 내용이 올바른지 확인하거나, 'emotion_data' 폴더를 완전히 삭제 후 데이터를 재수집해보세요.")
    exit()

# --- 각 레이블별 데이터 개수 확인 ---
label_counts_filtered = pd.Series(y).value_counts()
print("\n필터링 및 파싱 후 각 레이블별 데이터 개수 확인:")
print(label_counts_filtered)

# 필터링 후에도 각 클래스에 충분한 데이터가 있는지 확인
if (label_counts_filtered < 2).any(): 
    problem_labels = label_counts_filtered[label_counts_filtered < 2].index.tolist()
    print(f"\n경고: 다음 레이블에 2개 미만의 데이터가 있습니다: {problem_labels}")
    print("이 레이블의 데이터를 더 수집하거나, CSV 파일에서 해당 행을 제거해야 합니다.")
    exit("데이터 불균형으로 인한 학습 불가능. 위에 명시된 레이블을 확인하세요.")
elif len(label_counts_filtered) == 0: 
    print("\n오류: 필터링 후 유효한 감정 레이블이 전혀 없습니다.")
    print("CSV 파일에 'Happy', 'Neutral', 'Surprise', 'Sad' 레이블이 올바르게 기록되어 있는지 확인하세요.")
    exit("유효한 데이터 부족으로 인한 학습 불가능.")

# --- 데이터 전처리 ---
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 학습 세트와 테스트 세트로 분리
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

print(f"\n학습 데이터 수: {len(X_train)}")
print(f"테스트 데이터 수: {len(X_test)}")
print(f"인식할 감정: {list(label_encoder.classes_)}")

# --- 모델 학습 ---
print("\nRandomForestClassifier 모델 학습 중...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("모델 학습 완료!")

# --- 모델 평가 ---
print("\n모델 평가 중...")
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"정확도 (Accuracy): {accuracy:.4f}")

print("\n분류 보고서:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# --- 모델 저장 ---
print(f"\n모델을 '{model_filename}'에 저장 중...")
joblib.dump(model, model_filename)
joblib.dump(label_encoder, label_encoder_filename)
print("모델과 레이블 인코더 저장 완료.")