import cv2
import mediapipe as mp
import numpy as np 
import csv
import os
import time

# --- MediaPipe Face Mesh 초기화 ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh_detector = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- 웹캠 초기화 ---
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("오류: 웹캠을 열 수 없습니다.")
    exit()

# --- 데이터 저장 설정 ---
output_dir = "emotion_data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

csv_filename = os.path.join(output_dir, "face_emotion_landmarks.csv")

# CSV 파일 헤더 준비 (478개 랜드마크 기준으로 변경)
csv_header = ['label']
for i in range(478): # MediaPipe Face Landmarker는 478개의 랜드마크를 가집니다.
    csv_header.extend([f'lm_{i}_x', f'lm_{i}_y', f'lm_{i}_z'])

# --- CSV 파일 생성 또는 열기 (항상 새로 생성하도록 수정) ---
if os.path.exists(csv_filename):
    os.remove(csv_filename)
    print(f"DEBUG: 기존 파일 '{csv_filename}' 삭제 완료.")

csv_file = open(csv_filename, 'w', newline='') # 'w' 모드로 열어 항상 새로 작성
csv_writer = csv.writer(csv_file)

csv_writer.writerow(csv_header) # 헤더를 항상 새로 쓰기

print(f"데이터가 '{csv_filename}'에 저장됩니다.")
print("---")
print("키를 눌러 데이터 수집: ")
print("  Q: Happy (행복)")
print("  W: Neutral (중립)")
print("  E: Surprise (놀람)")
print("  R: Sad (슬픔)")
print("  Space: 데이터 수집 중지/재개")
print("  Esc: 프로그램 종료")
print("---")

current_label = "None"
is_collecting = False
last_label_change_time = 0

# --- 메인 루프 시작 ---
while cap.isOpened():
    success, image_raw = cap.read() # 변수명을 image_raw로 변경
    if not success:
        print("빈 카메라 프레임을 무시합니다.")
        continue

    # --- MediaPipe에 전달하기 전에 이미지를 좌우 반전 ---
    processed_image = cv2.flip(image_raw.copy(), 1) 
    
    processed_image.flags.writeable = False
    rgb_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB) # image 대신 processed_image 사용
    results = face_mesh_detector.process(rgb_image)
    processed_image.flags.writeable = True # processed_image 쓰기 가능하도록 설정

    # --- 화면에 표시될 이미지는 processed_image를 다시 반전하여 거울처럼 보이게 함 ---
    display_image = cv2.flip(processed_image.copy(), 1) 
    
    # 디스플레이 텍스트
    status_text = f"Status: {'COLLECTING' if is_collecting and current_label != 'None' else 'PAUSED'}"
    label_text = f"Label: {current_label}"
    
    cv2.putText(display_image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(display_image, label_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # 랜드마크 그리기 (이 랜드마크는 이미 processed_image 기준이므로,
            # display_image에 그릴 때는 X좌표를 다시 반전시켜야 합니다.)
            h, w, c = display_image.shape
            for i, lm in enumerate(face_landmarks.landmark):
                cx = int((1 - lm.x) * w) # x 좌표를 1에서 빼서 반전
                cy = int(lm.y * h)
                cv2.circle(display_image, (cx, cy), 1, (255, 0, 0), -1)

            # 얼굴이 감지되면 데이터 수집 및 CSV 저장
            if is_collecting and current_label != "None":
                # --- 핵심 수정 부분: 랜드마크 좌표를 리스트 컴프리헨션으로 직접 추출 (478개 기준) ---
                if len(face_landmarks.landmark) != 478: # 랜드마크 개수가 478개가 아니면 무시
                    print(f"DEBUG: 감지된 랜드마크 개수가 478개가 아님 (무시): {len(face_landmarks.landmark)}")
                    continue 

                # 리스트 컴프리헨션을 사용하여 랜드마크 좌표 추출
                flat_features = [coord for lm in face_landmarks.landmark for coord in [lm.x, lm.y, lm.z]]
                
                row_data = [current_label]
                row_data.extend(flat_features) 

                # --- 데이터 쓰기 전에 특징 개수 최종 확인 (디버깅용) ---
                expected_num_features = 478 * 3 # 478 * 3 = 1434
                if len(flat_features) != expected_num_features:
                    print(f"DEBUG: 저장 전 최종 비정상적인 특징 개수 발견: {len(flat_features)} (예상: {expected_num_features})")
                    continue # 특징 개수가 다르면 저장하지 않고 건너뜁니다.
                
                csv_writer.writerow(row_data)

    cv2.imshow('Face Emotion Data Collector', display_image)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        current_label = "Happy"
        is_collecting = True
        last_label_change_time = time.time()
        print(f"Label set to: {current_label}. Collecting data...")
    elif key == ord('w'):
        current_label = "Neutral"
        is_collecting = True
        last_label_change_time = time.time()
        print(f"Label set to: {current_label}. Collecting data...")
    elif key == ord('e'):
        current_label = "Surprise"
        is_collecting = True
        last_label_change_time = time.time()
        print(f"Label set to: {current_label}. Collecting data...")
    elif key == ord('r'):
        current_label = "Sad"
        is_collecting = True
        last_label_change_time = time.time()
        print(f"Label set to: {current_label}. Collecting data...")
    elif key == ord(' '):
        is_collecting = not is_collecting
        print(f"Data collection {'RESUMED' if is_collecting else 'PAUSED'}")
    elif key == 27:
        break

# --- 자원 해제 ---
cap.release()
cv2.destroyAllWindows()
csv_file.close()
print(f"데이터 수집이 완료되었습니다. '{csv_filename}' 파일을 확인하세요.")