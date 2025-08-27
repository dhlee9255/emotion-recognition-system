import cv2
import mediapipe as mp
import numpy as np
import time
import joblib  # 학습된 모델 로드를 위해 추가
import os

# --- 모델 파일 경로 설정 ---
MODEL_DIR = "emotion_models"
MODEL_FILENAME = os.path.join(MODEL_DIR, "emotion_classifier_rf.joblib")
LABEL_ENCODER_FILENAME = os.path.join(MODEL_DIR, "label_encoder.joblib")

# --- 학습된 모델 및 레이블 인코더 로드 ---
emotion_model = None
label_encoder = None
try:
    emotion_model = joblib.load(MODEL_FILENAME)
    label_encoder = joblib.load(LABEL_ENCODER_FILENAME)
    print(f"DEBUG: 감정 분류 모델 '{MODEL_FILENAME}' 로드 성공.")
    print(f"DEBUG: 인식 가능한 감정: {list(label_encoder.classes_)}")
except FileNotFoundError:
    print(f"오류: 감정 분류 모델 또는 레이블 인코더 파일을 찾을 수 없습니다.")
    print(f"'{MODEL_DIR}' 폴더에 '{os.path.basename(MODEL_FILENAME)}'과 '{os.path.basename(LABEL_ENCODER_FILENAME)}'이 있는지 확인해주세요.")
    print("먼저 'train_emotion_model.py'를 실행하여 모델을 학습시키세요.")
    exit()  # 모델 없으면면

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

# --- 눈 감김 감지 관련 변수 (기존과 동일) ---
LEFT_EYE_TOP_LM_IDX = 159
LEFT_EYE_BOTTOM_LM_IDX = 145
RIGHT_EYE_TOP_LM_IDX = 386
RIGHT_EYE_BOTTOM_LM_IDX = 374

EYE_CLOSED_THRESHOLD_SQUARED = 35
EYES_CLOSED_DURATION_SECONDS = 5.0

both_eyes_closed_start_time = None
sleep_mode_active = False

def calculate_squared_distance(p1, p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2
# --- 감정 계산 (학습된 모델 사용 ---
def get_emotion(face_landmarks_normalized_object):
   
    if emotion_model is None or label_encoder is None:
        return "Model_Error"

    # --- 랜드마크 개수 검증 -
    if len(face_landmarks_normalized_object) != 478:
        return "Landmark_Count_Mismatch" 

    # 모든 랜드마크의 x, y, z를 추출하여 1차원 배열로 평탄화 (478 * 3 = 1434 특징)
    flat_features = [coord for lm in face_landmarks_normalized_object for coord in [lm.x, lm.y, lm.z]]

    # 모델 입력 형태에 맞게 2D 배열로 변환 (1개의 샘플, N개의 특징)
    features_array = np.array(flat_features).reshape(1, -1)

    try:
        # 모델로 감정 예측
        prediction_encoded = emotion_model.predict(features_array)
        # 예측된 숫자 레이블을 다시 문자열 감정으로 변환
        predicted_emotion = label_encoder.inverse_transform(prediction_encoded)[0]
        return predicted_emotion
    except Exception as e:
        print(f"AI (Emotion Prediction Error): {e}")
        return "Prediction_Error"

# --- 메인 루프 시작 ---
if __name__ == "__main__":
    while cap.isOpened():
        success, image_raw = cap.read()
        if not success:
            print("빈 카메라 프레임을 무시합니다.")
            continue

        image_height, image_width, _ = image_raw.shape
        screen_center_x = image_width // 2
        screen_center_y = image_height // 2

        # --- MediaPipe에 전달하기 전에 이미지를 좌우 반전 (거울 모드) ---
        processed_image = cv2.flip(image_raw.copy(), 1)

        processed_image.flags.writeable = False
        rgb_image_orig = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        results = face_mesh_detector.process(rgb_image_orig)
        processed_image.flags.writeable = True

        face_center_x_offset_orig = 0
        face_center_y_offset_orig = 0
        face_center_pixel_orig = None

        left_eye_dist_sq_str = "L_dist: N/A"
        right_eye_dist_sq_str = "R_dist: N/A"
        closed_time_str = "Closed Time: 0.0s"
        current_emotion = "No Face"

        current_face_landmarks_pixel_orig = []  # 디스플레이용 픽셀 좌표 랜드마크
        current_face_landmarks_normalized_coords = []  # 모델 입력용 정규화된 좌표 랜드마크

        if results.multi_face_landmarks:
            for face_landmarks_normalized in results.multi_face_landmarks:
                current_face_landmarks_normalized_coords = face_landmarks_normalized.landmark

                current_face_landmarks_pixel_orig.clear()
                for idx, lm_normalized in enumerate(face_landmarks_normalized.landmark):
                    x_pixel_orig = int(lm_normalized.x * image_width)
                    y_pixel_orig = int(lm_normalized.y * image_height)
                    current_face_landmarks_pixel_orig.append([x_pixel_orig, y_pixel_orig])

                if not current_face_landmarks_pixel_orig:
                    continue

                # --- 1a. 원본 이미지 기준 얼굴 중심 및 오프셋 계산 ---
                if len(current_face_landmarks_pixel_orig) > 1:
                    # 얼굴 중심점은 MediaPipe가 제공하는 코 끝(랜드마크 1)을 사용
                    face_center_normalized_x = face_landmarks_normalized.landmark[1].x
                    face_center_normalized_y = face_landmarks_normalized.landmark[1].y

                    # 픽셀 좌표로 변환 (MediaPipe에 입력된 processed_image 기준)
                    face_center_pixel_orig_x = int(face_center_normalized_x * image_width)
                    face_center_pixel_orig_y = int(face_center_normalized_y * image_height)
                    face_center_pixel_orig = [face_center_pixel_orig_x, face_center_pixel_orig_y]

                    face_center_x_offset_orig = -(face_center_pixel_orig_x - screen_center_x)
                    face_center_y_offset_orig = -(face_center_pixel_orig_y - screen_center_y)
                    
                # --- 감정 인식 함수 호출 (학습된 모델 사용) ---
                current_emotion = get_emotion(current_face_landmarks_normalized_coords)

                # --- 1b. 원본 이미지 기준 눈 감김 감지 로직 (기존과 동일) ---
                required_eyes_indices = [LEFT_EYE_TOP_LM_IDX, LEFT_EYE_BOTTOM_LM_IDX,
                                         RIGHT_EYE_TOP_LM_IDX, RIGHT_EYE_BOTTOM_LM_IDX]

                if all(idx < len(current_face_landmarks_pixel_orig) for idx in required_eyes_indices):
                    left_eye_pt_top = current_face_landmarks_pixel_orig[LEFT_EYE_TOP_LM_IDX]
                    left_eye_pt_bottom = current_face_landmarks_pixel_orig[LEFT_EYE_BOTTOM_LM_IDX]
                    left_eye_dist_sq = calculate_squared_distance(left_eye_pt_top, left_eye_pt_bottom)
                    left_eye_dist_sq_str = f"L_dist: {left_eye_dist_sq:.0f}"

                    right_eye_pt_top = current_face_landmarks_pixel_orig[RIGHT_EYE_TOP_LM_IDX]
                    right_eye_pt_bottom = current_face_landmarks_pixel_orig[RIGHT_EYE_BOTTOM_LM_IDX]
                    right_eye_dist_sq = calculate_squared_distance(right_eye_pt_top, right_eye_pt_bottom)
                    right_eye_dist_sq_str = f"R_dist: {right_eye_dist_sq:.0f}"

                    if left_eye_dist_sq < EYE_CLOSED_THRESHOLD_SQUARED and \
                            right_eye_dist_sq < EYE_CLOSED_THRESHOLD_SQUARED:
                        if both_eyes_closed_start_time is None:
                            print("[DEBUG] 양쪽 눈 감김 시작!")
                            both_eyes_closed_start_time = time.time()

                        current_closed_time = time.time() - both_eyes_closed_start_time

                        if not sleep_mode_active and current_closed_time >= EYES_CLOSED_DURATION_SECONDS:
                            print("[DEBUG] SLEEP 모드 활성화!")
                            sleep_mode_active = True
                    else:  # 한쪽 눈이라도 뜨면
                        if both_eyes_closed_start_time is not None:
                            print("[DEBUG] 눈 뜸, 타이머 리셋.")
                        both_eyes_closed_start_time = None
                        if sleep_mode_active:
                            print("[DEBUG] SLEEP 모드 비활성화 (눈 뜸).")
                        sleep_mode_active = False
                else:  # 눈 감지용 랜드마크 부족
                    if both_eyes_closed_start_time is not None or sleep_mode_active:
                        print("[DEBUG] 눈 감지용 랜드마크 부족, SLEEP 타이머/모드 리셋.")
                    both_eyes_closed_start_time = None
                    sleep_mode_active = False
                    left_eye_dist_sq_str = "L_dist: N/A"
                    right_eye_dist_sq_str = "R_dist: N/A"
        else:  # 얼굴 미감지
            if both_eyes_closed_start_time is not None or sleep_mode_active:
                print("[DEBUG] 얼굴 미감지, SLEEP 타이머/모드 리셋.")
            both_eyes_closed_start_time = None
            sleep_mode_active = False
            left_eye_dist_sq_str = "L_dist: N/A"
            right_eye_dist_sq_str = "R_dist: N/A"
            current_emotion = "No Face"

        # 2. 출력용 이미지 준비 (MediaPipe에 입력된 이미지를 다시 좌우 반전하여 거울처럼 만듦)
        image_to_display = cv2.flip(processed_image.copy(), 1)

        # 3. 출력용 이미지(image_to_display)에 그리기
        # 화면 중심점 그리기 (출력용 이미지의 중앙)
        cv2.circle(image_to_display, (screen_center_x, screen_center_y), 5, (0, 0, 255), cv2.FILLED)

        # 얼굴 중심점 그리기 (MediaPipe 처리 후 반전된 이미지에 맞게 변환하여 그림)
        if face_center_pixel_orig is not None:
            flipped_face_center_x_for_display = image_width - 1 - face_center_pixel_orig[0]
            cv2.circle(image_to_display, (flipped_face_center_x_for_display, face_center_pixel_orig[1]), 5, (0, 255, 0), cv2.FILLED)

        # 오프셋 값 텍스트 출력 (값은 MediaPipe에 입력된 processed_image 기준)
        cv2.putText(image_to_display, f"Offset X (orig): {face_center_x_offset_orig}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(image_to_display, f"Offset Y (orig): {face_center_y_offset_orig}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # 눈 감김 디버깅 정보 텍스트 출력 (랜드마크 좌표는 MediaPipe 기준)
        cv2.putText(image_to_display, left_eye_dist_sq_str, (10, image_height - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 200), 2)
        cv2.putText(image_to_display, right_eye_dist_sq_str, (10, image_height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 200), 2)

        # 눈 감김 시간 실시간 업데이트 및 표시
        if both_eyes_closed_start_time is not None:
            current_closed_time_for_display = time.time() - both_eyes_closed_start_time
            closed_time_color = (0, 0, 255) if current_closed_time_for_display >= EYES_CLOSED_DURATION_SECONDS else (0, 200, 0)
            cv2.putText(image_to_display, f"Closed Time: {current_closed_time_for_display:.1f}s", (10, image_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, closed_time_color, 2)
        else:
            cv2.putText(image_to_display, "Closed Time: 0.0s", (10, image_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 200), 2)

        # 감정 상태 텍스트 출력
        emotion_display_text = f"Emotion: {current_emotion}"
        emotion_color = (0, 255, 0)  # 기본 초록색
        if current_emotion == "Happy":
            emotion_color = (0, 255, 255)  # 노란색
        elif current_emotion == "Surprise":
            emotion_color = (255, 100, 0)  # 하늘색
        elif current_emotion == "Sad":
            emotion_color = (255, 0, 255)  # 마젠타
        elif current_emotion in ["No Face", "Prediction_Error", "Model_Error", "Landmark_Count_Mismatch"]:
            emotion_color = (0, 0, 255)  # 빨간색

        cv2.putText(image_to_display, emotion_display_text, (image_width - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, emotion_color, 2)

        # "SLEEP" 모드 텍스트 출력
        if sleep_mode_active:
            sleep_text = "SLEEP"
            font_face = cv2.FONT_HERSHEY_TRIPLEX
            font_scale = 4
            thickness = 6
            text_size, _ = cv2.getTextSize(sleep_text, font_face, font_scale, thickness)
            text_x = (image_width - text_size[0]) // 2
            text_y = (image_height + text_size[1]) // 2
            cv2.putText(image_to_display, sleep_text, (text_x, text_y), font_face, font_scale, (255, 255, 255), thickness + 4, cv2.LINE_AA)
            cv2.putText(image_to_display, sleep_text, (text_x, text_y), font_face, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        # 최종적으로 처리된 출력용 이미지를 화면에 표시 (이미 좌우 반전됨)
        cv2.imshow('facetrackingforfinal', image_to_display)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# --- 자원 해제 ---
cap.release()
cv2.destroyAllWindows()