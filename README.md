# emotion-recognition-system
****
## 해당 프로젝트는 케어로봇의 시각시스템에 넣을 감정인식 시스템을 구현하고자 하는데에 목적이 있다.
<br>
//2025-1 피지컬컴퓨팅 기말과제를 위해 제작되었다.
<br>
//emotion_models은 임시로 넣어두었고 emotion_data는 파일이 너무 커서 넣어두진 않았다.
<br>

<img width="455" height="609" alt="Image" src="https://github.com/user-attachments/assets/438b87c1-4c57-45c5-b53a-ceba9cef12c9" />


<br>
<br>
<br>

## 데이터 수집단계
### **collect_emotion_data.py** 를 실행시켜 감정 데이터를 직접 레이블링하여 습득합니다.

"q": "Happy", "w": "Neutral", "e": "Surprise" , "r": "Sad" , "스페이스바": 학습중단 , "esc": "프로그램 종료"

각 해당하는 표정을 지으며, 레이블링을 직접 할수 있다. 

다음과 같이 **emotion_data** 폴더에 **face_emotion_landmarks.csv**파일로 저장이 된다.

`label,lm_0_x,lm_0_y,lm_0_z,.....lm_477_z,
Neutral,0.5209217071533203,0.6032217144966125,-0.0389031246304512, ......`

(다음과 같은 형식으로 mediapipe의 face_mesh의 각 특징점들의 xyz좌표값들이 레이블과 함꼐 프레임 단위로 한줄에 저장된다.)

<br>
<br>

## 모델 생성 및 학습
### **train_emotion_model.py** 를 실행시켜 습득한 얼굴 특징점 정보를 가지고 학습시킨 모델을 생성한다.

해당 제공 코드에선 RandomForestClassifier를 사용하였다. 

학습된 모델은 **emotion_models**폴더에 **emotion_classifier_rf.joblib ,label_encoder.joblib** 로저장된다.

<br>
<br>

## 최종 결과물 테스트
### **main_tracking_with_ml_emotion.py**를 실행시켜 모델을 사용해 감정인식을 해볼수 있다. 

(추가적으로 눈을 5초이상 감으면 sleep을 출력하도록 하였다.)
<br>
<img width="387" height="424" alt="Image" src="https://github.com/user-attachments/assets/815b3f8b-c9fa-4250-b1b9-fc7c25ba2f39" />
<br>
<예시 실행 사진입니다.>
<br>
<br>
#### 개선방향
*전체 윈도우에서의 특징점 좌표를 계산하기 때문에 데이터 수집 단계에서 얼굴의 위치를 직접 다양하게 만들어 줘야 정상적인 학습이됨.
이는 비효율, 비정확하기 때문에 랜드마크를 감싸는 바운딩 박스를 만들고 바운딩 박스 내부를 기준으로 좌표계를 새로 구성하여 수집을 하고 학습을 시키면

*조금은 더 나은 성능을 보여줄것으로 예상됨.
카메라 각도에도 영향을 많이 받기 때문에 개선의 여지가 있음
