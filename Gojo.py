import cv2
import mediapipe as mp
import math
import time

# 손 인식 모델 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

# 카메라 연결
cap = cv2.VideoCapture(0)

# 동영상 파일 경로
video_path = "C:/Users/User/Downloads/video.mp4"

# 동영상 재생 여부를 나타내는 변수
video_playing = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # 화면 반전
    frame = cv2.flip(frame, 1)

    # 손 인식
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 검지와 중지 손가락 끝 좌표 추출
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            
            # 검지와 중지 손가락 사이의 거리 계산
            index_middle_distance = math.sqrt((index_finger_tip.x - middle_finger_tip.x)**2 + (index_finger_tip.y - middle_finger_tip.y)**2)
            
            # 주먹을 쥐었는지 여부 확인 (검지와 중지 손가락 사이의 거리가 일정 값 이하인 경우)
            if index_middle_distance < 0.03:
                if not video_playing:
                    # 동영상 재생 시작
                    video = cv2.VideoCapture(video_path)
                    video_playing = True
                    start_time = time.time()  # 동영상 재생 시작 시간 기록
                else:
                    # 현재 시간과 동영상 재생 시작 시간의 차이를 계산하여 일정 시간이 지나면 동영상 종료
                    if time.time() - start_time > 5:  # 예: 5초 후에 동영상 종료
                        video_playing = False
                        video.release()  # 동영상 파일 해제
            else:
                # 주먹을 쥐지 않은 상태이므로 동영상 재생 중이라면 종료
                if video_playing:
                    video_playing = False
                    video.release()  # 동영상 파일 해제

    # 동영상 재생 중이면 프레임 합성
    if video_playing:
        ret, video_frame = video.read()
        if ret:
            # 웹캠 프레임과 동영상 프레임 크기 맞추기
            video_frame = cv2.resize(video_frame, (frame.shape[1], frame.shape[0]))

            # 웹캠 프레임과 동영상 프레임 합성
            combined_frame = cv2.addWeighted(frame, 0.7, video_frame, 0.3, 0)
            cv2.imshow('Hand Gesture Video', combined_frame)
        else:
            # 동영상 재생이 끝나면 종료
            video_playing = False
            video.release()  # 동영상 파일 해제
    else:
        cv2.imshow('Hand Gesture Video', frame)

    # 종료 키 설정
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()
