import cv2
import mediapipe as mp
import math

# 손 인식 모델 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

# 모자이크 함수 정의
def mosaic(img, x, y, w, h, size=30):
    for i in range(int(w / size)):
        for j in range(int(h / size)):
            xi = x + i * size
            yi = y + j * size
            roi = img[yi:yi + size, xi:xi + size]
            if not roi.size == 0:  # ROI가 비어 있지 않은지 확인
                img[yi:yi + size, xi:xi + size] = cv2.blur(roi, (23, 23))

# 카메라 연결
cap = cv2.VideoCapture(0)

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
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            ring_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
            
            # 검지와 중지 손가락 사이의 거리 계산
            index_middle_distance = math.sqrt((index_finger_tip.x - middle_finger_tip.x)**2 + (index_finger_tip.y - middle_finger_tip.y)**2)
            # 검지와 중지 손가락의 각도 계산
            index_middle_angle = math.degrees(math.atan2(index_finger_tip.y - middle_finger_tip.y, index_finger_tip.x - middle_finger_tip.x))
            
            # 주먹을 쥐었는지 여부 확인 (각도가 일정 값 이하이고, 검지와 중지 손가락 사이의 거리가 일정 값 이하인 경우)
            if index_middle_angle > 130 and index_middle_distance < 0.03:
                # 손의 경계 상자 추정
                x_min, y_min = int(min(l.x * frame.shape[1] for l in hand_landmarks.landmark)), int(min(l.y * frame.shape[0] for l in hand_landmarks.landmark))
                x_max, y_max = int(max(l.x * frame.shape[1] for l in hand_landmarks.landmark)), int(max(l.y * frame.shape[0] for l in hand_landmarks.landmark))

                # 손에만 모자이크 적용
                mosaic(frame, x_min, y_min, x_max - x_min, y_max - y_min)

    # 화면 표시
    cv2.imshow('Hand Gesture Mosaic', frame)

    # 종료 키 설정
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()
