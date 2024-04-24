import cv2
import mediapipe as mp
import math
import time

# 손 인식 모델 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

# 얼굴 인식을 위한 분류기 로드
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 카메라 연결
cap = cv2.VideoCapture(0)

# 동영상 파일 경로
video_path = "C:\\Users\\User\\Downloads\\aa.mp4"

# 이미지 불러오기
image_path = 'C:\\Users\\User\\Downloads\\c3251ca67fdc16fb7884451e63e7a5c8-removebg-preview.png'
overlay_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

# 이미지 크기 조정
new_width = 300  # 원하는 너비
new_height = 300  # 원하는 높이
overlay_image = cv2.resize(overlay_image, (new_width, new_height))

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
        # 그레이스케일 변환
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 얼굴 감지
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # 얼굴 주변에 이미지 오버레이
        for (x, y, w, h) in faces:
            # 얼굴 주변에 이미지 크기 조정
            resized_overlay = cv2.resize(overlay_image, (w*2, h*2))
            
            # 오버레이 이미지가 프레임을 벗어나지 않도록 처리
            y0, y1 = max(0, y - h//2), min(frame.shape[0], y + h + h//2)
            x0, x1 = max(0, x - w//2), min(frame.shape[1], x + w + w//2)
            
            # 프레임 영역 계산
            face_frame = frame[y0:y1, x0:x1]
            
            # 오버레이 이미지 조정
            overlay_resized = cv2.resize(resized_overlay, (x1-x0, y1-y0))
            
            # 이미지 오버레이
            for c in range(0, 3):
                alpha_mask = overlay_resized[:, :, 3] / 255.0
                frame[y0:y1, x0:x1, c] = (1.0 - alpha_mask) * frame[y0:y1, x0:x1, c] + alpha_mask * overlay_resized[:, :, c]

        cv2.imshow('Hand Gesture Video', frame)

    # 종료 키 설정
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()
