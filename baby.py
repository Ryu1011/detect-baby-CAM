import time
import math
import cv2
import mediapipe as mp
import numpy as np
import os


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

isDragging = False                         # 마우스 드래그 상태 저장
x0,y0,w,h = -1,-1,-1,-1                    # 영역 선택 좌표 저장
blue,red = (255,0,0),(0,0,255)  

def onMouse(event, x, y, flags, param):     # 마우스 이벤트 핸들 함수
    global isDragging, x0, y0, w, h, img        # 전역 변수 참조
    if event == cv2.EVENT_LBUTTONDOWN:      # 왼쪽 마우스 버튼 다운, 드래그 시작
        isDragging = True
        x0 = x
        y0 = y
        
    elif event == cv2.EVENT_MOUSEMOVE:      # 마우스 움직임
        if isDragging:                       # 드래그 진행 중
            img_draw = img.copy()            # 사각형 그림 표현을 위한 이미지 복제 (매번 같은 이미지에 그려지면 이미지가 더러워짐)
            cv2.rectangle(img_draw, (x0,y0), (x,y), blue, 2)  # 드래그 진행 영역 표시
            cv2.imshow('roi', img_draw)       # 사각형으로 표시된 그림 화면 출력
            
    elif event == cv2.EVENT_LBUTTONUP:       # 왼쪽 마우스 버튼 업
        if isDragging:                        # 드래그 중지
            isDragging = False               
            w= x - x0                         # 드래그 영역 폭 계산
            h= y - y0                         # 드래그 영역 높이 계산
            print("x%d, y%d, w%d, h%d" % (x0, y0, w, h) )
            if w>0 and h>0:                  # 폭과 높이가 양수이면 드래그 방향이 옳음
                img_draw = img.copy()         # 선택 영역에 사각형 그림을 표시할 이미지 복제
                cv2.rectangle(img_draw, (x0, y0), (x, y), red, 2) # 선택 영역에 빨간색 사각형 표시
                cv2.imshow('roi', img_draw)         # 빨간색 사각형이 그려진 이미지 화면 출력
                roi = img[y0:y0+h, x0:x0+w]         # 원본 이미지에서 선택 영역만 ROI로 지정
                cv2.imshow('cropped', roi)          # ROI 지정 영역을 새 창으로 표시
                cv2.moveWindow('cropped', 0,0)     # 새 창을 화면 좌측 상단으로 이동
                print('cropped.')
            
            else:
            # 드래그 방향이 잘못된 경우 사각형 그림이 없는 원본 이미지 출력
                cv2.imshow('img', img)
                print('좌측 상단에서 우측 하단으로 영역을 드래그하세요.')

cap = cv2.VideoCapture(0)

if cap.isOpened():
    while True:
        time.sleep(2)
        ret, img2 = cap.read()
        if ret:
            cv2.namedWindow("roi", 1)
            cv2.setMouseCallback("roi", onMouse)
            img = cv2.flip(img2, 1)
            cv2.imshow("roi", img)
           
            if cv2.waitKey(0) == ord('c'):
                cv2.destroyWindow("roi")
                break
        else:
            print("no fram")
            break
else:
    print("can't open camera")






count = 0
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    pose_landmarks = results.pose_landmarks

    
    if pose_landmarks is not None:
        # Check the number of landmarks and take pose landmarks.
        assert len(pose_landmarks.landmark) == 33, 'Unexpected number of predicted pose landmarks: {}'.format(len(pose_landmarks.landmark))
        pose_landmarks = [[lmk.x, lmk.y, lmk.z] for lmk in pose_landmarks.landmark]
        frame_height, frame_width = image.shape[:2]
        pose_landmarks *= np.array([frame_width, frame_height, frame_width])

        nosex = image.shape[1] - pose_landmarks[0][0]
        nosey = pose_landmarks[0][1]
        left_heelx = image.shape[1] - pose_landmarks[29][0]
        left_heely = pose_landmarks[29][1]
        right_heelx = image.shape[1] - pose_landmarks[30][0]
        right_heely = pose_landmarks[30][1]
        mx = (left_heelx+ right_heelx)/2
        my = (left_heely+ right_heely)/2

        # print(nosex, nosey)
        # print(mx, my)
        # print((x0, y0), (x0+w, y0+h))
        radius = math.sqrt((mx-nosex)**2 + (my-nosey)**2)
        # print(radius)
        
        distance1 = math.sqrt((mx-x0)**2 + (my-y0)**2)
        distance2 = math.sqrt((mx-x0-w)**2 + (my-y0)**2)
        distance3 = math.sqrt((mx-x0)**2 + (my-y0-h)**2)
        distance4 = math.sqrt((mx-x0-w)**2 + (my-y0-h)**2)
        if distance1 < radius or distance2 < radius or distance3 < radius or distance1 < radius:
            print("nono")
            print(distance1)
            if count > 10:
                os.system('say "접근금지"')
                count = 0
            else: count += 1




        
        
    


    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    
    
    # Flip the image horizontally for a selfie-view display.
    image2 = cv2.flip(image, 1)
    cv2.rectangle(image2, (x0, y0), (x0+w, y0+h), red, 2)
    cv2.circle(image2, (int(mx), int(my)), int(radius), blue, 4)
    cv2.imshow("test", image2)
    
    if cv2.waitKey(5) & 0xFF == 27:
      break
   

        
    
cap.release()