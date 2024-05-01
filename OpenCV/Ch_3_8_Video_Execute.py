import sys
import cv2


# 비디오 파일 열기
cap = cv2.VideoCapture('video1.mp4')

if not cap.isOpened():
    print("Video open failed!")
    sys.exit()


# 비디오 프레임 크기, 전체 프레임수, FPS 등 출력
# get() 함수는 값을 float 형태로 반환한다.
print('Frame width:', int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
print('Frame height:', int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print('Frame count:', int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

# frame per second
fps = cap.get(cv2.CAP_PROP_FPS)
print('FPS:', fps) # 24 frame/sec

# frame 간 시간 계산 : 1000msec / 24frame/sec
delay = round(1000 / fps)

# 비디오 매 프레임 처리
while True:
    ret, frame = cap.read()
    #--> 동영상 재생이 끝나면 False, None을 반환하게 된다.

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame', frame)
    cv2.imshow('gray', gray)

    # waitKey()에 인수를 입력하지 않으면
    # 첫 frame에서 다음 frame으로 넘어가지 못한다.
    if cv2.waitKey(delay) == 27:
        break

cap.release()
cv2.destroyAllWindows()
