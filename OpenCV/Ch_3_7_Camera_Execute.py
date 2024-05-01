import sys
import cv2


# 카메라 열기
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print('camera open failed!')
    sys.exit()
else:
    print('camera open succeeded!')


# 카메라 프레임 처리
while True:
    ret, frame = cap.read()
    #--> return value(True or False) & image(numpy.ndarray)

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame', frame)
    cv2.imshow('gray', gray)

    # delay=int...(millisecond)
    if cv2.waitKey(delay=1) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
