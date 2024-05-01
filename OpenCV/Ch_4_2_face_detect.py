import sys
import numpy as np
import cv2

# Caffe 에서 학습된 모델 및 설정 파일
model = 'opencv_face_detector/res10_300x300_ssd_iter_140000_fp16.caffemodel'
config = 'opencv_face_detector/deploy.prototxt'

# TensorFlow 에서 학습된 모델 및 설정 파일
# model = 'opencv_face_detector/opencv_face_detector_uint8.pb'
# config = 'opencv_face_detector/opencv_face_detector.pbtxt'

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print('Camera open failed!')
    sys.exit()

net = cv2.dnn.readNet(model, config)

if net.empty():
    print('Net open failed!')
    sys.exit()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    '''
    def blobFromImage(
            image: cv2.UMat, 
            scalefactor: float = ..., (1이면 0~255 사용을 의미)
            size: cv2.typing.Size = ..., (모델에 입력되는 이미지 크기)
            mean: cv2.typing.Scalar = ..., (평균 픽셀 값)
            swapRB: bool = ..., crop: bool = ..., ddepth: int = ...
        ) -> cv2.typing.MatLike: ...
    '''
    blob = cv2.dnn.blobFromImage(frame, 1, (300, 300), (104, 177, 123))
    net.setInput(blob)

    out = net.forward()
    detect = out[0, 0, :, :]

    (h, w) = frame.shape[:2]

    for i in range(detect.shape[0]):
        confidence = detect[i, 2]
        if confidence < 0.5:
            break

        x1 = int(detect[i, 3] * w)
        y1 = int(detect[i, 4] * h)
        x2 = int(detect[i, 5] * w)
        y2 = int(detect[i, 6] * h)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0))

        label = f'Face: {confidence:4.2f}'
        cv2.putText(frame, label, (x1, y1 - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
