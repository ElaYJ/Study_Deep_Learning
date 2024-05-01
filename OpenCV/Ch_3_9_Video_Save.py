import sys
import cv2


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera open failed!")
    sys.exit()

w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'DIVX') # *'DIVX' == 'D', 'I', 'V', 'X'
delay = round(1000 / fps)

print(f" Width: {w}\n Height: {h}\n FPS: {fps}\n delay: {delay}\n")

out = cv2.VideoWriter('output.avi', fourcc, fps, (w, h))

if not out.isOpened():
    print('File open failed!')
    cap.release()
    sys.exit()

while True:
    ret, frame = cap.read()

    if not ret:
        break
    
    # case_1. Basic
    # out.write(frame)
    # cv2.imshow('frame', frame)
    
    # case_2.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(gray, 50, 150)
    
    edge_bgr = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
    out.write(edge_bgr)

    cv2.imshow('frame', frame)
    cv2.imshow('gray', gray)
    cv2.imshow('edge', edge)
    cv2.imshow('edge_bgr', edge_bgr)
    

    if cv2.waitKey(delay) == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
