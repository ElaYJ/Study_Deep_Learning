import sys
import cv2


img = cv2.imread('cat.bmp')

if img is None:
    print('image load failed!')
    sys.exit()
    
cv2.namedWindow('image')
cv2.imshow('image', img) # winname: str, mat: UMat
cv2.waitKey() #--> 키보드 입력이 있을 때까지 화면에 영상 데이터를 출력
cv2.destroyWindow('image')