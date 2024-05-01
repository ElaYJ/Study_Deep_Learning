import sys
import cv2


img = cv2.imread('./_imgs/cat.bmp', flags=cv2.IMREAD_GRAYSCALE)
#--> img type: numpy.ndarray

if img is None:
    print('image load failed!')
    sys.exit()

if cv2.imwrite('./_imgs/cat_gray.png', img):
    print('Save Complete~!!')
else:
    sys.exit()
    
cv2.namedWindow('image') #, flags=cv2.WINDOW_NORMAL)
cv2.imshow('image', img) # winname: str, mat: UMat(matrix)
cv2.waitKey() #--> 키보드 입력이 있을 때까지 화면에 영상 데이터를 출력

cv2.destroyWindow('image')
# cv2.destroyAllWindows()