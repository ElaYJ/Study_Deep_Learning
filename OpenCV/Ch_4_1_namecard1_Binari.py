import sys
import random
import cv2


# 영상 불러오기
src = cv2.imread('./_imgs/namecard1.jpg')

if src is None:
    print('Image load failed!')
    sys.exit()

# 입력 영상을 그레이스케일 영상으로 변환
src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

# 자동 이진화(binarization)
th, src_bin = cv2.threshold(
    src=src_gray, thresh=0, maxval=255, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU
)
print(th)

cv2.imshow('src', src)
cv2.imshow('src_gray', src_gray)
cv2.imshow('src_bin', src_bin)
cv2.waitKey()
cv2.destroyAllWindows()
