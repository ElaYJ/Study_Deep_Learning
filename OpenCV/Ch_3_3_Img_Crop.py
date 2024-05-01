import numpy as np
import cv2


# 부분 영상 추출
img1 = cv2.imread('./_imgs/HappyFish.jpg')

# numpy.ndarray의 슬라이싱 기법 사용
img2 = img1[40:120, 30:150] #--> 참조 상태
img3 = img1[40:120, 30:150].copy()

img2.fill(0)

cv2.imshow('img1', img1)
cv2.imshow('img2', img2)
cv2.imshow('img3', img3)
cv2.waitKey()
cv2.destroyAllWindows()


# 부분 영상 처리
img = cv2.imread('./_imgs/lenna.bmp', cv2.IMREAD_GRAYSCALE)

img_face = img[200:400, 200:400]  # 얼굴 영역

cv2.imshow('img', img)
cv2.imshow('img_face', img_face)
cv2.waitKey()
cv2.destroyAllWindows()

cv2.add(img_face, 50, img_face)   # 밝기 조절
#--> (input, add_value, output)

cv2.imshow('img', img)
cv2.imshow('img_face', img_face)
cv2.waitKey()
cv2.destroyAllWindows()

cv2.circle(img=img_face, center=(100,100), radius=80, color=0, thickness=2)   # 원 그리기
#--> def circle(
#		img: UMat, 
# 		center: cv2.typing.Point, 
# 		radius: int, 
# 		color: cv2.typing.Scalar, (color=0 --> black)
# 		thickness: int=..., (pixel unit)
# 		lineType: int=..., 
# 		shift: int=...
#	 ) -> UMat

cv2.imshow('img', img)
cv2.imshow('img_face', img_face)
cv2.waitKey()
cv2.destroyAllWindows()
