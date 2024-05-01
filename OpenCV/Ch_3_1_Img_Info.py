import sys
import cv2


# 영상 불러오기
img1 = cv2.imread('./_imgs/cat.bmp', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('./_imgs/cat.bmp', cv2.IMREAD_COLOR)

if img1 is None or img2 is None:
    print('Image load failed!')
    sys.exit()


# 영상의 속성 참조
print('type(img1):', type(img1)) # type(img1): <class 'numpy.ndarray'>
print('img1.shape:', img1.shape) # img1.shape: (480, 640)
print('img2.shape:', img2.shape) # img2.shape: (480, 640, 3)
print('img1.dtype:', img1.dtype) # img1.dtype: uint8


# 영상의 크기 참조
h, w = img1.shape
print('img1 size: {} x {}'.format(w, h)) # img1 size: 640 x 480

h, w = img2.shape[:2]
print('img2 size: {} x {}'.format(w, h)) # img2 size: 640 x 480

# Gray 영상과 Color 영상 구분 방법
if len(img1.shape) == 2:
    print('img1 is a grayscale image')
elif len(img1.shape) == 3:
    print('img1 is a truecolor image')

cv2.imshow('img1', img1)
cv2.imshow('img2', img2)
cv2.waitKey()


# 영상의 픽셀 값 참조
'''
for y in range(h):
    for x in range(w):
        img1[y, x] = 255
        img2[y, x] = (0, 0, 255) # BGR
'''
img1[:,:] = 255
img2[:,:] = (0, 0, 255)

cv2.imshow('img1', img1)
cv2.imshow('img2', img2)
cv2.waitKey()

cv2.destroyAllWindows()
