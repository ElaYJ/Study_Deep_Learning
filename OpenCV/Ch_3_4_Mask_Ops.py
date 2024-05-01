import sys
import cv2

'''
 Mask Operations
'''


# 마스크 영상을 이용한 영상 합성
src = cv2.imread('./_imgs/airplane.bmp', cv2.IMREAD_COLOR)
mask = cv2.imread('./_imgs/mask_plane.bmp', cv2.IMREAD_GRAYSCALE)
dst = cv2.imread('./_imgs/field.bmp', cv2.IMREAD_COLOR)

if src is None or mask is None or dst is None:
    print('Image load failed!')
    sys.exit()

cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.imshow('mask', mask)
cv2.waitKey()
cv2.destroyAllWindows()

cv2.copyTo(src, mask, dst)
# dst[mask > 0] = src[mask > 0]

cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.imshow('mask', mask)
cv2.waitKey()
cv2.destroyAllWindows()
