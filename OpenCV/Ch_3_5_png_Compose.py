import sys
import cv2

'''
 Image Compose(합성)
'''


# 알파 채널을 마스크 영상으로 이용
src = cv2.imread('./_imgs/cat.bmp', cv2.IMREAD_COLOR)
logo = cv2.imread('./_imgs/opencv-logo-white.png', cv2.IMREAD_UNCHANGED) # alpha channel 포함 --> ndim=4

if src is None or logo is None:
    print('Image load failed!')
    sys.exit()

mask = logo[:, :, 3]    # mask는 알파 채널로 만든 마스크 영상
logo = logo[:, :, 0:3]  # logo는 b, g, r 3채널로 구성된 컬러 영상

print(mask.shape) #--> (222, 180)
cv2.imshow('mask', mask)
cv2.waitKey()
cv2.destroyAllWindows()

h, w = mask.shape[:2]
crop = src[10:10+h, 20:20+w]  # logo, mask와 같은 크기의 부분 영상 추출

cv2.copyTo(logo, mask, crop)
#crop[mask > 0] = logo[mask > 0]

cv2.imshow('src', src)
cv2.imshow('logo', logo)
cv2.imshow('mask', mask)
cv2.waitKey()
cv2.destroyAllWindows()
