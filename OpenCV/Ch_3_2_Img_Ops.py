import numpy as np
import cv2

'''
 Image Operations
'''

# 새 영상 생성하기
img1 = np.empty((240, 320), dtype=np.uint8)       # grayscale image
img2 = np.zeros((240, 320, 3), dtype=np.uint8)    # color image
img3 = np.ones((240, 320), dtype=np.uint8) * 255  # white
img4 = np.full((240, 320, 3), (0, 255, 255), dtype=np.uint8)  # yellow

cv2.imshow('img1', img1)
cv2.imshow('img2', img2)
cv2.imshow('img3', img3)
cv2.imshow('img4', img4)
cv2.waitKey()
cv2.destroyAllWindows()


# 영상 복사
img1 = cv2.imread('./_imgs/HappyFish.jpg')

img2 = img1 # shallow copy
img3 = img1.copy() # deep copy

img1[:, :, :] = 255

cv2.imshow('img1', img1)
cv2.imshow('img2', img2)
cv2.imshow('img3', img3)
cv2.waitKey()
cv2.destroyAllWindows()

