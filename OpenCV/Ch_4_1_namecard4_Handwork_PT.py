import sys
import numpy as np
import cv2

'''
 Handwork Perspective Transform
 수동 투시 변환
'''

# 영상 불러오기
src = cv2.imread('./_imgs/namecard1.jpg')

if src is None:
    print('Image load failed!')
    sys.exit()

# 출력 영상 설정 - Handwork
w, h = 720, 400
srcQuad = np.array([[324, 308], [760, 369], [718, 611], [231, 517]], np.float32)
dstQuad = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], np.float32)
dst = np.zeros((h, w), np.uint8)

# 투시 변환 행렬 구하기
'''
def getPerspectiveTransform(
        src: UMat, (4개의 원본 좌표점 --> numpy.ndarray. shape=(4, 2). e.g> np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.float32))
        dst: UMat, (4개의 결과 좌표점 --> numpy.ndarray. shape=(4, 2).)
        solveMethod: int = ...
    ) -> cv2.typing.MatLike: ... (3x3 크기의 투시 변환 행렬 반환)
'''
pers = cv2.getPerspectiveTransform(srcQuad, dstQuad)

# 영상 투시 변환 수행
'''
def warpPerspective(
        src: UMat, (입력 영상)
        M: UMat, (3x3 변환 행렬 --> numpy.ndarray)
        dsize: cv2.typing.Size, (결과 영상의 크기. (0, 0)을 지정하면 src와 같은 크기)
        dst: UMat | None = ..., (출력 영상)
        flags: int = ..., (보간법. 기본값은 cv2.INTER_LINEAR)
        borderMode: int = ..., (가장자리 픽셀 확장 방식)
        borderValue: cv2.typing.Scalar = ... (cv2.BORDER_CONSTANT일때 사용할 상수값. 기본값은 0)
    ) -> UMat: ...
'''
dst = cv2.warpPerspective(src, pers, (w, h))

cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()
