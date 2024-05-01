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


# 자동 이진화(Binarization)
'''
def threshold(
        src: UMat, thresh: float, maxval: float, type: int, dst: UMat | None = ...
    ) -> tuple[float, UMat]: ... (사용된 임계값, (출력)입계값 영상(src와 동일크기, 동일타입))
'''
_, src_bin = cv2.threshold(src=src_gray, thresh=0, maxval=255, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)


# 외곽선 검출 및 명함 검출
'''
def findContours(
        image: UMat, (이진화된 이미지)
        mode: int, (cv2.RETR_EXTERNAL--> 바깥쪽 외곽선만 검출, cv2.RETR_LIST--> 모든 외곽선 검출, etc.)
        method: int, (외곽선 근사화 방법 제시, cv2.CHAIN_APPROX_NONE--> 보통 근사화하지 않는다.)
        contours: _typing.Sequence[UMat] | None = ..., 
        hierarchy: UMat | None = ..., 
        offset: cv2.typing.Point = ...
    ) -> tuple[_typing.Sequence[UMat], UMat]: ... #--> (contours, hierarchy)
'''
contours, _ = cv2.findContours(image=src_bin, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)


# 이진 영상을 컬러 영상 형식으로 변환
# 외곽선을 색별로 구분해서 그리기 위해...
dst = cv2.cvtColor(src_bin, cv2.COLOR_GRAY2BGR)

# 검출된 외곽선(Contour) 모두 그리기
'''
def drawContours(
        image: UMat, contours: _typing.Sequence[UMat], contourIdx: int, color: cv2.typing.Scalar, 
        thickness: int = ..., lineType: int = ..., hierarchy: UMat | None = ..., maxLevel: int = ..., offset: cv2.typing.Point = ...
    ) -> UMat: ...
'''
for i in range(len(contours)):
    c = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    cv2.drawContours(dst, contours, i, c, 2)

# cv2.imshow('src', src)
cv2.imshow('src_bin', src_bin)
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()
