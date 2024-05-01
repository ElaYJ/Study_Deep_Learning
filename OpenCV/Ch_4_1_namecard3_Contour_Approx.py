import sys
import cv2


# 영상 불러오기
# src = cv2.imread('./_imgs/namecard1.jpg')
src = cv2.imread('./_imgs/namecard2.jpg')

if src is None:
    print('Image load failed!')
    sys.exit()

# 입력 영상을 그레이스케일 영상으로 변환
src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

# 자동 이진화
_, src_bin = cv2.threshold(src_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# 외곽선 검출 및 명함 검출
contours, _ = cv2.findContours(src_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# 외곽선 근사화로 사각형 찾기
for pts in contours:
    # 너무 작은 객체는 무시
    if cv2.contourArea(pts) < 1000:
        continue

    # 외곽선 근사화
    '''
    def approxPolyDP(
            curve: UMat, (입력 곡선 좌표 --> numpy.ndarray. shape=(K, 1, 2).)
            epsilon: float, (근사화 정밀도 조절. 입력 곡선과 근사화 곡선 간의 최대거리. e.g) (외곽선 전체 길이) * 0.02)
            closed: bool, (True를 전달하면 폐곡선으로 간주)
            approxCurve: UMat | None = ... (근사화된 곡선 좌표 --> numpy.ndarray. shape=(k, 1, 2).)
        ) -> UMat: ...
    '''
    approx = cv2.approxPolyDP(curve=pts, epsilon=cv2.arcLength(pts, True)*0.02, closed=True)

    # 사각형으로 근사화되면 외곽선 표시
    
    if len(approx) == 4:
        cv2.polylines(img=src, pts=[approx], isClosed=True, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

cv2.imshow('src', src)
cv2.waitKey()
cv2.destroyAllWindows()
