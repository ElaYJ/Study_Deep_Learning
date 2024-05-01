import numpy as np
import cv2


img = np.full((400, 400, 3), 255, np.uint8)
#img = cv2.imread('cat.bmp')

cv2.line(img, pt1=(50, 50), pt2=(200, 50), color=(0, 0, 255), thickness=5)
cv2.line(img, (50, 60), (150, 160), (0, 0, 128))

cv2.rectangle(img, rec=(50, 200, 150, 100), color=(0, 255, 0), thickness=2)
#--> rec=(left, top, width, height)
cv2.rectangle(img, pt1=(70, 220), pt2=(180, 280), color=(0, 128, 0), thickness=-1)
#--> pt1=(left, top), pt2=(right, bottom), thickness=-1(내부채움설정)

cv2.circle(img, center=(300, 100), radius=30, color=(255, 255, 0), thickness=-1, lineType=cv2.LINE_AA) #--> AntiAliasing
cv2.circle(img, (300, 100), 60, (255, 0, 0), 3, cv2.LINE_AA)

pts = np.array([[250, 200], [300, 200], [350, 300], [250, 300]])
cv2.polylines(img, [pts], isClosed=True, color=(255, 0, 255), thickness=2)
#--> isClosed=True(폐곡선 여부)

text = 'Hello? OpenCV ' + cv2.__version__
cv2.putText(img, text, (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
#--> def putText(
#           img: UMat, text: str, 
#           org: cv2.typing.Point, (문자열을 출력할 위치의 좌측하단[left, bottom] 좌표, (x, y)튜플)
#           fontFace: int, (cv2.FONT_HERSHEY_ 로 시작하는 상수값)
#           fontScale: float, 
#           color: cv2.typing.Scalar, thickness: int=..., lineType: int=..., 
#           bottomLeftOrigin: bool=...
#       ) -> UMat

cv2.imshow("img", img)
cv2.waitKey()
cv2.destroyAllWindows()

