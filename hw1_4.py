import cv2

def hw1_4():
	# 直接读取图像数据
	im = cv2.imread('images/Contour.png')
	im = im.copy()

	# 图像灰度处理
	gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

	# 将灰度图进行二进制阈值化
	ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

	# 轮廓检测
	image,contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	# 绘制轮廓线
	cv2.drawContours(im, contours, -1, (0,0,255), 3)

	cv2.imshow('img', im)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
