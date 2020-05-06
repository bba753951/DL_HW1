import cv2
import numpy as np


def hw1_1_1():
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

	w = 8
	h = 11
	objp = np.zeros((w*h,3), np.float32)
	objp[:,:2] = np.mgrid[0:w,0:h].T.reshape(-1,2)
	objpoints = [] # 在世界坐标系中的三维点
	imgpoints = [] # 在图像平面的二维点
	# print(np.mgrid[0:w,0:h].T)
	img_list = [(f'images/CameraCalibration/{num}.bmp') for num in range(1,16)]
	for filename in img_list:
		img = cv2.imread(filename)
		gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


		ret, corners = cv2.findChessboardCorners(gray, (w,h),None)
		if ret == True:
			cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
			objpoints.append(objp)
			imgpoints.append(corners)
			# 将角点在图像上显示
			cv2.drawChessboardCorners(img, (w,h), corners, ret)
			cv2.namedWindow('findCorners', 0)

			cv2.imshow('findCorners',img)
			cv2.waitKey(500)


	cv2.destroyAllWindows()
