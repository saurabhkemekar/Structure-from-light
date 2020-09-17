import cv2
import numpy as np
import os

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def order_points(points):
	len_ = len(points)
	a = points[0][0]
	b = points[len_ - 1][0]
	if a[0] > b[0] and a[1] > b[1]:
		points = points[::-1]
	return points


def Camera_calibration(path):
	fname = os.listdir(path+"/sample_data/camera_calibration/")
	objp = np.zeros((9 * 7, 3), np.float32)
	objp[:, :2] = np.mgrid[0:9, 0:7].T.reshape(-1, 2)*50.33

	objpoints = []  # 3d point in real world space
	imgpoints = []  # 2d points in image plane.
	for name in fname:   
		if name[-4:] == '.png':
			img = cv2.imread(path+"/sample_data/camera_calibration/"+name)

			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			ret, corners = cv2.findChessboardCorners(gray, (9, 7), None)
			if ret == True:
					objpoints.append(objp)
					corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
					corners2 = order_points(corners2)
					imgpoints.append(corners2)
					img = cv2.drawChessboardCorners(img, (9, 7), corners2, ret)
					cv2.imshow('camera', img)
					cv2.waitKey(1)
	cv2.destroyAllWindows()  
	ret,mtx_c,dist_c,rvec_c,tvec_c = cv2.calibrateCamera(objpoints,imgpoints,gray.shape[::-1],None,None)
	print('camera matrix')
	print(mtx_c, ret)
	return mtx_c,dist_c
