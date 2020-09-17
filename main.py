import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from Calibration import *
from Decode_pattern import *
import open3d as o3d



                
def arrange_points(points):
    number = np.array([a[0] for a in points])
    x_cen = 0
    y_cen = 0
    for [x, y] in number:
        x_cen = x_cen + x
        y_cen = y_cen + y
    x_cen = x_cen // 4
    y_cen = y_cen // 4
    sorted_x = np.array([[[0, 0]], [[0, 0]], [[0, 0]], [[0, 0]]])
    for [x, y] in number:
        if x <= x_cen and y <= y_cen:
            sorted_x[0][0] = np.array([x, y])
        if x <= x_cen and y >= y_cen:
            sorted_x[1][0] = np.array([x, y])
        if x >= x_cen and y >= y_cen:
            sorted_x[2][0] = np.array([x, y])
        if x >= x_cen and y <= y_cen:
            sorted_x[3][0] = np.array([x, y])
    points = np.array(sorted_x)
    return  points


path = os.getcwd()
mtx_c,dist_c = Camera_calibration(path)
objp = np.zeros((9 * 7, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:7].T.reshape(-1, 2) * 50.33

img = cv2.imread(path+'/sample_data/camera_pose.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,corners = cv2.findChessboardCorners(gray ,(9,7),None)
corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
imgpoints  = np.array(corners,np.float32)
flag,rvec_c,tvec_c = cv2.solvePnP(objp,imgpoints,mtx_c,dist_c)
Hc = np.zeros((3,3))
world_points = []
R,_ = cv2.Rodrigues(rvec_c)
Hc[:,:2] = R[:,:2]
Hc[:,2] = tvec_c.transpose()[0]
Hc = mtx_c @ Hc
inv_Hc = np.linalg.inv(Hc)
print("Homography Matrix ",Hc)

# projector points --- projector resolution was set to 3:4
print("Projector Calibration")
pose = cv2.imread(path+'/sample_data/chess.png')
gray = cv2.cvtColor(pose,cv2.COLOR_BGR2GRAY)
ret,corners = cv2.findChessboardCorners(gray,(15,11),None)
if ret:
        corners = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        corners = order_points(corners)    
        _ = cv2.drawChessboardCorners(pose,(15,11),corners,ret)
        img_points= np.array(corners,np.float32)    # these will be a points in projector


fname = os.listdir(path+ "/sample_data/projector_calibration/")
projector_points = []
world_points = []
for name in fname: 
     point = []
     if name[-4:] == '.png':
        img1 = cv2.imread(path + "/sample_data/projector_calibration/" + name)
        img = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        img = ~img
        blur = cv2.GaussianBlur(img,(5,5),-1)
        img = cv2.addWeighted(img,1.5,blur,-0.5,0,img)
        ret,corners = cv2.findChessboardCorners(img,(15,11),None)
        if ret :
            corners = cv2.cornerSubPix(img,corners,(11,11),(-1,-1),criteria)
            corners = order_points(corners)
            _ = cv2.drawChessboardCorners(img1,(15,11),corners,ret)
            corners = cv2.convertPointsToHomogeneous(corners)
            corners = corners.reshape(-1,3).T            
            a = (inv_Hc @ corners).T
            a[:,0] = a[:,0]/a[:,2]
            a[:,0] = a[:,0]/a[:,2]
            a[:,2] = 0
            world_points.append(a)
            projector_points.append(img_points)
            cv2.imshow('img',img1)
            cv2.waitKey(1)

cv2.destroyAllWindows()
print(np.array(world_points).shape)
print(np.array(projector_points).shape)
ret,mtx_p,dist_p,rvec_p,tvec_p = cv2.calibrateCamera(world_points,projector_points,(800,600),None,None)
print(mtx_p, ret)

img = cv2.imread(path +'/sample_data/projector_pose.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = ~gray
blur = cv2.GaussianBlur(gray,(5,5),-1)
gray = cv2.addWeighted(gray,1.5,blur,-0.5,0,gray)        
ret,corners = cv2.findChessboardCorners(gray ,(15,11),None)
print(ret)
corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
corners = order_points(corners)
imgpoints  = np.array(corners,np.float32)
point= []
for corner in corners:
                b = cv2.convertPointsToHomogeneous(corner)
                b= b.reshape(3,1)
                a = inv_Hc @ b
                point.append(a) 
points = cv2.convertPointsFromHomogeneous(np.array(point)) 
w = np.array([np.array([np.array([i[0][0],i[0][1],0],np.float32) for i in points],np.float32)])
print(w.shape)

flag,rvec_p,tvec_p = cv2.solvePnP(w,imgpoints,mtx_p,dist_p)
H_p = np.zeros((3,4))
world_points = []
R,_ = cv2.Rodrigues(rvec_p)
H_p[:,:3] = R[:,:3]
H_p[:,3] = tvec_p.transpose()[0]
print(H_p)
H_p = mtx_p @ H_p
H_c = np.zeros((3,4))
R,_ = cv2.Rodrigues(rvec_c)
H_c[:,:3] = R[:,:3]
H_c[:,3] = tvec_c.transpose()[0]
print(H_c)
H_c = mtx_c @ H_c

# Decoding the Structure Light
ansy,ansx,mask = decode_graycode(path)

print(ansx.shape[:2])
plt.imshow(ansx)
plt.show()
plt.imshow(ansy)
plt.show()

img = cv2.imread(path+'/sample_data/graycode/41.png')
cy, cx = np.where(mask == 255)
r,g,b = img[cy,cx].reshape(-1,3).T
py, px = ansy[cy, cx], ansx[cy, cx]
camera = np.ones((2,cx.shape[0]))
projector = np.ones((2,cx.shape[0]))
camera[0,:] = cx
camera[1,:] = cy
projector[0,:] = px
projector[1,:] = py
print(H_p.shape, H_c.shape)
points = cv2.triangulatePoints(H_c,H_p,camera,projector)
world_p = cv2.convertPointsFromHomogeneous(points.T.reshape(-1,1,4))        
x,y,z =  world_p.reshape(-1,3).T
print(x,y,z)


# In[39]:



xyzrgb = np.zeros((np.size(x),6))
xyzrgb[:,0] = np.array(x)
xyzrgb[:,1] = np.array(y)
xyzrgb[:,2] = np.array(z)
xyzrgb[:,3] = np.array(r)/255
xyzrgb[:,4] = np.array(g)/255
xyzrgb[:,5] = np.array(b)/255
print(xyzrgb)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyzrgb[:,:3])
pcd.colors = o3d.utility.Vector3dVector(xyzrgb[:,3:])
o3d.io.write_point_cloud(path+'/sample_data/data.ply',pcd)
pcd_load = o3d.io.read_point_cloud(path+'/sample_data/data.ply')
o3d.visualization.draw_geometries([pcd_load])


# In[ ]:




