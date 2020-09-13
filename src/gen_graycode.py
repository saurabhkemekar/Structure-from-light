import cv2
import  numpy as np

#n = int(input("enter the bit "))
n =10
# this will be graycode fr
l1 = ['0', '1']
l2 = l1.copy()
l2.reverse()

i = len(l1[0]) - 1
if n > 1:
    while i < n:

        for j in range(len(l1)):
            l1[j] = '0' + l1[j]
            l2[j] = '1' + l2[j]

        l1 = l1 + l2
        l2 = l1.copy()
        l2.reverse()
        i = len(l1[0])



for i in range(10):
    image = np.zeros([500,1024],np.uint8)

    for j in range(1024):
        if l1[j][i] == '1':
            image[:,j] = np.array(255,np.uint8)
    cv2.imshow('images',image)
    cv2.waitKey(0)

cv2.destroyAllWindows()
