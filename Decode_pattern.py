
def inversegrayCode(n): 
    inv = 0
    while(n): 
        inv = inv ^ n
        n = n >> 1
    return inv

def decode_graycode(path):        

        img1= cv2.imread(path + "/sample_data/graycode/40.png",0)
        img2 = cv2.imread(path + "/sample_data/graycode/41.png",0)
        mask = np.where( img2-img1 <50 ,np.uint8(0),np.uint8(255))
        contour,hiearchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(contour,key=lambda x:cv2.contourArea(x) , reverse = True)[:1]
        approx = None
        for cnt in cnts:
            epsilon = 0.01 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
        approx  = arrange_points(approx)   
        #print(approx) 
        mask = np.zeros(img2.shape,np.uint8)
        mask[approx[0][0][1]:approx[2][0][1],approx[0][0][0]:approx[2][0][0]] = np.uint8(255)        
        ansx = np.zeros(mask.shape,np.uint16)
        ansy = np.zeros(mask.shape,np.uint16) 
        for i in range(0,10):
        	for j in range(2):
                 name1= path + "/sample_data/graycode/"+str(i + j*10)+'.png'
                 rough1 = np.zeros(mask.shape,np.uint8)  
                 name2 = path + "/sample_data/graycode/"+str(i+20+ j*10)+'.png'         
                 img1= cv2.imread(name1,0)
                 img2 =cv2.imread(name2,0)
                 a = mask & img1
                 b = mask & img2
                 rough1 = np.where( a>b ,1,0)  
                 ansx = ansx + (rough1)*(2**(9+ j*10-i))
                 rough1= np.where(a>b,np.uint8(255),np.uint8(0))
                 cv2.imshow('img1',rough1.astype(np.uint8))
                 cv2.imshow('img2',rough2.astype(np.uint8))
                 cv2.waitKey(1)
        cv2.destroyAllWindows()
        for row in range(720):
               for col in range(1280):
                        ansy[row][col] = inversegrayCode(ansy[row][col])
                        ansx[row][col] = inversegrayCode(ansx[row][col])
        return ansx,ansy, mask

