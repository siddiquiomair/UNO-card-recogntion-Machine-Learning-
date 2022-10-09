from __future__ import print_function
from imutils.object_detection import non_max_suppression
from PedestrainPatterns.Maskbinarypatterns import Maskbinarypatterns
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import accuracy_score

import imutils
from glob import glob
import cv2


hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    
Ped = Maskbinarypatterns(24, 8)
data1 = []
labels1 = []
data2 = []
labels2 = []


#Training Dataset




for PedestrainPath in glob('*.jpg'):
    image = cv2.imread(PedestrainPath)
    
    scale_percent = 60 # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
   
    filename = PedestrainPath[PedestrainPath.rfind("/") + 1:]

    RGB_gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    GaussianBlur = cv2.GaussianBlur(RGB_gray,(5,5),0)

    ret, otsu = cv2.threshold(GaussianBlur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    MaskBinary1 = Ped.describe(otsu.reshape(1, -1))
    labels1.append(PedestrainPath)
    data1.append(MaskBinary1)
	
    
    
Model = RandomForestClassifier()
Model.fit(data1, labels1) 
print("Model is fitted")




#testing with CAMERA 


import cv2 

key = cv2. waitKey(1)
webcam = cv2.VideoCapture(0)
while True:
        check, frame = webcam.read()
        #print(check) #prints true as long as the webcam is running
        #print(frame) #prints matrix values of each framecd 
        cv2.imshow("Capturing", frame)
        key = cv2.waitKey(1)
        if key == ord('s'): 
            cv2.imwrite(filename='test images/.jpg', img=frame)
            
            image1 = cv2.imread('test images/.jpg')
            scale_percent = 60 # percent of original size
            width=600
            #width = int(image1.shape[1] * scale_percent / 100)
            #height = int(image1.shape[0] * scale_percent / 100)
            height=400
            dim = (width, height)
            resized1 = cv2.resize(image1, dim)
            resized1= cv2.imshow("Captured Image", resized1)
            RGB_gray1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
            GaussianBlur1 = cv2.GaussianBlur(RGB_gray1,(5,5),0)
            ret, otsu1 = cv2.threshold(GaussianBlur1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            MaskBinary2 = Ped.describe(otsu1.reshape(1, -1))
            data2=[]
            data2.append(MaskBinary2)
            prediction = Model.predict(data2) 
            print(prediction)
            
            
webcam.release()
cv2.destroyAllWindows()

"""

#Testing with folder 
 
for Path in glob(r'C:\Users\Admin\Documents\CARD MAIN\UNO CARDS\test images/*.jpg'):
    image1 = cv2.imread(Path)
    #cv2.imshow("m", image1 )
    #cv2.waitKey(0)
    scale_percent = 60 # percent of original size
    width = int(image1.shape[1] * scale_percent / 100)
    height = int(image1.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    resized1 = cv2.resize(image1, dim, interpolation = cv2.INTER_AREA)
    
    filename = Path[Path.rfind("/") + 1:]

    RGB_gray1 = cv2.cvtColor(resized1, cv2.COLOR_RGB2GRAY)
    #cv2.imshow("RGB_gray", RGB_gray1)
    #cv2.waitKey(0)
    
    GaussianBlur1 = cv2.GaussianBlur(RGB_gray1,(5,5),0)
    
    ret, otsu1 = cv2.threshold(GaussianBlur1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #cv2.imshow("mask",otsu1)
    #cv2.waitKey(0)  
    
    MaskBinary2 = Ped.describe(otsu1.reshape(1, -1))

    labels2.append(Path)
    data2=[]
    data2.append(MaskBinary2)
    prediction=0
    prediction = Model.predict(data2) 
    #prediction = np.array(prediction)
    
    print(prediction)

"""




