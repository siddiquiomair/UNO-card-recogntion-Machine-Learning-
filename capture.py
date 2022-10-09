# -*- coding: utf-8 -*-
"""
Created on Thu May 12 18:16:56 2022

@author: ANU
"""

import cv2 
count=0
key = cv2. waitKey(1)
webcam = cv2.VideoCapture(0)
while True:
        check, frame = webcam.read()
        #print(check) #prints true as long as the webcam is running
        #print(frame) #prints matrix values of each framecd 
        cv2.imshow("Capturing", frame)
        key = cv2.waitKey(1)
        if key == ord('s'): 
            cv2.imwrite(filename='trainCAM/CARD '+str(count)+'.jpg', img=frame)
            count+=1
          



import cv2 
import numpy as np
from matplotlib import pyplot as plt
from glob import glob
j=1
for rawdataset in glob('circle/*.png'):
    print ('test1')
    img2 = np.zeros([200,200,3],dtype=np.uint8)
    img2.fill(255) 
    img =  cv2.imread(rawdataset)
    cv2.imshow('images1', img) 
    #cv2.imread('rawdataset/circle/circle (1).png')
    #scale_percent = 60 # percent of original size
    #width = int(img.shape[1] * scale_percent / 100)
    #height = int(img.shape[0] * scale_percent / 100)
    #dim = (width, height)
    #resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, threshold = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #print (contours)
    #cv2.drawContours(img, [contours], 0, (0, 0, 0), 5)
    
    #cv2.imshow('images', img)
    
    i = 0
    
    for contour in contours:
        if i == 0:
            i = 1
            continue
        cv2.drawContours(img2, [contour], 0, (0, 0, 0), 3)
        
    cv2.imshow('images3', img)    
    
    cv2.imshow('images2', img2)  

    cv2.imwrite('train_test_data/circle/circle'+str(j)+'.jpg', img2)
    count+=1
cv2. waitKey(0)
cv2.destroyAllWindows()






vc = cv2.VideoCapture(0)  
while vc.isOpened():
    rval, frame = vc.read()    # read video frames again at each loop, as long as the stream is open
    cv2.imshow("stream", frame)# display each frame as an image, "stream" is the name of the window
    key = cv2.waitKey(1)       # allows user intervention without stopping the stream (pause in ms)
    if key == 27:              # exit on ESC
        break
cv2.destroyWindow("stream")    # close image window upon exit
vc.release()     
