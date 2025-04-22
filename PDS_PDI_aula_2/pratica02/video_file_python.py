#importing libraries
import numpy as np
import math as mt
import cv2 as cv

#initializing connected webcam
#if you have more than 2 cameras
#one will be 0 and the other 1
cap = cv.VideoCapture('tracking_colors.mp4')

#getting video informations
#frames per second
fps = int(cap.get(cv.CAP_PROP_FPS))
#total number of frames
number_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

count = 0
while(count<number_frames):
    #capturing frame-by-frame
    ret, frame = cap.read()

    #you can implement anything you want here

    #showing the resulting frame
    #press "q" to exit
    cv.imshow('frame',frame)
    if cv.waitKey(mt.floor(1000/fps)) & 0xFF == ord('q'):
        break
    count = count+1

#when everything done, release the capture
#if you don't release the capture
#camera will be still blocked
cap.release()
cv.destroyAllWindows()
