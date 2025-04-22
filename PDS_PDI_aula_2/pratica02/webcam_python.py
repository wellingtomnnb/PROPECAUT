#importing libraries
import numpy as np
import cv2 as cv

#initializing connected webcam
#if you have more than 2 cameras
#one will be 0 and the other 1
cap = cv.VideoCapture(0)

while(True):
    #capturing frame-by-frame
    ret, frame = cap.read()

    #you can implement anything you want here

    #showing the resulting frame
    #press "q" to exit
    cv.imshow('frame',frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

#when everything done, release the capture
#if you don't release the capture
#camera will be still blocked
cap.release()
cv.destroyAllWindows()