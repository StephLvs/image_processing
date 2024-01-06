import numpy as np
import cv2
import os
import time

folder_path = 'plates'
binary_folder = 'binary'
files = os.listdir(folder_path)

plate_images = []
binary_images = []
for f in files:
    plate_img = cv2.imread(os.path.join(folder_path, f), 0)
    binary_img = cv2.imread(os.path.join(binary_folder, f), 0)
    if plate_img is not None and binary_img is not None:
        plate_images.append(plate_img)
        binary_images.append(binary_img)

cam = cv2.VideoCapture(0)

while(True):
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect circles in the frame
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=4, minDist=1000, param1=80, param2=40, minRadius=60, maxRadius=200)
    if circles is not None:
        for i in circles[0,:]:
            i = list(map(int, i))
            cv2.circle(frame,(i[0],i[1]),i[2],(0,255,0),2)
            cv2.imshow('Detected',frame)
    if cv2.waitKey(5) == 27:
        break


cv2.destroyAllWindows()
cam.release()