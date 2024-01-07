import numpy as np
import cv2


cam = cv2.VideoCapture(0)


while(True):
    ret, frame = cam.read()
    hsv_refined = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


    # Apply masks and combine masks
    mask_red_refined = cv2.inRange(hsv_refined, np.array([0, 110, 150]), np.array([20, 255, 255]))
    mask_red2_refined = cv2.inRange(hsv_refined, np.array([150, 50, 20]), np.array([180, 200, 100]))
    red_mask = cv2.bitwise_or(mask_red_refined, mask_red2_refined)

  
    green_mask = cv2.inRange(hsv_refined, np.array([10, 20, 10]), np.array([25, 130, 155])) # brownish grayish light green


    output=cv2.bitwise_and(frame, frame, mask =red_mask)
    combined = np.hstack((frame, output))
    cv2.imshow('frame',combined)

    if cv2.waitKey(1) == 27:
        break


cv2.destroyAllWindows()
cam.release()